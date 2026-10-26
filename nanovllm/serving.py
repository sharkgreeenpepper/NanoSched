"""Single-worker online inference: HTTP never operates on the engine or GPU."""
import argparse
import asyncio
from concurrent.futures import Future
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
import json
from queue import Empty, Queue
from threading import Lock, Thread
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, StrictInt, StrictStr
from tokenizers.decoders import DecodeStream

from nanovllm.sampling_params import SamplingParams


class GenerateRequest(BaseModel):
    prompt: StrictStr | list[StrictInt]
    temperature: float = 0
    max_tokens: StrictInt = Field(default=64, ge=1)
    stream: bool = False
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    ignore_eos: bool = False


class ServiceError(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class TextAccumulator:
    """Incrementally decode byte tokens, flushing an incomplete tail at finish."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stream = DecodeStream(skip_special_tokens=True)
        self.token_ids = []
        self.text = ""
        self._fed = 0
        self._finalized = False

    def flush(self, final=False):
        delta = ""
        if self._fed < len(self.token_ids):
            delta = self.stream.step(self.tokenizer.backend_tokenizer, self.token_ids[self._fed:]) or ""
            self._fed = len(self.token_ids)
            self.text += delta
        if final and not self._finalized:
            # DecodeStream buffers incomplete UTF-8 without a flush API. Decode
            # once at completion/cancellation to preserve the exact final text.
            decoded = self.tokenizer.decode(self.token_ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            if not decoded.startswith(self.text):
                raise RuntimeError("Tokenizer revised an already emitted text prefix")
            delta += decoded[len(self.text):]
            self.text = decoded
            self._finalized = True
        return delta


@dataclass
class RequestState:
    request_id: str
    events: asyncio.Queue
    queued_at: float
    seq_id: int = -1
    decoder: TextAccumulator | None = None
    admitted_at: float | None = None
    first_token: float | None = None


class InferenceWorker:
    def __init__(self, engine_factory, max_pending=256, worker_context=None):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.engine_factory = engine_factory
        self.max_pending = max_pending
        self.worker_context = worker_context
        self.commands = Queue()
        self._lock = Lock()
        self._accepting = False
        self._failure = None
        self.thread = None
        self.requests = {}
        self.by_seq = {}
        self.timings = dict.fromkeys(("scheduler_s", "model_runner_s", "postprocess_s", "text_s", "event_s"), 0.0)
        self.steps = 0
        self.generated_tokens = 0

    async def start(self):
        self.loop = asyncio.get_running_loop()
        ready = Future()
        self.thread = Thread(target=self._run, args=(ready,), name="nanosched-inference", daemon=True)
        self.thread.start()
        await asyncio.wrap_future(ready)

    def _enqueue(self, command, args, future):
        with self._lock:
            if not self._accepting:
                raise ServiceError(503, self._failure or "Inference worker is not running")
            self.commands.put((command, args, future))

    async def command(self, command, *args):
        future = Future()
        self._enqueue(command, args, future)
        return await asyncio.wrap_future(future)

    async def submit(self, body, request_id, queued_at):
        state = RequestState(request_id, asyncio.Queue(), queued_at)
        try:
            await self.command("submit", body, state)
        except asyncio.CancelledError:
            self.cancel_nowait(request_id, state)
            raise
        return state

    def cancel_nowait(self, request_id, state=None):
        # Enqueue before an HTTP cancellation scope can cancel another await.
        with self._lock:
            if self._accepting:
                self.commands.put(("cancel", (request_id, state), None))

    async def stop(self):
        if self.thread is None:
            return
        with self._lock:
            accepting = self._accepting
        if accepting:
            await self.command("shutdown")
        await asyncio.to_thread(self.thread.join)

    def _emit(self, state, event, data):
        started = perf_counter()
        def deliver():
            data["emitted_at"] = perf_counter()
            state.events.put_nowait((event, data))
        self.loop.call_soon_threadsafe(deliver)
        self.timings["event_s"] += perf_counter() - started

    def _finish(self, state, reason, error=None):
        started = perf_counter()
        tail = state.decoder.flush(final=True) if error is None else ""
        self.timings["text_s"] += perf_counter() - started
        data = {"request_id": state.request_id, "token_ids": list(state.decoder.token_ids),
                "text": state.decoder.text, "text_delta": tail, "finish_reason": reason,
                "timing": {"queued_at": state.queued_at, "admitted_at": state.admitted_at,
                           "first_token": state.first_token, "finished_at": perf_counter()}}
        if error is not None:
            data["error"] = error
        self._emit(state, "done", data)
        del self.requests[state.request_id]
        del self.by_seq[state.seq_id]

    def _cancel(self, request_id, expected_state=None):
        state = self.requests.get(request_id)
        if state is None or expected_state is not None and state is not expected_state:
            return False
        self.engine.cancel_request(state.seq_id)
        self._finish(state, "cancelled")
        return True

    def _health(self):
        scheduler = self.engine.scheduler
        bm = scheduler.block_manager
        return {"status": "ok", "active": len(self.requests), "waiting": len(scheduler.waiting),
                "running": len(scheduler.running), "kv_used": len(bm.used_block_ids),
                "kv_free": len(bm.free_block_ids), "kv_total": len(bm.blocks),
                "steps": self.steps, "generated_tokens": self.generated_tokens,
                "timings": dict(self.timings),
                "timing_scope": "Cumulative worker CPU wall time; model_runner includes GPU synchronization; event is enqueue time"}

    def _dispatch(self, command, args):
        if command == "submit":
            body, state = args
            if state.request_id in self.requests:
                raise ServiceError(409, "request_id is already active")
            if len(self.requests) >= self.max_pending:
                raise ServiceError(429, "Inference queue is full")
            params = SamplingParams(temperature=body.temperature, max_tokens=body.max_tokens,
                                    ignore_eos=body.ignore_eos)
            state.seq_id = self.engine.add_request(body.prompt, params)
            state.decoder = TextAccumulator(self.engine.tokenizer)
            self.requests[state.request_id] = state
            self.by_seq[state.seq_id] = state
            return None
        if command == "cancel":
            return self._cancel(*args)
        if command == "health":
            return self._health()
        if command == "shutdown":
            for request_id in list(self.requests):
                self._cancel(request_id)
            return None
        raise ValueError(f"Unknown worker command: {command}")

    def _step(self):
        result = self.engine.step_with_events()
        self.steps += 1
        for key, value in result["timings"].items():
            self.timings[key] += value
        for seq_id in result["scheduled"]:
            state = self.by_seq[seq_id]
            if state.admitted_at is None:
                state.admitted_at = result["scheduled_at"]
        for token in result["tokens"]:
            state = self.by_seq[token["seq_id"]]
            if state.first_token is None:
                state.first_token = token["produced_at"]
            started = perf_counter()
            state.decoder.token_ids.append(token["token_id"])
            delta = state.decoder.flush(final=token["finish_reason"] is not None)
            self.timings["text_s"] += perf_counter() - started
            self.generated_tokens += 1
            self._emit(state, "token", {"request_id": state.request_id, "token_id": token["token_id"],
                                       "text": delta, "index": token["index"],
                                       "produced_at": token["produced_at"]})
            if token["finish_reason"] is not None:
                self._finish(state, token["finish_reason"])

    def _run(self, ready):
        self.engine = None
        try:
            self.engine = self.engine_factory()
            context = self.worker_context(self.engine) if self.worker_context else nullcontext()
            with context:
                with self._lock:
                    self._accepting = True
                ready.set_result(None)
                stopping = False
                while not stopping:
                    # Limit command service so sustained submissions still allow GPU steps.
                    for _ in range(128):
                        try:
                            item = self.commands.get(block=not self.requests) if _ == 0 else self.commands.get_nowait()
                        except Empty:
                            break
                        command, args, future = item
                        if future is not None and not future.set_running_or_notify_cancel():
                            continue
                        try:
                            result = self._dispatch(command, args)
                        except (ValueError, ServiceError) as error:
                            if future is not None:
                                future.set_exception(error)
                        except BaseException as error:
                            if future is not None:
                                future.set_exception(error)
                            raise
                        else:
                            if future is not None:
                                future.set_result(result)
                        if command == "shutdown":
                            stopping = True
                            break
                    if self.requests and not stopping:
                        self._step()
        except BaseException as error:
            self._failure = f"{type(error).__name__}: {error}"
            if not ready.done():
                ready.set_exception(error)
            for state in list(self.requests.values()):
                self._finish(state, "error", self._failure)
        finally:
            with self._lock:
                self._accepting = False
            while True:
                try:
                    _, _, future = self.commands.get_nowait()
                except Empty:
                    break
                if future is not None and future.set_running_or_notify_cancel():
                    future.set_exception(ServiceError(503, self._failure or "Inference worker stopped"))
            if self.engine is not None:
                self.engine.exit()


def create_app(engine_factory=None, *, model=None, engine_kwargs=None, max_pending=256, worker_context=None):
    if engine_factory is None:
        def engine_factory():
            from nanovllm import LLM
            return LLM(model, **(engine_kwargs or {}))
    worker = InferenceWorker(engine_factory, max_pending, worker_context)

    @asynccontextmanager
    async def lifespan(app):
        await worker.start()
        try:
            yield
        finally:
            await worker.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.worker = worker

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        return JSONResponse({"detail": error.detail}, status_code=error.status_code)

    @app.get("/health")
    async def health():
        return await worker.command("health")

    @app.delete("/requests/{request_id}")
    async def cancel(request_id: str):
        if not await worker.command("cancel", request_id):
            raise HTTPException(404, "Unknown active request")
        return {"request_id": request_id, "cancelled": True}

    @app.post("/generate")
    async def generate(body: GenerateRequest, request: Request):
        request_id = body.request_id or uuid4().hex
        try:
            state = await worker.submit(body, request_id, perf_counter())
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

        async def events():
            try:
                while True:
                    event, data = await state.events.get()
                    yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                    if event == "done":
                        break
            finally:
                worker.cancel_nowait(request_id, state)

        if body.stream:
            return StreamingResponse(events(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        try:
            while True:
                if await request.is_disconnected():
                    raise HTTPException(499, "Client disconnected")
                try:
                    event, data = await asyncio.wait_for(state.events.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                if event == "done":
                    return JSONResponse(data, status_code=500 if data["finish_reason"] == "error" else 200)
        finally:
            worker.cancel_nowait(request_id, state)

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy", choices=("original", "decode_aware", "adaptive_v2"), default="decode_aware")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--max-seqs", type=int, default=16)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--decode-limit", type=int, default=3)
    parser.add_argument("--decode-prefill-budget", type=int, default=2048)
    parser.add_argument("--memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--kv-blocks", type=int, default=-1)
    parser.add_argument("--max-pending", type=int, default=256)
    parser.add_argument("--profile-path")
    parser.add_argument("--profile-steps", type=int, default=32)
    args = parser.parse_args()
    engine_kwargs = {"scheduler_policy": "prefill_first" if args.policy == "original" else args.policy,
                     "tensor_parallel_size": args.tp, "enforce_eager": args.eager,
                     "max_num_seqs": args.max_seqs, "max_num_batched_tokens": args.budget,
                     "max_consecutive_decodes": args.decode_limit,
                     "decode_prefill_token_budget": args.decode_prefill_budget,
                     "gpu_memory_utilization": args.memory_utilization,
                     "max_model_len": args.max_model_len, "num_kvcache_blocks": args.kv_blocks}
    context = None
    if args.profile_path:
        from benchmarks.profile_serving import serving_profile
        context = lambda engine: serving_profile(engine, args.profile_path, args.profile_steps)
    app = create_app(model=args.model, engine_kwargs=engine_kwargs, max_pending=args.max_pending,
                     worker_context=context)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
