"""CPU checks for the worker contract using the real scheduler and step path."""
import asyncio
import json
from threading import get_ident
from time import perf_counter, sleep
from types import SimpleNamespace

import httpx
import pytest
from tokenizers import Tokenizer, decoders, models

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams
from nanovllm.serving import GenerateRequest, InferenceWorker, ServiceError, TextAccumulator, create_app


class ByteTokenizer:
    # Three model tokens form one UTF-8 character; EOS is omitted from text.
    def __init__(self):
        self.backend_tokenizer = Tokenizer(models.WordLevel(
            {"<0xE4>": 20, "<0xBD>": 21, "<0xA0>": 22, "!": 23, "<eos>": 99}))
        self.backend_tokenizer.decoder = decoders.ByteFallback()
        self.backend_tokenizer.add_special_tokens(["<eos>"])
        self.decode_calls = 0

    def encode(self, text):
        return [1] * len(text)

    def decode(self, tokens, **kwargs):
        self.decode_calls += 1
        return self.backend_tokenizer.decode(tokens, skip_special_tokens=True)


@pytest.mark.parametrize("tokens, expected", [([20, 21, 22, 23, 99], "你!"), ([20], "\ufffd")])
def test_incremental_text_and_final_utf8_flush(tokens, expected):
    tokenizer = ByteTokenizer()
    decoder = TextAccumulator(tokenizer)
    chunks = []
    for token in tokens:
        decoder.token_ids.append(token)
        chunks.append(decoder.flush())
    assert tokenizer.decode_calls == 0
    chunks.append(decoder.flush(final=True))
    assert "".join(chunks) == decoder.text == expected
    assert decoder.flush(final=True) == ""
    assert tokenizer.decode_calls == 1


def engine_factory(config, *, tokens=(20, 21, 22, 23, 99), delay=0):
    config.hf_config = SimpleNamespace(vocab_size=100)
    config.max_model_len = 256
    config.eos = 99
    calls = []

    class FakeRunner:
        def call(self, method, seqs=None, is_prefill=None):
            calls.append((method, get_ident()))
            if delay:
                sleep(delay)
            if method == "run":
                return [tokens[s.num_completion_tokens % len(tokens)] for s in seqs]

    def build():
        calls.append(("init", get_ident()))
        engine = LLMEngine.__new__(LLMEngine)
        engine.config = config
        engine._closed = False
        engine._rendezvous_dir = None
        engine.ps = []
        engine.scheduler = Scheduler(config)
        engine.model_runner = FakeRunner()
        engine.tokenizer = ByteTokenizer()
        return engine

    return build, calls


def parse_events(text):
    result = []
    for block in text.strip().split("\n\n"):
        event, data = block.split("\n", 1)
        result.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return result


def test_http_stream_utf8_nonstream_validation_and_thread_ownership(config):
    factory, calls = engine_factory(config)
    async def scenario():
        app = create_app(factory)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                response = await client.post("/generate", json={"prompt": [1] * 13, "max_tokens": 8, "stream": True})
                assert response.status_code == 200
                events = parse_events(response.text)
                tokens = [data for event, data in events if event == "token"]
                done = events[-1][1]
                assert [data["token_id"] for data in tokens] == done["token_ids"] == [20, 21, 22, 23, 99]
                assert "".join(data["text"] for data in tokens) == done["text"] == "你!"
                assert [data["index"] for data in tokens] == list(range(5))
                assert done["finish_reason"] == "stop"
                timing = done["timing"]
                assert timing["queued_at"] <= timing["admitted_at"] <= timing["first_token"] <= timing["finished_at"]
                assert all(data["produced_at"] <= data["emitted_at"] for data in tokens)
                final = (await client.post("/generate", json={"prompt": "abc", "max_tokens": 8})).json()
                assert final["token_ids"] == done["token_ids"]
                assert final["text"] == done["text"]
                assert (await client.post("/generate", json={"prompt": [], "max_tokens": 1})).status_code == 400
                assert (await client.post("/generate", json={"prompt": [100], "max_tokens": 1})).status_code == 400
                assert (await client.post("/generate", json={"prompt": [1], "max_tokens": 256})).status_code == 400
                assert (await client.post("/generate", json={"prompt": [True], "max_tokens": 1})).status_code == 422
                health = (await client.get("/health")).json()
                assert health["active"] == health["waiting"] == health["running"] == health["kv_used"] == 0
                assert health["kv_free"] == health["kv_total"]
                assert health["steps"] >= 10
                assert all(value >= 0 for value in health["timings"].values())
    asyncio.run(scenario())
    assert len({thread for _, thread in calls}) == 1
    assert calls[0][1] != get_ident()
    assert calls[-1][0] == "exit"


def test_dynamic_submit_cancel_queue_limit_and_shutdown(config):
    factory, calls = engine_factory(config, tokens=(23,), delay=0.002)

    async def scenario():
        worker = InferenceWorker(factory, max_pending=2)
        await worker.start()
        body = GenerateRequest(prompt=[1] * 20, max_tokens=100, ignore_eos=True)
        first = await worker.submit(body, "first", perf_counter())
        second = await worker.submit(body, "second", perf_counter())
        with pytest.raises(ServiceError) as duplicate:
            await worker.submit(body, "first", perf_counter())
        assert duplicate.value.status_code == 409
        with pytest.raises(ServiceError) as full:
            await worker.submit(body, "third", perf_counter())
        assert full.value.status_code == 429
        assert await worker.command("cancel", "first")
        assert await worker.command("cancel", "second")
        assert not await worker.command("cancel", "unknown")
        for state in (first, second):
            while True:
                event, data = await state.events.get()
                if event == "done":
                    assert data["finish_reason"] == "cancelled"
                    break
        health = await worker.command("health")
        assert health["active"] == health["waiting"] == health["running"] == health["kv_used"] == 0
        # An old connection's cleanup cannot cancel a newly reused request ID.
        current = await worker.submit(body, "first", perf_counter())
        worker.cancel_nowait("first", first)
        assert (await worker.command("health"))["active"] == 1
        await worker.stop()
        while True:
            event, data = await current.events.get()
            if event == "done":
                assert data["finish_reason"] == "cancelled"
                break
        assert not worker.thread.is_alive()

    asyncio.run(scenario())
    assert calls[-1][0] == "exit"


@pytest.mark.parametrize("phase", ["queued", "partial", "decode"])
def test_cancel_releases_waiting_partial_and_decode_blocks(config, phase):
    factory, _ = engine_factory(config)
    engine = factory()
    seq_id = engine.add_request([1] * 13, SamplingParams(temperature=0, max_tokens=8))
    if phase == "partial":
        engine.step()
    if phase == "decode":
        while not engine.scheduler.running:
            engine.step()
    assert engine.cancel_request(seq_id)
    assert not engine.cancel_request(seq_id)
    assert engine.is_finished()
    bm = engine.scheduler.block_manager
    assert not bm.used_block_ids
    assert all(block.ref_count == 0 for block in bm.blocks)


def test_reject_final_kv_overflow_but_allow_uncached_final_token(config):
    config.num_kvcache_blocks = 2
    factory, _ = engine_factory(config)
    engine = factory()
    with pytest.raises(ValueError, match="KV pool"):
        engine.add_request([1] * 4, SamplingParams(max_tokens=6))
    engine.add_request([1] * 4, SamplingParams(max_tokens=5))
    assert len(engine.scheduler.waiting) == 1
