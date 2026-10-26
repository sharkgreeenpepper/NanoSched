"""Real HTTP validation of streaming, concurrent arrivals, cancellation and KV release."""
import argparse
import asyncio
import json
from pathlib import Path
import random
import time

import aiohttp

from benchmarks.online import checked_json, health, replay, sse_events, stream_request


async def wait_idle(session, url, timeout=30):
    deadline = time.perf_counter() + timeout
    while True:
        state = await health(session, url)
        if all(state[key] == 0 for key in ("active", "waiting", "running", "kv_used")):
            assert state["kv_free"] == state["kv_total"], "KV free list did not recover"
            return state
        if time.perf_counter() >= deadline:
            raise AssertionError(f"Server did not release all requests/KV blocks: {state}")
        await asyncio.sleep(0.1)


async def run(args):
    results = []
    prefix = f"validate-{time.time_ns()}"
    serial = 0

    def body(prompt, max_tokens=32, ignore_eos=True):
        nonlocal serial
        serial += 1
        return {"prompt": prompt, "temperature": 0, "max_tokens": max_tokens,
                "ignore_eos": ignore_eos, "request_id": f"{prefix}-{serial}"}

    def record(check, **details):
        row = {"check": check, "passed": True, **details}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600),
                                         connector=aiohttp.TCPConnector(limit=0)) as session:
            await wait_idle(session, args.url)
            prompt = "请用中文介绍北京："
            streamed = await stream_request(session, args.url, body(prompt, 64))
            events, done = streamed["events"], streamed["done"]
            gaps = [b["received_at"] - a["received_at"] for a, b in zip(events, events[1:])]
            assert len(events) == 64
            assert events[0]["received_at"] < done["timing"]["finished_at"], "First token was delivered only after model completion"
            assert sum(gap > 0.001 for gap in gaps) >= 2, "Response arrived as a buffered completion, not a token stream"
            assert "\ufffd" not in done["text"], "Final text contains broken UTF-8 decoding"
            assert all("\ufffd" not in event["text"] for event in events), "A streamed delta exposed an incomplete Unicode character"
            assert any("\u4e00" <= char <= "\u9fff" for char in done["text"]), "Chinese request produced no Chinese text"
            record("streaming_and_chinese_utf8", tokens=len(events), first_received_at=events[0]["received_at"],
                   server_finished_at=done["timing"]["finished_at"], nonzero_stream_gaps=sum(gap > 0.001 for gap in gaps))
            async with session.post(args.url.rstrip("/") + "/generate", json={**body(prompt, 64), "stream": False}) as response:
                nonstream = await checked_json(response)
            assert nonstream["token_ids"] == done["token_ids"], "Streaming changed generated token IDs"
            assert nonstream["text"] == done["text"], "Streaming changed final text"
            await wait_idle(session, args.url)
            record("stream_nonstream_equality", tokens=len(done["token_ids"]))

            if args.offline_baseline:
                baseline = json.loads(args.offline_baseline.read_text())
                cases = baseline["cases"] if isinstance(baseline, dict) else baseline
                for case in cases:
                    generated = await stream_request(session, args.url, body(case["prompt"], case["max_tokens"], case.get("ignore_eos", False)))
                    assert generated["done"]["token_ids"] == case["token_ids"], f"Offline/online token mismatch: {case['id']}"
                    record("offline_online_token_equality", case=case["id"], tokens=len(case["token_ids"]))
                await wait_idle(session, args.url)

            for concurrency in args.concurrencies:
                generated = await asyncio.gather(*(stream_request(session, args.url, body("Count from one to ten:", 32))
                                                   for _ in range(concurrency)))
                assert all(len(result["events"]) == 32 for result in generated), "Concurrent request lost tokens"
                assert len({result["done"]["request_id"] for result in generated}) == concurrency
                state = await wait_idle(session, args.url)
                record("concurrent_completion", requests=concurrency, kv_used_after=state["kv_used"])

            rng = random.Random(20261023)
            items = [{"id": f"join-{i}", "arrival_s": (i // 4) * 0.05,
                      "prompt": [rng.randrange(100, 20000) for _ in range(args.long_input_length if i % 4 == 0 else 128)],
                      "max_tokens": 32} for i in range(12)]
            joined = await replay(session, [args.url], items, 16, prefix + "-join")
            assert not joined["errors"], joined["errors"]
            assert len(joined["requests"]) == len(items)
            await wait_idle(session, args.url)
            record("dynamic_join_long_short", requests=len(items), elapsed_s=joined["elapsed_s"])

            first_token = asyncio.Event()
            async def received_token(event):
                first_token.set()
            cancelled_body = body("Write a long story about a library:", 256)
            task = asyncio.create_task(stream_request(session, args.url, cancelled_body, received_token))
            await asyncio.wait_for(first_token.wait(), 30)
            async with session.delete(args.url.rstrip("/") + "/requests/" + cancelled_body["request_id"]) as response:
                cancelled = await checked_json(response)
            assert cancelled["cancelled"]
            result = await task
            assert result["done"]["finish_reason"] == "cancelled"
            assert len(result["events"]) < 256
            state = await wait_idle(session, args.url)
            record("explicit_cancellation", emitted_tokens=len(result["events"]), kv_used_after=state["kv_used"])

            disconnected_body = body("Write a long story about a river:", 256)
            saw_token = False
            async with session.post(args.url.rstrip("/") + "/generate", json={**disconnected_body, "stream": True}) as response:
                if response.status >= 400:
                    await checked_json(response)
                async for event, data, received_at in sse_events(response):
                    if event == "token":
                        saw_token = True
                        break
            assert saw_token, "Disconnect check never received a token"
            state = await wait_idle(session, args.url)
            record("disconnect_releases_request", kv_used_after=state["kv_used"])
    except Exception as error:
        results.append({"check": "validation_failure", "passed": False, "error": repr(error)})
        raise
    finally:
        args.output.write_text(json.dumps({"clock": "same-host perf_counter", "checks": results}, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline-baseline", type=Path)
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--long-input-length", type=int, default=8192)
    args = parser.parse_args()
    if min(*args.concurrencies, args.long_input_length) < 1:
        parser.error("Concurrency and prompt length must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
