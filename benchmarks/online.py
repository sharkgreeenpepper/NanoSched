"""Open-loop HTTP/SSE replay; cross-process latency attribution assumes one host."""
import argparse
import asyncio
import csv
import json
from pathlib import Path
import random
import time

import aiohttp

from benchmarks.metrics import percentiles, summarize


async def sse_events(response):
    event, data = "message", []
    async for raw in response.content:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield event, json.loads("\n".join(data)), time.perf_counter()
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event, json.loads("\n".join(data)), time.perf_counter()


async def checked_json(response):
    if response.status >= 400:
        raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:1000]}")
    return await response.json()


async def health(session, url):
    async with session.get(url.rstrip("/") + "/health") as response:
        return await checked_json(response)


async def stream_request(session, url, body, on_token=None):
    sent_at = time.perf_counter()
    tokens, done = [], None
    async with session.post(url.rstrip("/") + "/generate", json={**body, "stream": True}) as response:
        if response.status >= 400:
            await checked_json(response)
        headers_at = time.perf_counter()
        async for event, data, received_at in sse_events(response):
            if event == "token":
                if data["request_id"] != body["request_id"] or data["index"] != len(tokens):
                    raise AssertionError("SSE request identity or token order changed")
                tokens.append({**data, "received_at": received_at})
                if on_token:
                    await on_token(data)
            elif event == "done":
                done = {**data, "received_at": received_at}
            elif event == "error":
                raise RuntimeError(f"SSE error: {data}")
    if done is None:
        raise RuntimeError(f"SSE ended without done: {body['request_id']}")
    if done["request_id"] != body["request_id"]:
        raise AssertionError("SSE completion belongs to another request")
    if done["finish_reason"] == "error":
        raise RuntimeError(f"Inference failed: {done.get('error')}")
    if [token["token_id"] for token in tokens] != done["token_ids"]:
        raise AssertionError("SSE tokens differ from final token_ids")
    if "".join(token["text"] for token in tokens) + done.get("text_delta", "") != done["text"]:
        raise AssertionError("SSE text deltas differ from final text")
    return {"sent_at": sent_at, "headers_at": headers_at, "events": tokens, "done": done}


def make_workload(args):
    rng = random.Random(args.seed)
    items = []
    for i in range(args.num_requests):
        length = min(1024, args.input_length) if args.workload == "mixed" and i % 4 else args.input_length
        items.append({"id": f"request-{i}", "arrival_s": (i // args.burst_size) * args.burst_size / args.arrival_rate,
                      "prompt": [rng.randrange(100, 20000) for _ in range(length)],
                      "max_tokens": args.output_length, "group": "short" if length < args.input_length else "long"})
    return items


def warmup_workload(items, repeat):
    """Exercise the same lengths without warming the measured prefix hashes."""
    result = []
    for item in items:
        prompt = item["prompt"]
        if isinstance(prompt, list):
            prompt = [(prompt[0] + 1 + abs(repeat)) % 20000, *prompt[1:]]
        else:
            prompt = f"Warmup {repeat}:\n" + prompt
        result.append({**item, "prompt": prompt})
    return result


async def replay(session, urls, items, concurrency, prefix):
    semaphore = asyncio.Semaphore(concurrency)
    origin = time.perf_counter()

    async def request(index, item):
        await asyncio.sleep(max(0, origin + item["arrival_s"] - time.perf_counter()))
        ready_at = time.perf_counter()
        url = urls[index % len(urls)]
        body = {"prompt": item["prompt"], "temperature": 0,
                "max_tokens": item["max_tokens"], "ignore_eos": item.get("ignore_eos", True),
                "request_id": f"{prefix}-{item['id']}"}
        async with semaphore:
            result = await stream_request(session, url, body)
        if body["ignore_eos"] and len(result["events"]) != item["max_tokens"]:
            raise AssertionError(f"Incomplete response for {body['request_id']}: {result['done']['finish_reason']}")
        return {"id": item["id"], "request_id": body["request_id"], "url": url,
                "group": item.get("group", "all"), "submitted_at": item["arrival_s"],
                "planned_at": origin + item["arrival_s"], "ready_at": ready_at,
                "token_times": [t["received_at"] - origin for t in result["events"]],
                "finished_at": result["done"]["received_at"] - origin, **result}

    results = await asyncio.gather(*(request(i, item) for i, item in enumerate(items)), return_exceptions=True)
    elapsed = time.perf_counter() - origin
    errors = [{"id": item["id"], "error": repr(result)} for item, result in zip(items, results) if isinstance(result, BaseException)]
    rows = [result for result in results if not isinstance(result, BaseException)]
    return {"origin": origin, "elapsed_s": elapsed, "requests": rows, "errors": errors}


def summarize_online(trace, ttft_slo, tpot_slo):
    rows, origin = trace["requests"], trace["origin"]
    model_rows = [{**row, "token_times": [event["produced_at"] - origin for event in row["events"]],
                   "finished_at": row["events"][-1]["produced_at"] - origin} for row in rows]
    result = {"client": summarize(rows, trace["elapsed_s"], ttft_slo, tpot_slo),
              "model": summarize(model_rows, trace["elapsed_s"], ttft_slo, tpot_slo)}
    for metrics in result.values():
        metrics["request_mean_tpot_s"] = metrics["tpot_s"]
        metrics["token_itl_s"] = metrics["inter_token_gap_s"]
    result["latency_attribution_s"] = {
        "arrival_lag": percentiles([r["ready_at"] - r["planned_at"] for r in rows]),
        "client_concurrency_wait": percentiles([r["sent_at"] - r["ready_at"] for r in rows]),
        "client_queue": percentiles([r["sent_at"] - r["planned_at"] for r in rows]),
        "dispatch_transport": percentiles([r["done"]["timing"]["queued_at"] - r["sent_at"] for r in rows]),
        "server_queue": percentiles([r["done"]["timing"]["admitted_at"] - r["done"]["timing"]["queued_at"] for r in rows]),
        "server_admitted_to_first_token": percentiles([r["done"]["timing"]["first_token"] - r["done"]["timing"]["admitted_at"] for r in rows]),
        "produced_to_received": percentiles([e["received_at"] - e["produced_at"] for r in rows for e in r["events"]]),
        "produced_to_emitted": percentiles([e["emitted_at"] - e["produced_at"] for r in rows for e in r["events"]]),
        "emitted_to_received": percentiles([e["received_at"] - e["emitted_at"] for r in rows for e in r["events"]]),
    }
    return result


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    items = json.loads(args.workload_file.read_text()) if args.workload_file else make_workload(args)
    (args.output / "workload.json").write_text(json.dumps(items) + "\n")
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update(timestamp_clock="absolute perf_counter; client/server attribution requires the same host",
                    prefix_cache="retained between repeats; warmup uses a different first token/prefix",
                    metric_definitions={"client": "CPU-visible HTTP token receipt; TTFT starts at planned arrival",
                                        "model": "server model-produced token timestamps; same planned arrivals and replay duration",
                                        "request_mean_tpot_s": "Percentiles across request-mean token intervals",
                                        "token_itl_s": "Percentiles across all token intervals excluding first tokens",
                                        "goodput": "Requests meeting TTFT and request-mean TPOT SLO per second"})
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    summaries = []
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        for repeat in range(-args.warmup_runs, args.repeats):
            before = {url: await health(session, url) for url in args.urls}
            trace = await replay(session, args.urls, warmup_workload(items, repeat) if repeat < 0 else items,
                                 args.concurrency, f"{time.time_ns()}-r{repeat}")
            after = {url: await health(session, url) for url in args.urls}
            trace.update(health_before=before, health_after=after)
            name = f"warmup-{abs(repeat)}" if repeat < 0 else f"trace-{repeat}"
            (args.output / f"{name}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
            if trace["errors"]:
                raise RuntimeError(f"{len(trace['errors'])} HTTP requests failed; see {name}.json")
            summary = summarize_online(trace, args.ttft_slo, args.tpot_slo)
            summary["server_wall_s"] = {field: sum(after[url]["timings"][field] - before[url]["timings"][field] for url in args.urls)
                                        for field in before[args.urls[0]]["timings"]}
            summary["prefix_cache_state"] = "first measured replay; warmup uses distinct prefixes; server cache is not reset" if repeat == 0 else "retained from earlier replay"
            if repeat < 0:
                print(json.dumps({"warmup": repeat, "elapsed_s": trace["elapsed_s"]}), flush=True)
                continue
            summaries.append(summary)
            with (args.output / f"tokens-{repeat}.csv").open("w") as stream:
                writer = csv.writer(stream)
                writer.writerow(["request_id", "index", "token_id", "produced_at", "emitted_at", "received_at"])
                writer.writerows((r["request_id"], e["index"], e["token_id"], e["produced_at"], e["emitted_at"], e["received_at"])
                                 for r in trace["requests"] for e in r["events"])
            (args.output / "summary.json").write_text(json.dumps({"runs": summaries}, indent=2) + "\n")
            print(json.dumps({"repeat": repeat, **summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", "--urls", dest="urls", nargs="+", default=["http://127.0.0.1:8000"])
    parser.add_argument("--workload-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--workload", choices=["uniform", "mixed"], default="uniform")
    parser.add_argument("--input-length", type=int, default=8192)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--arrival-rate", type=float, default=2)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--burst-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--ttft-slo", type=float, default=5)
    parser.add_argument("--tpot-slo", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if min(args.concurrency, args.repeats, args.input_length, args.output_length, args.num_requests,
           args.burst_size, args.arrival_rate, args.timeout) <= 0 or args.warmup_runs < 0:
        parser.error("Lengths, counts, rates and timeout must be positive; warmup-runs must be nonnegative")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
