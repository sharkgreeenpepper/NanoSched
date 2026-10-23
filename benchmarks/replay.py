"""Single-process open-loop replay. Works against an untouched upstream worktree."""
import argparse
import csv
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from queue import Queue
import subprocess
import sys
import time
from threading import Thread

import numpy as np
import torch

from benchmarks.metrics import percentiles, summarize


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def workload(name, seed):
    rng = random.Random(seed)
    def request(key, arrival, length, output, group):
        return {"id": key, "arrival_s": arrival, "prompt": [rng.randrange(100, 20000) for _ in range(length)],
                "max_tokens": output, "group": group}
    if name == "competition":
        # The incumbent starts decoding before a sustained stream of distinct long prompts.
        items = [request("incumbent", 0, 128, 128, "incumbent")]
        items += [request(f"long-{i}", 0.05 + i * 0.025, 4096, 16, "long") for i in range(12)]
        items += [request(f"short-{i}", 0.08 + i * 0.04, 128, 64, "short") for i in range(8)]
    elif name == "short":
        items = [request(f"short-{i}", 0, 128, 128, "short") for i in range(8)]
    elif name == "long":
        items = [request(f"long-{i}", 0, 4096, 128, "long") for i in range(4)]
    else:
        raise ValueError(name)
    return sorted(items, key=lambda r: r["arrival_s"])


def arrival_workload(name, seed, input_length, output_length, rate, count):
    """Fixed request set shared by all policies; no completion-driven arrivals."""
    rng = random.Random(seed)
    items = []
    for i in range(count):
        length = input_length
        group = name
        output = output_length
        if name == "mixed" and i % 4 != 0:
            length, group = min(1024, input_length), "short"
        if name == "competition" and i == 0:
            length, output, group = 128, max(256, output_length), "incumbent"
        items.append({"id": f"request-{i}", "arrival_s": i / rate,
                      "prompt": [rng.randrange(100, 20000) for _ in range(length)],
                      "max_tokens": output, "group": group})
    return items


def summarize_timings(requests, steps):
    """Attribute existing batch wall times to requests; these are not GPU kernel times.

    Service includes schedule/postprocess time and is shared by every request in
    a batch. Phase totals count each batch once. Execution excludes the measured
    scheduler/postprocess time, but still includes model host work.
    """
    request_steps = {r["seq_id"]: [] for r in requests}
    phase_steps = {phase: [] for phase in ("prefill", "decode")}
    for step in steps:
        phase_steps[step["phase"]].append(step)
        for seq_id in step["seq_ids"]:
            request_steps[seq_id].append(step)
    fields = ("external_admission_s", "engine_queue_s", "prefill_service_s",
              "prefill_wait_s", "decode_service_s")
    for request in requests:
        scheduled = request_steps[request["seq_id"]]
        first_token = request["token_times"][0]
        prefill = [s for s in scheduled if s["phase"] == "prefill" and s["completed_at"] <= first_token]
        first_start = prefill[0]["scheduled_at"]
        service = sum(s["completed_at"] - s["scheduled_at"] for s in prefill)
        request.update(
            external_admission_s=request["enqueued_at"] - request["arrived_at"],
            engine_queue_s=first_start - request["enqueued_at"],
            prefill_service_s=service,
            prefill_wait_s=max(0, first_token - first_start - service),
            decode_service_s=sum(s["completed_at"] - s["scheduled_at"] for s in scheduled if s["phase"] == "decode"),
        )
    summary = {field: percentiles([r[field] for r in requests]) for field in fields}
    for phase, batches in phase_steps.items():
        summary[f"{phase}_total_step_s"] = sum(s["completed_at"] - s["scheduled_at"] for s in batches)
        summary[f"{phase}_execution_s"] = sum(s["completed_at"] - s["scheduled_at"] - s["schedule_s"] - s["postprocess_s"] for s in batches)
    budgets = [s["budget"] for s in phase_steps["prefill"]]
    summary["prefill_budget_steps"] = dict(Counter(map(str, budgets)))
    summary["prefill_budget_changes"] = sum(a != b for a, b in zip(budgets, budgets[1:]))
    summary["prefill_step_tokens"] = percentiles([s["tokens"] for s in phase_steps["prefill"]])
    summary["decode_batch_size"] = percentiles([len(s["seq_ids"]) for s in phase_steps["decode"]])
    return summary


def kv_snapshot(scheduler, pending_prefills=()):
    """Count unique owned/written blocks; free cached prefixes remain reclaimable.

    A partly written block counts as written. Reserved-unwritten therefore counts
    whole owned blocks beyond actual compute progress, not unused token slots.
    """
    bm = scheduler.block_manager
    seqs = {s.seq_id: s for s in (*scheduler.waiting, *scheduler.running)}
    owned, written = set(), set()
    for seq in seqs.values():
        owned.update(seq.block_table)
        written.update(seq.block_table[:(seq.num_cached_tokens + bm.block_size - 1) // bm.block_size])
    prefills = {s.seq_id for s in scheduler.waiting if s.block_table}
    prefills.update(s.seq_id for s in pending_prefills)
    return {"free_blocks": len(bm.free_block_ids), "used_blocks": len(owned),
            "written_blocks": len(written), "reserved_unwritten_blocks": len(owned - written),
            "active_prefills": len(prefills),
            "active_decodes": sum(s.seq_id not in prefills for s in scheduler.running)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", default="/data1/fangdongqing/infra/models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--policy", choices=["original", "prefill_first", "decode_aware", "adaptive", "adaptive_v2"], default="original")
    parser.add_argument("--workload", choices=["competition", "short", "long", "mixed", "pressure"], default="competition")
    parser.add_argument("--input-length", type=int, default=1024)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--arrival-rate", type=float, help="Requests/second; omit to replay the historical workload")
    parser.add_argument("--num-requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=32, help="Maximum engine-inflight requests; excess arrivals wait externally")
    parser.add_argument("--workload-file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--max-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--decode-limit", type=int, default=1)
    parser.add_argument("--decode-prefill-budget", type=int, default=1024)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--prefill-max-wait", type=float, default=0.5, help="Seconds since the head prefill last received service")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--memory-utilization", type=float, default=0.45)
    parser.add_argument("--kv-blocks", type=int, default=-1, help="Artificial exact KV pool size; -1 uses the GPU memory budget")
    parser.add_argument("--kv-allocation", choices=["full", "incremental"], default="full")
    parser.add_argument("--kv-admission", action="store_true", help="Keep a simple KV reserve for active decodes when admitting prefill")
    parser.add_argument("--ttft-slo", type=float, default=0.5)
    parser.add_argument("--tpot-slo", "--target-tpot", dest="tpot_slo", type=float, default=0.1, help="Seconds; configured objective, not a guaranteed latency")
    parser.add_argument("--profile", action="store_true", help="Separate diagnostic run; excludes profiler runs from comparisons")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup_runs < 0:
        parser.error("repeats must be positive and warmup-runs nonnegative")
    if args.kv_blocks != -1 and args.kv_blocks < 1:
        parser.error("kv-blocks must be -1 or positive")
    if min(args.input_length, args.output_length, args.num_requests, args.concurrency) < 1 or (args.arrival_rate is not None and args.arrival_rate <= 0):
        parser.error("Lengths, request count, concurrency and arrival rate must be positive")
    if args.workload in ("mixed", "pressure") and not args.arrival_rate and not args.workload_file:
        parser.error("mixed/pressure require --arrival-rate or --workload-file")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.engine_root.resolve()))
    from nanovllm import LLM, SamplingParams
    from nanovllm.config import Config
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy = "prefill_first" if args.policy == "original" else args.policy
    config = dict(enforce_eager=args.eager, max_model_len=args.max_model_len, max_num_batched_tokens=args.budget,
                  max_num_seqs=args.max_seqs, gpu_memory_utilization=args.memory_utilization, tensor_parallel_size=1)
    if "scheduler_policy" in Config.__dataclass_fields__:
        config.update(scheduler_policy=policy, max_consecutive_decodes=args.decode_limit,
                      decode_prefill_token_budget=args.decode_prefill_budget,
                      target_tpot_s=args.tpot_slo, prefill_max_wait_s=args.prefill_max_wait,
                      adaptive_chunk_sizes=tuple(args.chunk_sizes))
    elif policy != "prefill_first":
        parser.error("upstream only supports prefill_first")
    if "kv_allocation" in Config.__dataclass_fields__:
        config.update(kv_allocation=args.kv_allocation, kv_admission=args.kv_admission)
    elif args.kv_allocation != "full" or args.kv_admission:
        parser.error("engine does not support KV allocation/admission options")
    if args.kv_blocks > 0:
        config["num_kvcache_blocks"] = args.kv_blocks
    gpu_free_before, gpu_total = torch.cuda.mem_get_info()
    env = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "engine_root": str(args.engine_root.resolve()),
           "commit": git(args.engine_root, "rev-parse", "HEAD"),
           "dirty": bool(git(args.engine_root, "status", "--porcelain", "--untracked-files=no")),
           "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"), "model": args.model,
           "config": config, "seed": args.seed, "temperature": args.temperature, "profile": args.profile,
           "gpu_total_bytes": gpu_total, "gpu_free_bytes_before_init": gpu_free_before,
           "kv_capacity_mode": "artificial_block_cap" if args.kv_blocks > 0 else "gpu_memory_budget",
           "metric_definitions": {"request_mean_tpot_s": "Percentiles across each request's mean inter-token interval",
                                  "token_itl_s": "Percentiles across all token intervals, excluding first tokens",
                                  "slo_goodput_requests_per_s": "Requests meeting both TTFT and request-mean TPOT SLO, divided by replay duration",
                                  "engine_queue_s": "Engine enqueue to first prefill; includes all scheduler and KV admission delay, not KV-only delay",
                                  "kv_reserved_unwritten_blocks": "Unique owned blocks with zero computed tokens; excludes reclaimable free prefix blocks"},
           "benchmark": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    env["warmup_replays"] = args.warmup_runs
    (args.output/"environment.json").write_text(json.dumps(env, indent=2)+"\n")
    t = time.perf_counter()
    engine = LLM(args.model, **config)
    torch.cuda.synchronize()
    env.update(initialization_s=time.perf_counter()-t, gpu_allocated_bytes=torch.cuda.memory_allocated(),
               gpu_reserved_bytes=torch.cuda.memory_reserved(), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
               kv_blocks=len(engine.scheduler.block_manager.blocks),
               kv_allocated_bytes=engine.model_runner.kv_cache.numel() * engine.model_runner.kv_cache.element_size(),
               gpu_free_bytes_after_init=torch.cuda.mem_get_info()[0])
    if args.kv_blocks > 0 and env["kv_blocks"] != args.kv_blocks:
        raise RuntimeError(f"Requested {args.kv_blocks} KV blocks, but engine allocated {env['kv_blocks']}")
    env["weight_load_report"] = getattr(engine.model_runner, "weight_load_report", None)
    (args.output/"environment.json").write_text(json.dumps(env, indent=2)+"\n")
    # Explicit warmup exercises prefill, decode, sampler and graph replay before timing.
    engine.generate([[42+i]*128 for i in range(8)], SamplingParams(temperature=args.temperature, max_tokens=8, ignore_eos=True), use_tqdm=False)
    items = (json.loads(args.workload_file.read_text()) if args.workload_file else
             arrival_workload(args.workload, args.seed, args.input_length, args.output_length, args.arrival_rate, args.num_requests)
             if args.arrival_rate else workload(args.workload, args.seed))
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":"))
    (args.output/"workload.json").write_text(canonical+"\n")
    summaries = []
    for repeat in range(-args.warmup_runs, args.repeats):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Discard cached prefixes from warmup and preceding repeats; preserve the allocated GPU pool.
        from nanovllm.engine.block_manager import BlockManager
        old = engine.scheduler.block_manager
        engine.scheduler.block_manager = BlockManager(len(old.blocks), old.block_size)
        if hasattr(engine.scheduler, "reset_stats"):
            engine.scheduler.reset_stats()
        records, steps = {}, []
        counters = {"preemptions": 0, "recomputed_prefill_tokens": 0, "prefix_cached_tokens": 0}
        high_watermarks = {}
        trace = {"step": None}
        original_schedule, original_postprocess = engine.scheduler.schedule, engine.scheduler.postprocess
        original_preempt = engine.scheduler.preempt
        def preempt(seq):
            counters["preemptions"] += 1
            return original_preempt(seq)
        def schedule():
            uncached = {s.seq_id for s in engine.scheduler.waiting if not s.block_table}
            start = time.perf_counter()
            seqs, is_prefill = original_schedule()
            now = time.perf_counter()
            bm = engine.scheduler.block_manager
            trace["step"] = {"scheduled_at": start-origin, "schedule_s": now-start,
                "phase": "prefill" if is_prefill else "decode", "seq_ids": [s.seq_id for s in seqs],
                "tokens": sum(s.num_scheduled_tokens for s in seqs),
                "prefill_chunks": [s.num_scheduled_tokens for s in seqs] if is_prefill else [],
                "budget": getattr(engine.scheduler, "selected_budget", args.budget),
                "waiting": len(engine.scheduler.waiting), "running": len(engine.scheduler.running),
                "kv_used_blocks": len(bm.used_block_ids), "kv_total_blocks": len(bm.blocks),
                "kv_before_execution": kv_snapshot(engine.scheduler, seqs if is_prefill else ())}
            for seq in seqs:
                end = seq.num_cached_tokens + seq.num_scheduled_tokens
                if is_prefill:
                    counters["recomputed_prefill_tokens"] += max(0, min(end, high_watermarks.get(seq.seq_id, 0))-seq.num_cached_tokens)
                    if seq.seq_id in uncached:
                        counters["prefix_cached_tokens"] += seq.num_cached_tokens
                high_watermarks[seq.seq_id] = max(end, high_watermarks.get(seq.seq_id, 0))
            return seqs, is_prefill
        def postprocess(seqs, token_ids, is_prefill):
            before = [s.num_completion_tokens for s in seqs]
            start = time.perf_counter()
            original_postprocess(seqs, token_ids, is_prefill)
            trace["step"]["postprocess_s"] = time.perf_counter() - start
            now = time.perf_counter()-origin  # .tolist() in ModelRunner has already synchronized GPU results.
            trace["step"]["kv_after_execution"] = kv_snapshot(engine.scheduler)
            for seq, count in zip(seqs, before):
                r = records[seq.seq_id]
                if seq.num_completion_tokens > count:
                    r["token_times"].append(now)
                    r["output_ids"].append(seq.last_token)
                if seq.is_finished:
                    r["finished_at"] = now
            trace["step"]["completed_at"] = now
            steps.append(trace["step"])
        engine.scheduler.schedule, engine.scheduler.postprocess = schedule, postprocess
        engine.scheduler.preempt = preempt
        origin = time.perf_counter()
        arrivals = Queue()
        def produce():
            for item in items:
                time.sleep(max(0, origin + item["arrival_s"] - time.perf_counter()))
                arrivals.put((item, time.perf_counter() - origin))
        producer = Thread(target=produce, daemon=True)
        producer.start()
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) if args.profile and repeat >= 0 else nullcontext()
        with profiler as prof:
            pending = 0
            while pending < len(items) or not engine.is_finished():
                while not arrivals.empty() and len(engine.scheduler.waiting) + len(engine.scheduler.running) < args.concurrency:
                    item, arrived = arrivals.get_nowait()
                    submitted = item["arrival_s"]
                    engine.add_request(item["prompt"], SamplingParams(temperature=args.temperature, max_tokens=item["max_tokens"], ignore_eos=True))
                    seq = engine.scheduler.waiting[-1]
                    records[seq.seq_id] = {"id": item["id"], "group": item["group"], "seq_id": seq.seq_id,
                        "submitted_at": submitted, "arrived_at": arrived, "enqueued_at": time.perf_counter()-origin,
                        "prompt_tokens": len(item["prompt"]), "token_times": [], "output_ids": []}
                    pending += 1
                if not engine.is_finished():
                    engine.step()
                elif pending < len(items):
                    time.sleep(0.001)
        producer.join()
        elapsed = time.perf_counter()-origin
        engine.scheduler.schedule, engine.scheduler.postprocess = original_schedule, original_postprocess
        engine.scheduler.preempt = original_preempt
        if args.profile and repeat >= 0:
            prof.export_chrome_trace(str(args.output/f"profile-{repeat}.json"))
            (args.output/f"profile-{repeat}.txt").write_text(prof.key_averages().table(sort_by="self_device_time_total", row_limit=30))
        rows = list(records.values())
        assert len(rows) == len(items) and all(len(r["token_times"]) == next(i["max_tokens"] for i in items if i["id"] == r["id"]) for r in rows)
        summary = summarize(rows, elapsed, args.ttft_slo, args.tpot_slo)
        summary["request_mean_tpot_s"] = summary["tpot_s"]
        summary["token_itl_s"] = summary["inter_token_gap_s"]
        summary.update(summarize_timings(rows, steps))
        summary["groups"] = {g: summarize([r for r in rows if r["group"] == g], elapsed, args.ttft_slo, args.tpot_slo) for g in sorted({r["group"] for r in rows})}
        summary["max_inter_token_gap_s"] = max((max(np.diff(r["token_times"]), default=0) for r in rows), default=0)
        summary["scheduler_total_s"] = sum(s["schedule_s"] for s in steps)
        summary["scheduler_mean_us"] = summary["scheduler_total_s"] / len(steps) * 1e6
        summary["scheduler_p95_us"] = float(np.percentile([s["schedule_s"] for s in steps], 95)) * 1e6
        summary["postprocess_total_s"] = sum(s["postprocess_s"] for s in steps)
        summary["scheduler_cpu_pct"] = 100 * (summary["scheduler_total_s"] + summary["postprocess_total_s"]) / elapsed
        summary["arrival_lag_p95_s"] = float(np.percentile([r["arrived_at"]-r["submitted_at"] for r in rows], 95))
        summary["admission_wait_p95_s"] = float(np.percentile([r["enqueued_at"]-r["arrived_at"] for r in rows], 95))
        summary["kv_peak_utilization"] = max(s["kv_used_blocks"]/s["kv_total_blocks"] for s in steps)
        kv_states = [s["kv_after_execution"] for s in steps]
        for field in ("free_blocks", "used_blocks", "written_blocks", "reserved_unwritten_blocks", "active_prefills", "active_decodes"):
            values = [s[field] for s in kv_states]
            summary[f"kv_{field}_peak"] = max(values)
            summary[f"kv_{field}_step_mean"] = sum(values) / len(values)
        summary["kv_free_blocks_min"] = min(s["free_blocks"] for s in kv_states)
        summary["kv_pre_execution_reserved_unwritten_blocks_peak"] = max(s["kv_before_execution"]["reserved_unwritten_blocks"] for s in steps)
        summary["kv_written_utilization_step_mean"] = summary["kv_written_blocks_step_mean"] / len(engine.scheduler.block_manager.blocks)
        summary["kv_active_requests_peak"] = max(s["active_prefills"] + s["active_decodes"] for s in kv_states)
        summary["prefill_chunk_tokens"] = dict(Counter(str(n) for s in steps for n in s["prefill_chunks"]))
        summary.update(counters)
        summary["scheduler_stats"] = engine.scheduler.snapshot() if hasattr(engine.scheduler, "snapshot") else None
        if repeat < 0:
            (args.output/f"warmup-trace-{abs(repeat)}.json").write_text(json.dumps({"requests": rows, "steps": steps}, indent=2)+"\n")
            print(json.dumps({"warmup": repeat, "elapsed_s": elapsed}), flush=True)
            continue
        summaries.append(summary)
        (args.output/f"trace-{repeat}.json").write_text(json.dumps({"requests": rows, "steps": steps}, indent=2)+"\n")
        (args.output/"summary.json").write_text(json.dumps({"runs": summaries}, indent=2)+"\n")
        with (args.output/f"requests-{repeat}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (args.output/f"tokens-{repeat}.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["request_id", "token_index", "token_id", "produced_at_s"])
            writer.writerows((r["id"], i, token, at) for r in rows for i, (token, at) in enumerate(zip(r["output_ids"], r["token_times"])))
        print(json.dumps({"repeat": repeat, **summary}), flush=True)
    # Upstream exit is not idempotent; unregister before explicitly releasing this engine.
    import atexit
    atexit.unregister(engine.exit)
    engine.exit()


if __name__ == "__main__":
    main()
