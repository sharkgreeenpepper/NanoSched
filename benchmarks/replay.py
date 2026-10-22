"""Single-process open-loop replay. Works against an untouched upstream worktree."""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from benchmarks.metrics import summarize


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", default="/data1/fangdongqing/infra/models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--policy", choices=["prefill_first", "decode_aware"], default="prefill_first")
    parser.add_argument("--workload", choices=["competition", "short", "long"], default="competition")
    parser.add_argument("--workload-file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--max-seqs", type=int, default=32)
    parser.add_argument("--decode-limit", type=int, default=1)
    parser.add_argument("--decode-prefill-budget", type=int, default=1024)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--memory-utilization", type=float, default=0.45)
    parser.add_argument("--ttft-slo", type=float, default=0.5)
    parser.add_argument("--tpot-slo", type=float, default=0.02)
    parser.add_argument("--profile", action="store_true", help="Separate diagnostic run; excludes profiler runs from comparisons")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup_runs < 0:
        parser.error("repeats must be positive and warmup-runs nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.engine_root.resolve()))
    from nanovllm import LLM, SamplingParams
    from nanovllm.config import Config
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = dict(enforce_eager=args.eager, max_model_len=8192, max_num_batched_tokens=args.budget,
                  max_num_seqs=args.max_seqs, gpu_memory_utilization=args.memory_utilization, tensor_parallel_size=1)
    if "scheduler_policy" in Config.__dataclass_fields__:
        config.update(scheduler_policy=args.policy, max_consecutive_decodes=args.decode_limit,
                      decode_prefill_token_budget=args.decode_prefill_budget)
    elif args.policy != "prefill_first":
        parser.error("upstream only supports prefill_first")
    env = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "engine_root": str(args.engine_root.resolve()),
           "commit": git(args.engine_root, "rev-parse", "HEAD"),
           "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"), "model": args.model,
           "config": config, "seed": args.seed, "temperature": args.temperature, "profile": args.profile}
    env["warmup_replays"] = args.warmup_runs
    (args.output/"environment.json").write_text(json.dumps(env, indent=2)+"\n")
    t = time.perf_counter()
    engine = LLM(args.model, **config)
    torch.cuda.synchronize()
    env.update(initialization_s=time.perf_counter()-t, gpu_allocated_bytes=torch.cuda.memory_allocated(),
               gpu_reserved_bytes=torch.cuda.memory_reserved(), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
               kv_blocks=engine.scheduler.block_manager.blocks.__len__())
    env["weight_load_report"] = getattr(engine.model_runner, "weight_load_report", None)
    (args.output/"environment.json").write_text(json.dumps(env, indent=2)+"\n")
    # Explicit warmup exercises prefill, decode, sampler and graph replay before timing.
    engine.generate([[42+i]*128 for i in range(8)], SamplingParams(temperature=args.temperature, max_tokens=8, ignore_eos=True), use_tqdm=False)
    items = json.loads(args.workload_file.read_text()) if args.workload_file else workload(args.workload, args.seed)
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
            start = time.perf_counter()
            uncached = {s.seq_id for s in engine.scheduler.waiting if not s.block_table}
            seqs, is_prefill = original_schedule()
            now = time.perf_counter()
            bm = engine.scheduler.block_manager
            trace["step"] = {"scheduled_at": start-origin, "schedule_s": now-start,
                "phase": "prefill" if is_prefill else "decode", "seq_ids": [s.seq_id for s in seqs],
                "tokens": sum(s.num_scheduled_tokens for s in seqs),
                "prefill_chunks": [s.num_scheduled_tokens for s in seqs] if is_prefill else [],
                "waiting": len(engine.scheduler.waiting), "running": len(engine.scheduler.running),
                "kv_used_blocks": len(bm.used_block_ids), "kv_total_blocks": len(bm.blocks)}
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
            original_postprocess(seqs, token_ids, is_prefill)
            now = time.perf_counter()-origin  # .tolist() in ModelRunner has already synchronized GPU results.
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
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) if args.profile and repeat >= 0 else nullcontext()
        with profiler as prof:
            pending = 0
            while pending < len(items) or not engine.is_finished():
                now = time.perf_counter()-origin
                while pending < len(items) and items[pending]["arrival_s"] <= now:
                    item = items[pending]
                    submitted = item["arrival_s"]
                    engine.add_request(item["prompt"], SamplingParams(temperature=args.temperature, max_tokens=item["max_tokens"], ignore_eos=True))
                    seq = engine.scheduler.waiting[-1]
                    records[seq.seq_id] = {"id": item["id"], "group": item["group"], "seq_id": seq.seq_id,
                        "submitted_at": submitted, "enqueued_at": time.perf_counter()-origin,
                        "prompt_tokens": len(item["prompt"]), "token_times": [], "output_ids": []}
                    pending += 1
                if not engine.is_finished():
                    engine.step()
                elif pending < len(items):
                    time.sleep(min(0.001, max(0, items[pending]["arrival_s"]-(time.perf_counter()-origin))))
        elapsed = time.perf_counter()-origin
        engine.scheduler.schedule, engine.scheduler.postprocess = original_schedule, original_postprocess
        engine.scheduler.preempt = original_preempt
        if args.profile and repeat >= 0:
            prof.export_chrome_trace(str(args.output/f"profile-{repeat}.json"))
            (args.output/f"profile-{repeat}.txt").write_text(prof.key_averages().table(sort_by="self_device_time_total", row_limit=30))
        rows = list(records.values())
        assert len(rows) == len(items) and all(len(r["token_times"]) == next(i["max_tokens"] for i in items if i["id"] == r["id"]) for r in rows)
        summary = summarize(rows, elapsed, args.ttft_slo, args.tpot_slo)
        summary["groups"] = {g: summarize([r for r in rows if r["group"] == g], elapsed, args.ttft_slo, args.tpot_slo) for g in sorted({r["group"] for r in rows})}
        summary["max_inter_token_gap_s"] = max((max(np.diff(r["token_times"]), default=0) for r in rows), default=0)
        summary["scheduler_total_s"] = sum(s["schedule_s"] for s in steps)
        summary["kv_peak_utilization"] = max(s["kv_used_blocks"]/s["kv_total_blocks"] for s in steps)
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
        print(json.dumps({"repeat": repeat, **summary}), flush=True)
    # Upstream exit is not idempotent; unregister before explicitly releasing this engine.
    import atexit
    atexit.unregister(engine.exit)
    engine.exit()


if __name__ == "__main__":
    main()
