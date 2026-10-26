"""Bounded profiles of the real engine, including its inference worker thread.

Use serving_profile as the HTTP worker's context factory, or run this module
for a short TP=1/2/4 generation comparison. Profiled timings are diagnostic;
use the unprofiled replay/serving benchmarks for performance comparisons.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import gzip
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

import torch
import torch.distributed as dist

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.context import reset_context


def cuda_activity_from_trace(path):
    """Count CUPTI activities, excluding enclosing GPU user annotations."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        trace = json.load(stream)
    events = defaultdict(lambda: {"count": 0, "device_total_us": 0.0})
    categories = defaultdict(lambda: {"count": 0, "device_total_us": 0.0})
    nccl = defaultdict(lambda: {"count": 0, "device_total_us": 0.0})
    intervals = []
    for event in trace["traceEvents"]:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        name, duration = event["name"], event["dur"]
        groups = [events[name], categories[category]]
        if category == "kernel" and "nccl" in name.lower():
            groups.append(nccl[name])
        for row in groups:
            row["count"] += 1
            row["device_total_us"] += duration
        intervals.append((event["ts"], event["ts"] + duration))
    union = 0.0
    previous_end = float("-inf")
    for start, end in sorted(intervals):
        union += max(0.0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    span = max(end for _, end in intervals) - min(start for start, _ in intervals) if intervals else 0.0
    return {
        "device_events": dict(events), "device_activity_categories": dict(categories),
        "nccl_device_events": dict(nccl), "device_activity_union_us": union,
        "device_activity_span_us": span,
    }


def compress_trace(path):
    path = Path(path)
    destination = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(destination, "wb") as target:
        shutil.copyfileobj(source, target)
    path.unlink()
    return destination


def memory_snapshot(runner):
    free, total = torch.cuda.mem_get_info()
    return {
        "rank": runner.rank, "device": torch.cuda.current_device(),
        "gpu": torch.cuda.get_device_name(), "free_bytes": free, "total_bytes": total,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in runner.model.parameters()),
        "kv_blocks": runner.config.num_kvcache_blocks,
        "kv_bytes": runner.kv_cache.numel() * runner.kv_cache.element_size(),
        "weight_load_report": runner.weight_load_report,
    }


class ProfileModelRunner(ModelRunner):
    """Benchmark-only RPC for memory snapshots from every existing TP rank."""

    def profile_memory(self, reset_peak=False):
        rows = [None] * self.world_size
        dist.all_gather_object(rows, memory_snapshot(self))
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        return rows if self.rank == 0 else None


class EngineProfile:
    def __init__(self, engine, output, steps=64):
        if steps < 1:
            raise ValueError("Profile steps must be positive")
        self.engine = engine
        self.output = Path(output)
        self.limit = steps
        self.steps = 0
        self.started = False
        self.stopped = False
        self.originals = []
        self.spans = []

    def _patch(self, obj, name, replacement):
        self.originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def _memory(self, reset_peak=False):
        runner = self.engine.model_runner
        if isinstance(runner, ProfileModelRunner):
            return runner.call("profile_memory", reset_peak)
        result = [memory_snapshot(runner)]
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        return result

    @contextmanager
    def _span(self, name, cuda=False):
        start = end = None
        if cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        before = perf_counter()
        with torch.profiler.record_function("nanosched." + name):
            yield
        elapsed = perf_counter() - before
        if end is not None:
            end.record()
        self.spans.append((name, elapsed * 1e6, start, end))

    def _start(self):
        self.started = True
        self.output.mkdir(parents=True, exist_ok=True)
        self.memory_before = self._memory(reset_peak=True)
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False)
        self.profiler.start()
        runner = self.engine.model_runner

        # Preserve the actual run path, separating the existing .tolist() GPU
        # wait from sampling kernel launches. Other TP ranks run unchanged.
        def run(seqs, is_prefill):
            phase = "prefill" if is_prefill else "decode"
            with self._span(phase + ".metadata", cuda=True):
                inputs, positions = runner.prepare_prefill(seqs) if is_prefill else runner.prepare_decode(seqs)
            with self._span(phase + ".sampling_metadata", cuda=True):
                temperatures = runner.prepare_sample(seqs)
            with self._span(phase + ".model", cuda=True):
                logits = runner.run_model(inputs, positions, is_prefill)
            with self._span(phase + ".sample", cuda=True):
                sampled = runner.sampler(logits, temperatures)
            with self._span(phase + ".token_sync"):
                tokens = sampled.tolist()
            reset_context()
            return tokens

        self._patch(runner, "run", run)
        for obj, name, label in [
            (runner, "write_shm", "tp.serialize_notify"),
            (self.engine.scheduler, "schedule", "scheduler.schedule"),
            (self.engine.scheduler, "postprocess", "scheduler.postprocess"),
            (dist, "all_reduce", "collective.all_reduce"),
            (dist, "gather", "collective.gather"),
        ]:
            original = getattr(obj, name)
            def wrapped(*args, _original=original, _label=label, **kwargs):
                with self._span(_label):
                    return _original(*args, **kwargs)
            self._patch(obj, name, wrapped)

    def _stop(self):
        self.stopped = True
        self.profiler.stop()
        torch.cuda.synchronize()
        trace_path = self.output / "profile.json"
        self.profiler.export_chrome_trace(str(trace_path))
        activity = cuda_activity_from_trace(trace_path)
        compressed_trace = compress_trace(trace_path)
        components = defaultdict(lambda: {"count": 0, "host_wall_total_us": 0.0,
                                           "host_wall_max_us": 0.0, "cuda_stream_total_us": 0.0})
        for name, wall, start, end in self.spans:
            row = components[name]
            row["count"] += 1
            row["host_wall_total_us"] += wall
            row["host_wall_max_us"] = max(row["host_wall_max_us"], wall)
            if start is not None:
                row["cuda_stream_total_us"] += start.elapsed_time(end) * 1000
        sync = defaultdict(lambda: {"count": 0, "host_total_us": 0.0})
        for event in self.profiler.events():
            if "CPU" in str(event.device_type) and "synchronize" in event.name.lower():
                row = sync[event.name]
                row["count"] += 1
                row["host_total_us"] += event.cpu_time_total
        # Restore collective wrappers before the all-rank memory RPC.
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)
        self.originals.clear()
        config = self.engine.config
        summary = {
            "profiled_steps": self.steps,
            "model": config.model,
            "tensor_parallel_size": config.tensor_parallel_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_graph": not config.enforce_eager,
            "scheduler_policy": config.scheduler_policy,
            "token_budget": config.max_num_batched_tokens,
            "max_sequences": config.max_num_seqs,
            "components": dict(components), "synchronization_calls": dict(sync),
            **activity, "trace_file": compressed_trace.name,
            "memory_before": self.memory_before, "memory_after": self._memory(),
            "measurement_notes": [
                "Rank 0 trace; host wall includes Python work and GPU waits.",
                "CUDA stream spans include launch gaps; device events use only Chrome kernel/gpu_memcpy/gpu_memset categories.",
                "Device-event sums may overlap; the union counts overlapping intervals once.",
                "Device activity span runs from the first activity start to the last activity end; it is not summed step wall time.",
                "Python collective counts exclude CUDA graph replay; NCCL device events include replay kernels.",
                "All-rank memory is available with the standalone ProfileModelRunner; ordinary HTTP engines report rank 0.",
                "Initialization and rank-memory collectives are outside the profiled window.",
            ],
        }
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if not activity["device_events"]:
            raise RuntimeError("CUDA profiler recorded no device events; inspect the profiler/CUPTI output")

    def __enter__(self):
        name = "step_with_events" if hasattr(self.engine, "step_with_events") else "step"
        self.step_name = name
        self.original_step = getattr(self.engine, name)

        def step(*args, **kwargs):
            if self.stopped:
                return self.original_step(*args, **kwargs)
            if not self.started:
                self._start()
            with self._span("step.total"):
                result = self.original_step(*args, **kwargs)
            self.steps += 1
            self.profiler.step()
            if self.steps >= self.limit:
                self._stop()
            return result

        setattr(self.engine, name, step)
        return self

    def __exit__(self, *exception):
        try:
            if self.started and not self.stopped:
                self._stop()
        finally:
            setattr(self.engine, self.step_name, self.original_step)


def serving_profile(engine, output, steps=64):
    return EngineProfile(engine, output, steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--profile-steps", type=int, default=64)
    parser.add_argument("--input-length", type=int, default=1024)
    parser.add_argument("--output-length", type=int, default=16)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--max-seqs", type=int, default=8)
    parser.add_argument("--memory-utilization", type=float, default=0.45)
    parser.add_argument("--kv-blocks", type=int, default=-1)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--reference", type=Path, help="generation.json from the same inputs at TP=1")
    args = parser.parse_args()
    import nanovllm.engine.llm_engine as engine_module
    from nanovllm import LLM, SamplingParams
    engine_module.ModelRunner = ProfileModelRunner
    args.output.mkdir(parents=True, exist_ok=True)
    engine = LLM(args.model, tensor_parallel_size=args.tp, max_model_len=args.input_length + args.output_length,
                 max_num_batched_tokens=args.budget, max_num_seqs=args.max_seqs,
                 gpu_memory_utilization=args.memory_utilization, num_kvcache_blocks=args.kv_blocks,
                 scheduler_policy="decode_aware", max_consecutive_decodes=3,
                 decode_prefill_token_budget=args.budget, enforce_eager=args.eager)
    try:
        cases = [("What is 17 + 25? Answer with only the number.", "42"),
                 ("What is the capital of France? Answer with only the city name.", "Paris"),
                 ("What is 8 times 7? Answer with only the number.", "56"),
                 ("Write the word apple and nothing else.", "apple")]
        prompts = []
        tokenizer = engine.tokenizer
        filler = tokenizer.encode("The library contains books about science and history. ")
        for question, expected in cases:
            prefix = tokenizer.encode("<|im_start|>user\nBackground notes: ")
            suffix = tokenizer.encode("\nIgnore the background notes and answer this question. "
                                      + question + "<|im_end|>\n<|im_start|>assistant\n")
            remaining = args.input_length - len(prefix) - len(suffix)
            if remaining < 0:
                raise ValueError("Input length is shorter than the correctness prompt")
            prompts.append(prefix + (filler * (remaining // len(filler) + 1))[:remaining] + suffix)
        # Exercise sampling and graph replay before starting the measured window.
        engine.generate([[42] * min(128, args.input_length)],
                        SamplingParams(temperature=0, max_tokens=min(4, args.output_length)), use_tqdm=False)
        with serving_profile(engine, args.output, args.profile_steps):
            outputs = engine.generate(prompts, SamplingParams(temperature=0, max_tokens=args.output_length), use_tqdm=False)
        rows = [{"prompt_ids": prompt, "expected": expected, **output,
                 "answer": tokenizer.decode(output["token_ids"], skip_special_tokens=True).strip()}
                for prompt, (_, expected), output in zip(prompts, cases, outputs)]
        for row in rows:
            row["semantic_match"] = row["answer"] == row["expected"]
        result = {"tp": args.tp, "cases": rows}
        if args.reference:
            reference = json.loads(args.reference.read_text())["cases"]
            result["reference_inputs_match"] = [r["prompt_ids"] for r in rows] == [r["prompt_ids"] for r in reference]
            result["reference_tokens_match"] = [r["token_ids"] for r in rows] == [r["token_ids"] for r in reference]
        (args.output / "generation.json").write_text(json.dumps(result, indent=2) + "\n")
        assert all(row["semantic_match"] for row in rows), "Semantic check failed; inspect generation.json"
        if args.reference:
            assert result["reference_inputs_match"], "Reference prompt IDs differ"
            assert result["reference_tokens_match"], "TP changed greedy outputs; inspect generation.json"
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
