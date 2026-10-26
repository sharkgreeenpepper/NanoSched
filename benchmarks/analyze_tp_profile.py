"""Reproduce phase, NCCL, head-gap and sampling statistics from saved CPU/GPU traces.

Run with Python only; no model or CUDA initialization. Existing extra metrics
fields are preserved. Times are microseconds; *_count fields are counts.
"""
import argparse
import bisect
from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import median


def distribution(values):
    return {"mean": sum(values) / len(values), "median": median(values),
            "p95": sorted(values)[int(.95 * (len(values) - 1))]} if values else None


def union(events):
    end, total = float("-inf"), 0.0
    for start, stop in sorted((e["ts"], e["ts"] + e["dur"]) for e in events):
        total += max(0.0, stop - max(start, end))
        end = max(stop, end)
    return total


def inside(event, span):
    return span["ts"] <= event["ts"] < span["ts"] + span["dur"]


def analyze(folder):
    summary = json.loads((folder / "profile" / "summary.json").read_text())
    with gzip.open(folder / "profile" / summary["trace_file"], "rt") as stream:
        events = [e for e in json.load(stream)["traceEvents"] if e.get("ph") == "X"]
    steps = sorted((e for e in events if e.get("cat") == "user_annotation"
                    and e["name"] == "nanosched.step.total"), key=lambda e: e["ts"])
    starts, buckets = [e["ts"] for e in steps], [[] for _ in steps]
    outside = 0
    for event in events:
        index = bisect.bisect_right(starts, event["ts"]) - 1
        if index >= 0 and inside(event, steps[index]):
            buckets[index].append(event)
        elif event.get("cat") == "kernel":
            outside += 1
    phases, sampling = defaultdict(list), []
    for step, events in zip(steps, buckets):
        annotations = {e["name"]: e for e in events if e.get("cat") == "user_annotation"}
        phase = "decode" if "nanosched.decode.model" in annotations else "prefill"
        gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
        kernels = [e for e in gpu if e["cat"] == "kernel"]
        nccl = [e for e in kernels if "nccl" in e["name"].lower()]
        reductions = [e for e in nccl if "AllReduce" in e["name"]]
        gather = [e for e in nccl if "SendRecv" in e["name"]]
        runtime = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")
                   and "correlation" in e.get("args", {})]

        def associated(span):
            correlations = {e["args"]["correlation"] for e in runtime if span and inside(e, span)}
            return [e for e in gpu if e.get("args", {}).get("correlation") in correlations]

        row = {"wall_us": step["dur"], "gpu_busy_us": union(gpu),
               "kernel_sum_us": sum(e["dur"] for e in kernels)}
        for name, group in (("nccl", nccl), ("all_reduce", reductions), ("gather", gather)):
            row[name + "_count"] = len(group)
            row[name + "_sum_us"] = sum(e["dur"] for e in group)
        for name in ("metadata", "sampling_metadata", "model", "sample", "token_sync"):
            row[name + "_host_us"] = sum(e["dur"] for e in events if e.get("cat") == "user_annotation"
                                         and e["name"] == f"nanosched.{phase}.{name}")
        for name in ("tp.serialize_notify", "collective.gather", "collective.all_reduce",
                     "scheduler.schedule", "scheduler.postprocess"):
            row[name + "_host_us"] = sum(e["dur"] for e in events if e.get("cat") == "user_annotation"
                                         and e["name"] == "nanosched." + name)
        if phase == "decode":
            model = annotations["nanosched.decode.model"]
            within = [e for e in events if inside(e, model)]
            linear = [e for e in within if e.get("cat") == "cpu_op" and e["name"] == "aten::linear"]
            row.update(linear_host_us=sum(e["dur"] for e in linear), linear_count=len(linear),
                       cat_host_us=sum(e["dur"] for e in within if e.get("cat") == "cpu_op" and e["name"] == "aten::cat"),
                       graph_launch_host_us=sum(e["dur"] for e in within if e["name"] == "cudaGraphLaunch"))
            if summary["cuda_graph"]:
                assert len(linear) == 1, (folder, len(linear))  # The eager LM head.
                head = [e for e in associated(linear[0]) if e["cat"] == "kernel"]
                body = [e for e in kernels if e.get("args", {}).get("graph id", 0) > 0]
                row.update(head_kernel_us=sum(e["dur"] for e in head), head_kernel_count=len(head),
                           body_to_head_gap_us=min(e["ts"] for e in head) - max(e["ts"] + e["dur"] for e in body),
                           head_to_gather_gap_us=min(e["ts"] for e in gather) - max(e["ts"] + e["dur"] for e in head) if gather else None)
            sample = [e for e in associated(annotations["nanosched.decode.sample"]) if e["cat"] == "kernel"]
            metadata = associated(annotations.get("nanosched.decode.sampling_metadata"))
            copied = [e for e in associated(annotations["nanosched.decode.token_sync"]) if e["cat"] == "gpu_memcpy"]
            assert sample and copied, (folder, "Sampling kernels or D2H copy absent")
            first, last = min(e["ts"] for e in sample), max(e["ts"] + e["dur"] for e in sample)
            compute_end = max(e["ts"] + e["dur"] for e in kernels if e not in sample and e["ts"] < first)
            sample_us = sum(e["dur"] for e in sample)
            sampling.append({
                "sampler_kernel_count": len(sample), "sampler_kernel_us": sample_us,
                "sampler_span_us": last - first, "kernel_internal_gap_us": last - first - sample_us,
                "logits_ready_to_sample_gpu_us": first - compute_end,
                "sampling_metadata_copy_us": sum(e["dur"] for e in metadata if e["cat"] == "gpu_memcpy"),
                "sampling_metadata_host_us": row["sampling_metadata_host_us"],
                "sample_to_d2h_gap_us": min(e["ts"] for e in copied) - last,
                "d2h_us": sum(e["dur"] for e in copied),
                "stochastic_fused_kernel_us": sum(e["dur"] for e in sample if "exponential" in e["name"]),
                "softmax_only_kernel_us": sum(e["dur"] for e in sample if e["name"] in
                                               ("triton_red_fused__softmax_4", "triton_per_fused__softmax_5")),
                "rng_kernel_us": sum(e["dur"] for e in sample if "distribution_elementwise" in e["name"]),
                "step_wall_us": step["dur"],
            })
        phases[phase].append(row)
    result = {"tp": summary["tensor_parallel_size"], "cuda_graph": summary["cuda_graph"],
              "gpu_busy_us": summary["device_activity_union_us"], "gpu_span_us": summary["device_activity_span_us"],
              "busy_ratio": summary["device_activity_union_us"] / summary["device_activity_span_us"],
              "kernel_count": summary["device_activity_categories"]["kernel"]["count"],
              "kernel_sum_us": summary["device_activity_categories"]["kernel"]["device_total_us"],
              "nccl_sum_us": sum(v["device_total_us"] for v in summary["nccl_device_events"].values()),
              "nccl_count": sum(v["count"] for v in summary["nccl_device_events"].values()),
              "outside_step_kernel_count": outside, "phases": {}}
    for phase, rows in phases.items():
        result["phases"][phase] = {"steps": len(rows), "measurements": {
            key: distribution([r[key] for r in rows if r[key] is not None]) for key in rows[0]}}
    sampled = {key: distribution([r[key] for r in sampling]) for key in sampling[0]}
    sampled.update(steps=len(sampling), kernel_fraction_of_decode_wall=
                   sum(r["sampler_kernel_us"] for r in sampling) / sum(r["step_wall_us"] for r in sampling))
    return result, sampled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports/phase5"))
    args = parser.parse_args()
    output = args.root / "tp-profile-metrics.json"
    metrics = json.loads(output.read_text()) if output.exists() else {}
    sampling = metrics.setdefault("greedy_sampling", {})
    for folder in sorted(args.root.glob("online-profile-*")):
        if not (folder / "profile" / "summary.json").exists():
            continue  # The bounded profile is still being collected.
        result, sampled = analyze(folder)
        metrics[folder.name] = result
        sampling[folder.name.removeprefix("online-profile-tp")] = sampled
        print(f"{folder.name}: GPU busy {result['busy_ratio']:.2%}; "
              f"decode sample {sampled['sampler_kernel_count']['mean']:.0f} kernels, "
              f"{sampled['sampler_kernel_us']['mean']:.2f} us; outside-step kernels {result['outside_step_kernel_count']}")
    output.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
