"""Summarize measured runs, retaining both improvements and regressions."""
import argparse
import csv
import json
from pathlib import Path
import statistics


METRICS = {
    "ttft_p95_ms": ("ttft_s.p95", 1000), "tpot_p95_ms": ("tpot_s.p95", 1000),
    "itl_p95_ms": ("inter_token_gap_s.p95", 1000), "max_gap_ms": ("max_inter_token_gap_s", 1000),
    "e2e_p95_ms": ("e2e_s.p95", 1000), "output_tok_s": ("output_tokens_per_s", 1),
    "slo_goodput_req_s": ("slo_goodput_requests_per_s", 1),
    "tpot_met_pct": ("tpot_slo_met_fraction", 100), "itl_met_pct": ("itl_slo_met_fraction", 100),
    "scheduler_mean_us": ("scheduler_mean_us", 1), "scheduler_p95_us": ("scheduler_p95_us", 1),
    "scheduler_cpu_pct": ("scheduler_cpu_pct", 1), "kv_peak_pct": ("kv_peak_utilization", 100),
    "preemptions": ("preemptions", 1), "recomputed_tokens": ("recomputed_prefill_tokens", 1),
    "arrival_lag_p95_ms": ("arrival_lag_p95_s", 1000), "admission_wait_p95_ms": ("admission_wait_p95_s", 1000),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads((args.suite / "commands.json").read_text())
    rows, workloads, configs = [], {}, {}
    for job in jobs:
        path = Path(job["path"])
        runs = json.loads((path / "summary.json").read_text())["runs"]
        env = json.loads((path / "environment.json").read_text())
        workload = json.loads((path / "workload.json").read_text())
        case = job["case"]
        assert workload == workloads.setdefault(case, workload), "Policies used different request sets"
        common = {k: env["config"][k] for k in ("max_num_batched_tokens", "max_num_seqs", "max_model_len", "gpu_memory_utilization", "enforce_eager")}
        common.update({k: env[k] for k in ("model", "seed", "temperature", "cuda_visible_devices")})
        assert common == configs.setdefault(case, common), "Policies used different hardware/model settings"
        assert len(runs) >= 3 and not env["profile"]
        row = {"case": case, "policy": job["policy"], "runs": len(runs), "source": str(path)}
        for metric, (key, scale) in METRICS.items():
            values = []
            for run in runs:
                value = run
                for part in key.split("."):
                    value = value[part]
                values.append(value * scale)
            row[metric] = statistics.mean(values)
            row[metric + "_std"] = statistics.stdev(values)
        rows.append(row)
    (args.suite / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.suite / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = ["ttft_p95_ms", "tpot_p95_ms", "e2e_p95_ms", "itl_p95_ms", "output_tok_s", "slo_goodput_req_s", "tpot_met_pct", "scheduler_mean_us", "kv_peak_pct", "preemptions"]
    lines = ["|Case|Policy|TTFT P95 ms|TPOT P95 ms|E2E P95 ms|ITL P95 ms|Output tok/s|TTFT+TPOT goodput req/s|TPOT met %|Schedule μs|KV peak %|Preemptions|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("|" + "|".join([row["case"], row["policy"], *[f"{row[k]:.2f} ± {row[k+'_std']:.2f}" for k in keys]]) + "|")
    (args.suite / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
