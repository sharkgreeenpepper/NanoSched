"""Aggregate saved runs without inferring token timings from total runtime."""
import argparse
import csv
import json
from pathlib import Path
import statistics


METRICS = {
    "ttft_p50_ms": ("ttft_s.p50", 1000), "ttft_p95_ms": ("ttft_s.p95", 1000),
    "ttft_p99_ms": ("ttft_s.p99", 1000), "tpot_p50_ms": ("tpot_s.p50", 1000),
    "tpot_p95_ms": ("tpot_s.p95", 1000), "tpot_p99_ms": ("tpot_s.p99", 1000),
    "e2e_p95_ms": ("e2e_s.p95", 1000), "max_gap_ms": ("max_inter_token_gap_s", 1000),
    "output_tok_s": ("output_tokens_per_s", 1), "request_s": ("requests_per_s", 1),
    "goodput_request_s": ("slo_goodput_requests_per_s", 1),
    "scheduler_ms": ("scheduler_total_s", 1000), "kv_peak_pct": ("kv_peak_utilization", 100),
    "preemptions": ("preemptions", 1), "recomputed_tokens": ("recomputed_prefill_tokens", 1),
}


def get(row, path):
    for key in path.split("."):
        row = row[key]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--competition-eager-override", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for mode in ("eager", "graph"):
        for load in ("competition", "short", "long"):
            pair = []
            for policy in ("prefill_first", "decode_aware"):
                path = args.suite/f"{load}_{policy}_{mode}"
                if load == "competition" and policy == "decode_aware" and mode == "eager" and args.competition_eager_override:
                    path = args.competition_eager_override
                summary = json.loads((path/"summary.json").read_text())
                environment = json.loads((path/"environment.json").read_text())
                runs = summary["runs"]
                assert len(runs) >= 3, path
                assert not environment["profile"], "Do not aggregate profiler runs"
                pair.append((summary, environment, json.loads((path/"workload.json").read_text())))
                row = {"workload": load, "policy": policy, "mode": mode, "runs": len(runs),
                       "commit": environment["commit"], "source": str(path)}
                for name, (path_key, scale) in METRICS.items():
                    values = [get(r, path_key)*scale for r in runs]
                    row[name+"_mean"] = statistics.mean(values)
                    row[name+"_std"] = statistics.stdev(values)
                    row[name+"_variance"] = statistics.variance(values)
                if load == "competition":
                    for name, metric in (("incumbent_tpot_ms", "groups.incumbent.tpot_s.p50"),
                                         ("short_ttft_p95_ms", "groups.short.ttft_s.p95")):
                        values = [get(r, metric)*1000 for r in runs]
                        row[name+"_mean"], row[name+"_std"] = statistics.mean(values), statistics.stdev(values)
                rows.append(row)
            assert pair[0][2] == pair[1][2]
            for key in ("seed", "temperature", "model", "cuda_visible_devices"):
                assert pair[0][1][key] == pair[1][1][key], key
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output/"summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    table = ["|负载|Graph|策略|TTFT P95 ms|TPOT P95 ms|最长 Token 间隔 ms|Output tok/s|Request/s|Goodput req/s|",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(row, key):
        return f"{row[key+'_mean']:.2f} ± {row[key+'_std']:.2f}"
    for row in rows:
        table.append("|"+"|".join([row["workload"], row["mode"], row["policy"],
            *[fmt(row, key) for key in ("ttft_p95_ms", "tpot_p95_ms", "max_gap_ms", "output_tok_s", "request_s", "goodput_request_s")]])+"|")
    (args.output/"table.md").write_text("\n".join(table)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, key, label in zip(axes, ("max_gap_ms", "ttft_p95_ms", "output_tok_s"),
                              ("Maximum inter-token gap (ms)", "TTFT P95 (ms)", "Output throughput (tokens/s)")):
        for i, policy in enumerate(("prefill_first", "decode_aware")):
            selected = [r for r in rows if r["workload"] == "competition" and r["policy"] == policy]
            ax.bar([j+(i-0.5)*0.35 for j in range(2)], [r[key+"_mean"] for r in selected], width=0.35,
                   yerr=[r[key+"_std"] for r in selected], capsize=3, label=policy)
        ax.set_xticks([0, 1], ["Graph off", "Graph on"])
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    fig.suptitle("H20 / Qwen2.5-7B BF16 / online arrival replay / 3 runs, mean ± SD", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output/"competition.png", dpi=180)
    print("\n".join(table))


if __name__ == "__main__":
    main()
