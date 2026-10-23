"""Reproduce the H20 scheduling matrix; each case uses identical arrivals across policies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


# Input tokens, arrival rate, engine concurrency, request count, memory fraction.
CASES = {
    "short-1k": ("short", 1024, 2, 4, 12, 0.45),
    "mixed-1k": ("mixed", 1024, 2, 8, 12, 0.45),
    "mixed-4k": ("mixed", 4096, 2, 8, 12, 0.45),
    "mixed-8k": ("mixed", 8192, 2, 8, 12, 0.45),
    "mixed-16k": ("mixed", 16384, 2, 8, 12, 0.45),
    "competition-4k-low": ("competition", 4096, 0.5, 4, 8, 0.45),
    "competition-4k": ("competition", 4096, 4, 16, 12, 0.45),
    "competition-4k-c4": ("competition", 4096, 4, 4, 12, 0.45),
    "competition-8k": ("competition", 8192, 2, 16, 12, 0.45),
    "competition-16k": ("competition", 16384, 1, 16, 12, 0.45),
    "pressure-16k": ("pressure", 16384, 2, 16, 16, 0.22),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--policies", nargs="+", choices=["original", "decode_aware", "adaptive", "adaptive_v2"], default=["original", "decode_aware", "adaptive", "adaptive_v2"])
    parser.add_argument("--fixed-chunk", type=int, default=1024)
    parser.add_argument("--fixed-decode-limit", type=int, default=1)
    parser.add_argument("--target-tpot", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, name in enumerate(args.cases):
        load, length, rate, concurrency, count, memory = CASES[name]
        policies = args.policies[index % len(args.policies):] + args.policies[:index % len(args.policies)]
        for policy in policies:
            path = args.output / f"{name}-{policy}"
            command = [sys.executable, "-m", "benchmarks.replay", "--model", args.model,
                       "--policy", policy, "--workload", load, "--input-length", str(length),
                       "--output-length", "64", "--arrival-rate", str(rate), "--num-requests", str(count),
                       "--concurrency", str(concurrency), "--max-seqs", "32", "--budget", "2048",
                       "--decode-prefill-budget", str(args.fixed_chunk), "--memory-utilization", str(memory),
                       "--target-tpot", str(args.target_tpot), "--ttft-slo", "5", "--seed", str(args.seed),
                       "--repeats", str(args.repeats), "--output", str(path)]
            if policy == "decode_aware":
                command += ["--decode-limit", str(args.fixed_decode_limit)]
            jobs.append({"case": name, "policy": policy, "path": str(path), "command": command})
    (args.output / "commands.json").write_text(json.dumps(jobs, indent=2) + "\n")
    for job in jobs:
        print(f"Running {job['case']} / {job['policy']}", flush=True)
        with Path(job["path"] + ".stdout").open("w") as stream:
            subprocess.run(job["command"], stdout=stream, stderr=subprocess.STDOUT, check=True)
    if args.repeats >= 3:
        subprocess.run([sys.executable, "-m", "benchmarks.report", "--suite", str(args.output)], check=True)
    else:
        print("Quick run complete; per-policy summaries saved. Aggregate report requires at least 3 repeats.")


if __name__ == "__main__":
    main()
