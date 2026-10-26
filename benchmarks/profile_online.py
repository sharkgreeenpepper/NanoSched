"""Profile a bounded window inside the actual HTTP inference worker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from benchmarks.online_suite import CASES, make_workload, stop_server, wait_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--case", choices=list(CASES), default="1k-256-c32-burst")
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trace = args.output / "workload.json"
    trace.write_text(json.dumps(make_workload(args.case, 20261023)) + "\n")
    command = [sys.executable, "-m", "nanovllm.serving", "--model", args.model,
               "--port", "8000", "--tp", str(args.tp), "--max-seqs", "32",
               "--budget", "2048", "--decode-limit", "3", "--decode-prefill-budget", "2048",
               "--memory-utilization", "0.45", "--max-model-len", "32768",
               "--profile-path", str(args.output / "profile"), "--profile-steps", str(args.steps)]
    if args.eager:
        command.append("--eager")
    client = [sys.executable, "-m", "benchmarks.online", "--url", "http://127.0.0.1:8000",
              "--workload-file", str(trace), "--output", str(args.output / "client"),
              "--concurrency", str(CASES[args.case][2]), "--warmup-runs", "0", "--repeats", "1"]
    (args.output / "commands.json").write_text(json.dumps({"server": command, "client": client}, indent=2) + "\n")
    with (args.output / "server.stdout").open("w") as server_log:
        process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_ready(process, "http://127.0.0.1:8000")
            with (args.output / "client.stdout").open("w") as client_log:
                subprocess.run(client, stdout=client_log, stderr=subprocess.STDOUT, check=True)
        finally:
            stop_server(process)


if __name__ == "__main__":
    main()
