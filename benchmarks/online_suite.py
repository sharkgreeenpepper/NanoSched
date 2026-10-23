"""Serial online comparisons: a fresh serving process and identical trace per case."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
from threading import Event, Thread
import time
from urllib.request import urlopen


# Input length, output length, client concurrency, count, arrivals/s (0 = burst), mixed.
CASES = {
    "1k-128-c1": (1024, 128, 1, 1, 0, False),
    "1k-256-c32-burst": (1024, 256, 32, 32, 0, False),
    "4k-128-c4": (4096, 128, 4, 8, 4, False),
    "4k-256-c16-burst": (4096, 256, 16, 16, 0, False),
    "8k-128-c8": (8192, 128, 8, 8, 4, False),
    "8k-256-c16-mixed": (8192, 256, 16, 16, 8, True),
    "16k-128-c4": (16384, 128, 4, 8, 2, False),
    "16k-256-c32-mixed": (16384, 256, 32, 32, 8, True),
    "16k-256-c1": (16384, 256, 1, 1, 0, False),
}


def make_workload(case, seed):
    length, output, concurrency, count, rate, mixed = CASES[case]
    rng = random.Random(seed)
    items = []
    for i in range(count):
        n = 1024 if mixed and i % 4 else length
        items.append({"id": f"request-{i}", "arrival_s": i / rate if rate else 0,
                      "prompt": [rng.randrange(100, 20000) for _ in range(n)],
                      "max_tokens": output, "group": "short" if n < length else "long"})
    return items


def wait_ready(process, url, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Serving exited during initialization: {process.returncode}")
        try:
            with urlopen(url + "/health", timeout=2) as response:
                return json.load(response)
        except OSError:
            time.sleep(0.2)
    raise TimeoutError("Serving did not become ready; see server.stdout")


def stop_server(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise RuntimeError("Serving failed graceful shutdown; its process group was stopped")
    if process.returncode not in (0, -signal.SIGINT):
        raise RuntimeError(f"Serving exit status {process.returncode}; see server.stdout")


def sample_gpus(stop, gpu_ids, path):
    command = ["nvidia-smi", f"--id={gpu_ids}",
               "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw",
               "--format=csv,noheader,nounits"]
    with path.open("w") as stream:
        while not stop.is_set():
            output = subprocess.check_output(command, text=True)
            stream.write(json.dumps({"at": time.perf_counter(), "gpus": output.strip().splitlines()}) + "\n")
            stream.flush()
            stop.wait(1)


def aggregate(output, jobs):
    rows = []
    for job in jobs:
        path = Path(job["path"])
        if not (path / "summary.json").exists():
            continue
        runs = json.loads((path / "summary.json").read_text())["runs"]
        for repeat, run in enumerate(runs):
            client = run["client"]
            row = {"case": job["case"], "policy": job["policy"], "tp": job["tp"], "repeat": repeat}
            for metric in ("ttft_s", "token_itl_s", "request_mean_tpot_s", "e2e_s"):
                for percentile in ("p50", "p95", "p99"):
                    row[f"{metric}_{percentile}"] = client[metric][percentile]
            for metric in ("output_tokens_per_s", "requests_per_s", "slo_goodput_requests_per_s", "elapsed_s"):
                row[metric] = client[metric]
            row["output_tokens_per_gpu_s"] = client["output_tokens_per_s"] / job["tp"]
            row.update(run["server_wall_s"])
            for key, values in run["latency_attribution_s"].items():
                row[key + "_p95_s"] = values["p95"]
            trace = json.loads((path / f"trace-{repeat}.json").read_text())
            samples = [json.loads(line) for line in (path / "gpu.jsonl").read_text().splitlines()]
            gpu_rows = [line.split(", ") for sample in samples
                        if trace["origin"] <= sample["at"] <= trace["origin"] + trace["elapsed_s"]
                        for line in sample["gpus"]]
            row["gpu_peak_used_mib"] = max((float(r[2]) for r in gpu_rows), default=None)
            row["gpu_utilization_mean_pct"] = sum(float(r[4]) for r in gpu_rows) / len(gpu_rows) if gpu_rows else None
            rows.append(row)
    (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (output / "results.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", choices=list(CASES), nargs="+", default=list(CASES))
    parser.add_argument("--policies", choices=["original", "decode_aware", "adaptive_v2"], nargs="+", default=["decode_aware"])
    parser.add_argument("--tp-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gpu_ids = args.gpus.split(",")
    url = f"http://127.0.0.1:{args.port}"
    jobs = []
    for case in args.cases:
        trace = args.output / f"{case}-workload.json"
        trace.write_text(json.dumps(make_workload(case, args.seed)) + "\n")
        for tp in args.tp_sizes:
            if tp > len(gpu_ids):
                parser.error("--gpus must contain at least --tp-sizes GPUs")
            for policy in args.policies:
                path = args.output / f"{case}-{policy}-tp{tp}"
                command = [sys.executable, "-m", "nanovllm.serving", "--model", args.model,
                           "--port", str(args.port), "--policy", policy, "--tp", str(tp),
                           "--max-seqs", "32", "--budget", "2048", "--decode-limit", "3",
                           "--decode-prefill-budget", "2048", "--memory-utilization", "0.45",
                           "--max-model-len", "32768"]
                if args.eager:
                    command.append("--eager")
                client = [sys.executable, "-m", "benchmarks.online", "--url", url,
                          "--workload-file", str(trace), "--output", str(path),
                          "--concurrency", str(CASES[case][2]), "--repeats", "1", "--warmup-runs", "1"]
                jobs.append({"case": case, "tp": tp, "policy": policy, "path": str(path),
                             "gpus": ",".join(gpu_ids[:tp]), "server": command, "client": client,
                             "status": "pending"})
    command_file = args.output / "commands.json"
    command_file.write_text(json.dumps(jobs, indent=2) + "\n")
    for job in jobs:
        path = Path(job["path"])
        path.mkdir(parents=True, exist_ok=True)
        print(f"Running {job['case']} / {job['policy']} / TP={job['tp']}", flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=job["gpus"])
        with (path / "server.stdout").open("w") as stdout:
            process = subprocess.Popen(job["server"], env=env, stdout=stdout, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            stop = Event()
            sampler = None
            try:
                ready = wait_ready(process, url)
                (path / "server-ready.json").write_text(json.dumps(ready, indent=2) + "\n")
                sampler = Thread(target=sample_gpus, args=(stop, job["gpus"], path / "gpu.jsonl"))
                sampler.start()
                with (path / "client.stdout").open("w") as client_stdout:
                    subprocess.run(job["client"], stdout=client_stdout, stderr=subprocess.STDOUT, check=True)
                with urlopen(url + "/health", timeout=10) as response:
                    final = json.load(response)
                (path / "server-final.json").write_text(json.dumps(final, indent=2) + "\n")
                job["status"] = "completed"
            except BaseException:
                job["status"] = "failed"
                raise
            finally:
                stop.set()
                if sampler:
                    sampler.join()
                stop_server(process)
                command_file.write_text(json.dumps(jobs, indent=2) + "\n")
                aggregate(args.output, jobs)


if __name__ == "__main__":
    main()
