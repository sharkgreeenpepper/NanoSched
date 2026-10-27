"""Four independent TP=1 servers, using the same traces as online_suite."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from urllib.request import urlopen

from benchmarks.online_suite import CASES, make_workload, sample_gpus, stop_server, wait_ready


SELECTED_CASES = ['1k-256-c32-burst', '16k-256-c32-mixed']
ROOT = Path(__file__).resolve().parents[1]


def aggregate(output, jobs):
    rows = []
    for job in jobs:
        path = Path(job['path'])
        if job['status'] != 'completed':
            continue
        run = json.loads((path / 'summary.json').read_text())['runs'][0]
        client = run['client']
        row = dict(case=job['case'], deployment='4xTP1', policy='decode_aware',
                   tp_per_instance=1, replicas=4, gpu_count=4, repeat=0)
        for metric in ('ttft_s', 'token_itl_s', 'request_mean_tpot_s', 'e2e_s'):
            for percentile in ('p50', 'p95', 'p99'):
                row[f'{metric}_{percentile}'] = client[metric][percentile]
        for metric in ('output_tokens_per_s', 'requests_per_s', 'slo_goodput_requests_per_s', 'elapsed_s'):
            row[metric] = client[metric]
        row['output_tokens_per_gpu_s'] = client['output_tokens_per_s'] / 4
        row.update(run['server_wall_s'])
        for key, values in run['latency_attribution_s'].items():
            row[key + '_p95_s'] = values['p95']
        trace = json.loads((path / 'trace-0.json').read_text())
        samples = [json.loads(line) for line in (path / 'gpu.jsonl').read_text().splitlines()]
        gpu_rows = [line.split(', ') for sample in samples
                    if trace['origin'] <= sample['at'] <= trace['origin'] + trace['elapsed_s']
                    for line in sample['gpus']]
        row['gpu_peak_used_mib'] = max((float(r[2]) for r in gpu_rows), default=None)
        row['gpu_utilization_mean_pct'] = sum(float(r[4]) for r in gpu_rows) / len(gpu_rows) if gpu_rows else None
        rows.append(row)
    (output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    if rows:
        with (output / 'results.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--cases', choices=SELECTED_CASES, nargs='+', default=SELECTED_CASES)
    args = parser.parse_args()
    gpu_ids = args.gpus.split(',')
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        parser.error('--gpus must identify four distinct GPUs')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    urls = [f'http://127.0.0.1:{8000 + i}' for i in range(4)]
    jobs = []
    for case in args.cases:
        path = output / f'{case}-fixed-4xtp1'
        workload = make_workload(case, 20261023)
        trace = output / f'{case}-workload.json'
        trace.write_text(json.dumps(workload) + '\n')
        servers = []
        for i, gpu in enumerate(gpu_ids):
            command = [sys.executable, '-m', 'nanovllm.serving', '--model', args.model,
                       '--port', str(8000 + i), '--policy', 'decode_aware', '--tp', '1',
                       '--max-seqs', '32', '--budget', '2048', '--decode-limit', '3',
                       '--decode-prefill-budget', '2048', '--memory-utilization', '0.45',
                       '--max-model-len', '32768']
            assigned = workload[i::4]
            servers.append(dict(gpu=gpu, url=urls[i], command=command,
                                assigned_requests=len(assigned),
                                assigned_input_tokens=sum(len(item['prompt']) for item in assigned),
                                assigned_output_tokens=sum(item['max_tokens'] for item in assigned)))
        client = [sys.executable, '-m', 'benchmarks.online', '--urls', *urls,
                  '--workload-file', str(trace), '--output', str(path),
                  '--concurrency', str(CASES[case][2]), '--repeats', '1', '--warmup-runs', '1']
        jobs.append(dict(case=case, path=str(path), servers=servers, client=client,
                         routing='request-index round-robin; mixed trace concentrates all long prompts on replica 0',
                         status='pending'))
    commands_file = output / 'commands.json'
    commands_file.write_text(json.dumps(jobs, indent=2) + '\n')
    for job in jobs:
        print(f"Running {job['case']} / fixed / four TP1 replicas", flush=True)
        path = Path(job['path'])
        path.mkdir()
        processes, streams = [], []
        stop, sampler, error = Event(), None, None
        try:
            for index, server in enumerate(job['servers']):
                stream = (path / f'server-{index}.stdout').open('w')
                streams.append(stream)
                processes.append(subprocess.Popen(server['command'], cwd=ROOT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=server['gpu']),
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
            ready = {server['url']: wait_ready(process, server['url'])
                     for server, process in zip(job['servers'], processes)}
            (path / 'server-ready.json').write_text(json.dumps(ready, indent=2) + '\n')
            sampler = Thread(target=sample_gpus, args=(stop, args.gpus, path / 'gpu.jsonl'))
            sampler.start()
            with (path / 'client.stdout').open('w') as stream:
                subprocess.run(job['client'], cwd=ROOT, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            final = {}
            for url in urls:
                with urlopen(url + '/health', timeout=10) as response:
                    final[url] = json.load(response)
            (path / 'server-final.json').write_text(json.dumps(final, indent=2) + '\n')
            for url, state in final.items():
                assert all(state[key] == 0 for key in ('active', 'waiting', 'running', 'kv_used')), f'Outstanding request/KV ownership at {url}: {state}'
                assert state['kv_free'] == state['kv_total'], f'Incomplete KV release at {url}'
            job['status'] = 'completed'
        except BaseException as caught:
            error = caught
            job['status'] = 'failed'
            job['error'] = repr(caught)
        finally:
            stop.set()
            if sampler:
                sampler.join()
            for process in reversed(processes):
                try:
                    stop_server(process)
                except BaseException as caught:
                    job['status'] = 'failed'
                    job.setdefault('shutdown_errors', []).append(repr(caught))
                    if error is None:
                        error = caught
            for stream in streams:
                stream.close()
            commands_file.write_text(json.dumps(jobs, indent=2) + '\n')
            aggregate(output, jobs)
        if error is not None:
            raise error


if __name__ == '__main__':
    main()
