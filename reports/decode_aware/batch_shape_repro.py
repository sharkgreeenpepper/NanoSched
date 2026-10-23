import json
from collections import deque
from pathlib import Path
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager

root = Path('reports/decode_aware')
workload = json.loads((root/'prefill_first_eager/workload.json').read_text())
reference = json.loads((root/'prefill_first_eager/trace-0.json').read_text())
item = next(r for r in workload if r['id'] == 'long-4')
ref = next(r for r in reference['requests'] if r['id'] == 'long-4')['output_ids']
engine = LLM('/data1/fangdongqing/infra/models/Qwen/Qwen2.5-7B-Instruct', enforce_eager=True,
             scheduler_policy='prefill_first', max_model_len=8192, max_num_batched_tokens=1024,
             max_num_seqs=32, gpu_memory_utilization=0.45)
selected = []
original_run = engine.model_runner.run
original_sample = engine.model_runner.sampler.forward
captured = []

def run(seqs, is_prefill):
    selected[:] = seqs
    return original_run(seqs, is_prefill)

def sample(logits, temperatures):
    tokens = original_sample(logits, temperatures)
    if target in selected and target.num_cached_tokens + target.num_scheduled_tokens == len(target):
        row = selected.index(target)
        t = target.num_completion_tokens
        values, indices = logits[row].float().topk(5)
        if t == 2:
            captured.append({'batch_size': len(selected), 'row': row, 'top_ids': indices.tolist(),
                             'top_logits': values.tolist(), 'reference_token': ref[t]})
        if t < 2:
            tokens[row] = ref[t]
    return tokens

engine.model_runner.run = run
engine.model_runner.sampler.forward = sample
for batch, row in [(21, 7), (7, 5)]:
    old = engine.scheduler.block_manager
    engine.scheduler.block_manager = BlockManager(len(old.blocks), old.block_size)
    engine.scheduler.waiting.clear()
    engine.scheduler.running.clear()
    engine.add_request(item['prompt'], SamplingParams(temperature=0, max_tokens=16, ignore_eos=True))
    target = engine.scheduler.waiting[-1]
    while target.num_completion_tokens == 0:
        engine.step()
    for i in range(batch-1):
        engine.add_request([100+i]*128, SamplingParams(temperature=0, max_tokens=64, ignore_eos=True))
    while engine.scheduler.waiting:
        engine.step()
    others = [s for s in engine.scheduler.running if s is not target]
    engine.scheduler.running = deque(others[:row]+[target]+others[row:])
    while target.num_completion_tokens < 3:
        engine.step()
    captured[-1]['emitted_token'] = target.last_token
engine.exit()
print(json.dumps(captured))
