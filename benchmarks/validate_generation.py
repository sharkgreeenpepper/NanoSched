"""GPU checks of semantic answers and token equality across prompt/chunk lengths."""
import argparse
from copy import copy
import json
from pathlib import Path

from nanovllm import LLM, SamplingParams
from nanovllm.engine.scheduler import Scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", choices=["original", "decode_aware", "adaptive", "adaptive_v2"],
                        default=["original", "decode_aware", "adaptive"])
    parser.add_argument("--kv-modes", nargs="+", choices=["full", "full_admission", "incremental", "incremental_admission"],
                        default=["full"])
    parser.add_argument("--kv-blocks", type=int, default=-1)
    args = parser.parse_args()
    engine = LLM(args.model, max_model_len=32768, max_num_batched_tokens=2048,
                 max_num_seqs=8, gpu_memory_utilization=0.45, tensor_parallel_size=1,
                 num_kvcache_blocks=args.kv_blocks)
    tokenizer = engine.tokenizer
    cases = [(1024, "What is 17 + 25? Answer with only the number.", "42"),
             (4096, "What is the capital of France? Answer with only the city name.", "Paris"),
             (8192, "What is 8 times 7? Answer with only the number.", "56"),
             (16384, "Write the word apple and nothing else.", "apple")]
    requests = []
    for length, question, expected in cases:
        prefix = tokenizer.encode("<|im_start|>user\nBackground notes: ")
        suffix = tokenizer.encode("\nIgnore the background notes and answer this question. " + question + "<|im_end|>\n<|im_start|>assistant\n")
        filler = tokenizer.encode("The library contains books about science and history. ")
        remaining = length - len(prefix) - len(suffix)
        tokens = prefix + (filler * (remaining // len(filler) + 1))[:remaining] + suffix
        requests.append(tokens)
    rows = []
    reference = None
    # Include all candidate chunk sizes. These are correctness runs, not timing comparisons.
    policy_budgets = {"original": [2048], "decode_aware": [1024], "adaptive": [256, 512, 2048],
                      "adaptive_v2": [256, 512, 2048]}
    for selected, budget, kv_mode in [(p, b, k) for p in args.policies for b in policy_budgets[p] for k in args.kv_modes]:
        policy = "prefill_first" if selected == "original" else selected
        config = copy(engine.config)
        config.scheduler_policy = policy
        config.max_num_batched_tokens = budget
        config.kv_allocation = "incremental" if kv_mode.startswith("incremental") else "full"
        config.kv_admission = kv_mode.endswith("_admission")
        engine.scheduler = Scheduler(config)
        outputs = []
        for tokens in requests:
            outputs.extend(engine.generate([tokens], SamplingParams(temperature=0, max_tokens=16), use_tqdm=False))
        assert not engine.scheduler.block_manager.used_block_ids, "Serial generation leaked owned KV blocks"
        assert all(block.ref_count == 0 for block in engine.scheduler.block_manager.blocks), "Serial generation leaked KV references"
        prefix_cached_tokens = engine.scheduler.stats["prefix_cached_tokens"]
        if reference is None:
            reference = [o["token_ids"] for o in outputs]
        for (length, question, expected), output, ref in zip(cases, outputs, reference):
            answer = tokenizer.decode(output["token_ids"], skip_special_tokens=True).strip()
            row = {"policy": policy, "budget": budget, "kv_mode": kv_mode, "mode": "serial",
                   "group_prefix_cached_tokens": prefix_cached_tokens, "input_length": length, "question": question,
                   "expected": expected, **output, "answer": answer, "semantic_match": answer == expected,
                   "token_match": output["token_ids"] == ref}
            rows.append(row)
            print(json.dumps(row), flush=True)
        # Concurrent real prompts exercise phase interleaving as well as chunk boundaries.
        engine.scheduler = Scheduler(config)
        outputs = engine.generate(requests, SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)
        assert not engine.scheduler.block_manager.used_block_ids, "Concurrent generation leaked owned KV blocks"
        assert all(block.ref_count == 0 for block in engine.scheduler.block_manager.blocks), "Concurrent generation leaked KV references"
        prefix_cached_tokens = engine.scheduler.stats["prefix_cached_tokens"]
        for (length, question, expected), output, ref in zip(cases, outputs, reference):
            answer = tokenizer.decode(output["token_ids"], skip_special_tokens=True).strip()
            row = {"policy": policy, "budget": budget, "kv_mode": kv_mode, "mode": "concurrent",
                   "group_prefix_cached_tokens": prefix_cached_tokens, "input_length": length,
                   "question": question, "expected": expected, **output, "answer": answer,
                   "semantic_match": answer == expected, "token_match": output["token_ids"] == ref}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    engine.exit()
    assert all(row["semantic_match"] for row in rows), "A generated answer is incorrect; see saved outputs"
    assert all(row["token_match"] for row in rows), "Chunk sizes changed generated tokens; see saved outputs"


if __name__ == "__main__":
    main()
