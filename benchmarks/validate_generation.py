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
    args = parser.parse_args()
    engine = LLM(args.model, max_model_len=32768, max_num_batched_tokens=2048,
                 max_num_seqs=8, gpu_memory_utilization=0.45, tensor_parallel_size=1)
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
    for policy, budget in [("prefill_first", 2048), ("decode_aware", 1024), ("adaptive", 256), ("adaptive", 512), ("adaptive", 2048)]:
        config = copy(engine.config)
        config.scheduler_policy = policy
        config.max_num_batched_tokens = budget
        engine.scheduler = Scheduler(config)
        outputs = []
        for tokens in requests:
            outputs.extend(engine.generate([tokens], SamplingParams(temperature=0, max_tokens=16), use_tqdm=False))
        if reference is None:
            reference = [o["token_ids"] for o in outputs]
        for (length, question, expected), output, ref in zip(cases, outputs, reference):
            row = {"policy": policy, "budget": budget, "input_length": length, "question": question,
                   "expected": expected, **output, "semantic_match": output["text"].strip() == expected,
                   "token_match": output["token_ids"] == ref}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    engine.exit()
    assert all(row["semantic_match"] for row in rows), "A generated answer is incorrect; see saved outputs"
    assert all(row["token_match"] for row in rows), "Chunk sizes changed generated tokens; see saved outputs"


if __name__ == "__main__":
    main()
