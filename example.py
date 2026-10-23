"""Generate text with NanoSched or a specified unmodified nano-vLLM checkout."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-root", type=Path)
    parser.add_argument("--policy", choices=["original", "decode_aware", "adaptive"], default="adaptive")
    parser.add_argument("--prompt", default="What is 17 + 25? Answer with only the number.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--target-tpot", type=float, default=0.1, help="Seconds")
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.engine_root:
        sys.path.insert(0, str(args.engine_root.resolve()))
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import torch
    torch.manual_seed(20261023)
    config = dict(tensor_parallel_size=1, max_model_len=32768,
                  max_num_batched_tokens=args.budget, max_num_seqs=8, gpu_memory_utilization=0.45)
    if args.policy != "original":
        config.update(scheduler_policy=args.policy, target_tpot_s=args.target_tpot)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True)
    engine = LLM(args.model, **config)
    outputs = engine.generate([prompt], SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens), use_tqdm=False)
    result = {"model": args.model, "policy": args.policy, "engine_root": str(args.engine_root),
              "prompt": args.prompt, "outputs": outputs}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    import atexit
    atexit.unregister(engine.exit)
    engine.exit()


if __name__ == "__main__":
    main()
