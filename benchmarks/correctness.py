"""Compare real local Qwen2.5 greedy generations against Transformers BF16/SDPA."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/data1/fangdongqing/infra/models/Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--policy", default="prefill_first")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(20261022)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts = ["What is 17 + 25? Answer with the number only.",
             "请用一句话介绍中国的首都。", "List the first five prime numbers.",
             "Context: " + "The red fox lives near the river. " * 150 + "\nWhat animal lives near the river?"]
    prompts = [tokenizer.encode(tokenizer.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)) for t in texts]
    report = {"model": args.model, "dtype": "bfloat16", "gpu": torch.cuda.get_device_name(0),
              "temperature": 0, "reference": "Transformers AutoModelForCausalLM, SDPA, greedy",
              "prompt_lengths": list(map(len, prompts)), "prompt_ids": prompts, "max_tokens": 24,
              "cuda_graph": not args.eager, "policy": args.policy}
    start = time.perf_counter()
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    report["reference_initialization_s"] = time.perf_counter()-start
    gen = GenerationConfig(do_sample=False, max_new_tokens=24, eos_token_id=None, pad_token_id=tokenizer.pad_token_id)
    expected = []
    with torch.inference_mode():
        for prompt in prompts:
            inputs = torch.tensor([prompt], device="cuda")
            output = hf.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=gen)
            expected.append(output[0, len(prompt):].tolist())
    del hf, inputs, output
    gc.collect()
    torch.cuda.empty_cache()
    from nanovllm import LLM, SamplingParams
    engine = LLM(args.model, max_model_len=4096, max_num_batched_tokens=256, max_num_seqs=8,
                 gpu_memory_utilization=0.4, enforce_eager=args.eager, scheduler_policy=args.policy)
    report["weight_load"] = engine.model_runner.weight_load_report
    sp = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)
    serial = [engine.generate([prompt], sp, use_tqdm=False)[0] for prompt in prompts]
    # Reuses cached prefixes, crosses prefill chunks, and changes the decode batch size.
    batch = engine.generate(prompts, sp, use_tqdm=False)
    repeated = engine.generate(prompts, sp, use_tqdm=False)
    report["cases"] = []
    for i, (ref, one, many, again) in enumerate(zip(expected, serial, batch, repeated)):
        case = {"case": i, "reference_ids": ref, "serial_ids": one["token_ids"], "batch_ids": many["token_ids"],
                "repeat_ids": again["token_ids"], "reference_text": tokenizer.decode(ref), "engine_text": one["text"],
                "serial_exact": ref == one["token_ids"], "batch_exact": ref == many["token_ids"],
                "repeat_exact": many["token_ids"] == again["token_ids"]}
        report["cases"].append(case)
    report["all_exact"] = all(c["serial_exact"] and c["batch_exact"] and c["repeat_exact"] for c in report["cases"])
    report["model_config_sha256"] = hashlib.sha256((Path(args.model)/"config.json").read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    engine.exit()
    print(json.dumps({"all_exact": report["all_exact"], "weight_load": report["weight_load"], "output": str(args.output)}))
    if not report["all_exact"]:
        raise SystemExit("Greedy outputs diverged: inspect saved token IDs")


if __name__ == "__main__":
    main()
