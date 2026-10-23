"""Compare real local Qwen2.5 greedy generations against Transformers BF16/SDPA."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import importlib.metadata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/data1/fangdongqing/infra/models/Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--policy", default="prefill_first")
    p.add_argument("--reference-attention", choices=["sdpa", "flash_attention_2"], default="sdpa")
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
              "temperature": 0, "reference": f"Transformers AutoModelForCausalLM, {args.reference_attention}, greedy",
              "prompt_lengths": list(map(len, prompts)), "prompt_ids": prompts, "max_tokens": 24,
              "cuda_graph": not args.eager, "policy": args.policy}
    report["torch"] = torch.__version__
    report["transformers"] = importlib.metadata.version("transformers")
    start = time.perf_counter()
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.reference_attention).to("cuda").eval()
    report["reference_initialization_s"] = time.perf_counter()-start
    gen = GenerationConfig(do_sample=False, max_new_tokens=24, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    expected, reference_logits = [], []
    with torch.inference_mode():
        for prompt in prompts:
            inputs = torch.tensor([prompt], device="cuda")
            output = hf.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=gen,
                                 return_dict_in_generate=True, output_logits=True)
            expected.append(output.sequences[0, len(prompt):].tolist())
            reference_logits.append(torch.cat([step.float().cpu() for step in output.logits]))
    del hf, inputs, output
    gc.collect()
    torch.cuda.empty_cache()
    from nanovllm import LLM, SamplingParams
    engine = LLM(args.model, max_model_len=4096, max_num_batched_tokens=256, max_num_seqs=8,
                 gpu_memory_utilization=0.4, enforce_eager=args.eager, scheduler_policy=args.policy)
    report["weight_load"] = engine.model_runner.weight_load_report
    sp = SamplingParams(temperature=0, max_tokens=24, ignore_eos=False)
    serial = [engine.generate([prompt], sp, use_tqdm=False)[0] for prompt in prompts]
    # Reuses cached prefixes, crosses prefill chunks, and changes the decode batch size.
    batch = engine.generate(prompts, sp, use_tqdm=False)
    repeated = engine.generate(prompts, sp, use_tqdm=False)
    # At identical token histories compare logits, so a near-tie divergence does
    # not turn all following positions into comparisons of different contexts.
    original_sampler = engine.model_runner.sampler.forward
    teacher_forced = []
    for prompt, ref, ref_logits in zip(prompts, expected, reference_logits):
        from nanovllm.engine.block_manager import BlockManager
        old = engine.scheduler.block_manager
        engine.scheduler.block_manager = BlockManager(len(old.blocks), old.block_size)
        captured = []
        def force(logits, temperatures):
            queues = engine.scheduler.waiting or engine.scheduler.running
            seq = queues[0]
            if seq.num_cached_tokens+seq.num_scheduled_tokens < len(seq):
                return torch.zeros(1, dtype=torch.long, device=logits.device)
            token = ref[len(captured)]
            captured.append(logits[0].float().cpu())
            return torch.tensor([token], dtype=torch.long, device=logits.device)
        engine.model_runner.sampler.forward = force
        # Route through the forced sampler instead of the all-greedy fast path.
        engine.generate([prompt], SamplingParams(temperature=1, max_tokens=len(ref)), use_tqdm=False)
        checks = []
        for t in range(len(ref)):
            actual, reference = captured[t], ref_logits[t]
            delta = actual-reference
            top = reference.topk(2).values
            relative_l2 = float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(reference))
            checks.append({"position": t, "max_abs_error": float(delta.abs().max()),
                "mean_abs_error": float(delta.abs().mean()), "relative_l2_error": relative_l2,
                "reference_top1": int(reference.argmax()), "engine_top1": int(actual.argmax()),
                "reference_margin": float(top[0]-top[1]),
                "reference_top1_in_engine_top5": int(reference.argmax()) in actual.topk(5).indices.tolist(),
                "within_tolerance": relative_l2 <= 0.02 and int(reference.argmax()) in actual.topk(5).indices.tolist()})
        teacher_forced.append(checks)
    engine.model_runner.sampler.forward = original_sampler
    report["cases"] = []
    for i, (ref, one, many, again) in enumerate(zip(expected, serial, batch, repeated)):
        case = {"case": i, "reference_ids": ref, "serial_ids": one["token_ids"], "batch_ids": many["token_ids"],
                "repeat_ids": again["token_ids"], "reference_text": tokenizer.decode(ref), "engine_text": one["text"],
                "serial_exact": ref == one["token_ids"], "batch_exact": ref == many["token_ids"],
                "repeat_exact": many["token_ids"] == again["token_ids"]}
        case["teacher_forced"] = teacher_forced[i]
        report["cases"].append(case)
    report["all_exact"] = all(c["serial_exact"] and c["batch_exact"] and c["repeat_exact"] for c in report["cases"])
    report["numerical_tolerance"] = "relative L2 <= 0.02 and reference top-1 in engine top-5 at every identical history"
    report["numerical_checks_passed"] = all(t["within_tolerance"] for checks in teacher_forced for t in checks)
    report["model_config_sha256"] = hashlib.sha256((Path(args.model)/"config.json").read_bytes()).hexdigest()
    report["scheduler_snapshot"] = engine.scheduler.snapshot()
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    engine.exit()
    print(json.dumps({"all_exact": report["all_exact"], "numerical_checks_passed": report["numerical_checks_passed"], "weight_load": report["weight_load"], "output": str(args.output)}))
    if not report["numerical_checks_passed"] or not all(c["repeat_exact"] for c in report["cases"]):
        raise SystemExit("Numerical or reproducibility checks failed: inspect saved evidence")


if __name__ == "__main__":
    main()
