"""Validate one TP configuration offline, then through a fresh HTTP server."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from urllib.request import urlopen

from benchmarks.online_suite import stop_server, wait_ready


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def engine_config(args):
    return {"scheduler_policy": "decode_aware", "tensor_parallel_size": args.tp,
            "enforce_eager": args.eager, "max_num_seqs": 32,
            "max_num_batched_tokens": 2048, "max_consecutive_decodes": 3,
            "decode_prefill_token_budget": 2048, "gpu_memory_utilization": 0.45,
            "max_model_len": 32768}


def offline(args, reference):
    from nanovllm import LLM, SamplingParams

    result = {"model": reference["model"], "tp": args.tp, "eager": args.eager,
              "reference": str(args.reference.resolve()), "config": engine_config(args),
              "cases": [], "errors": []}
    engine = None
    path = args.output / "offline-baseline.json"
    try:
        assert reference["tp"] == 1, "Cross-TP reference must come from TP=1"
        assert len(reference["cases"]) == 5
        assert sum("expected" in case for case in reference["cases"]) == 4
        engine = LLM(reference["model"], **engine_config(args))
        for case in reference["cases"]:
            params = SamplingParams(temperature=0, max_tokens=case["max_tokens"],
                                    ignore_eos=case.get("ignore_eos", False))
            generated = engine.generate([case["prompt"]], params, use_tqdm=False)[0]
            tokens = generated["token_ids"]
            text = engine.tokenizer.decode(tokens, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
            reference_tokens = case["token_ids"]
            equal = tokens == reference_tokens
            mismatch = None if equal else next(
                (i for i, (actual, expected) in enumerate(zip(tokens, reference_tokens)) if actual != expected),
                min(len(tokens), len(reference_tokens)))
            bm = engine.scheduler.block_manager
            released = (engine.is_finished() and not bm.used_block_ids
                        and len(bm.free_block_ids) == len(bm.blocks)
                        and all(block.ref_count == 0 for block in bm.blocks))
            row = {**case, "token_ids": tokens, "text": text,
                   "tp1_token_ids": reference_tokens, "tp1_token_equal": equal,
                   "first_tp1_difference": mismatch, "kv_released": released}
            if "expected" in case:
                row["semantic_match"] = text.strip() == case["expected"]
            result["cases"].append(row)
            save(path, result)
            print(json.dumps({key: value for key, value in row.items()
                              if key not in ("prompt", "token_ids", "tp1_token_ids")}, ensure_ascii=False), flush=True)
            assert released, f"KV ownership leaked after {case['id']}"
            if "expected" in case:
                assert row["semantic_match"], f"Semantic mismatch: {case['id']}: {text!r}"
        result["all_tp1_token_equal"] = all(case["tp1_token_equal"] for case in result["cases"])
        result["semantic_checks_passed"] = True
        if not result["all_tp1_token_equal"]:
            result["numerical_limitation"] = (
                "Generated token IDs differ from TP=1; all four exact semantic answers passed. "
                "This comparison does not establish logit-level numerical equivalence.")
        result["status"] = "passed" if result["all_tp1_token_equal"] else "passed_with_cross_tp_token_differences"
    except BaseException as error:
        result["status"] = "failed"
        result["errors"].append({"error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        try:
            if engine is not None:
                engine.exit()
        except BaseException as error:
            result["status"] = "failed"
            result["errors"].append({"phase": "offline_shutdown", "error": repr(error),
                                     "traceback": traceback.format_exc()})
            raise
        finally:
            save(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("reports/phase4/offline-baseline.json"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--offline-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.offline_only:
        offline(args, reference)
        return
    if (args.output / "commands.json").exists() or (args.output / "offline-baseline.json").exists():
        raise FileExistsError(f"Validation output already exists: {args.output}")
    offline_command = [sys.executable, str(Path(__file__).resolve()), "--tp", str(args.tp),
                       "--output", str(args.output.resolve()), "--reference", str(args.reference.resolve()),
                       "--offline-only"]
    server_command = [sys.executable, "-m", "nanovllm.serving", "--model", reference["model"],
                      "--host", "127.0.0.1", "--port", str(args.port), "--tp", str(args.tp),
                      "--policy", "decode_aware", "--max-seqs", "32", "--budget", "2048",
                      "--decode-limit", "3", "--decode-prefill-budget", "2048",
                      "--memory-utilization", "0.45", "--max-model-len", "32768"]
    if args.eager:
        offline_command.append("--eager")
        server_command.append("--eager")
    url = f"http://127.0.0.1:{args.port}"
    validation_command = [sys.executable, "-m", "benchmarks.validate_online", "--url", url,
                          "--output", str(args.output / "online-validation.json"),
                          "--offline-baseline", str(args.output / "offline-baseline.json"),
                          "--concurrencies", "8", "16", "32", "--long-input-length", "8192"]
    report = {"tp": args.tp, "eager": args.eager, "config": engine_config(args),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "reference": str(args.reference.resolve()), "cwd": str(Path.cwd()),
              "offline": offline_command, "server": server_command, "validation": validation_command,
              "status": "running", "errors": []}
    save(args.output / "commands.json", report)
    phase = "offline"
    server = None
    try:
        # A separate offline process releases its CUDA context and all TP ranks
        # before online initialization; neither model runs alongside the other.
        with (args.output / "offline.stdout").open("w") as stream:
            subprocess.run(offline_command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        baseline = json.loads((args.output / "offline-baseline.json").read_text())
        report["all_tp1_token_equal"] = baseline["all_tp1_token_equal"]
        report["cross_tp_differences"] = [case["id"] for case in baseline["cases"] if not case["tp1_token_equal"]]
        phase = "online"
        with (args.output / "server.stdout").open("w") as stream:
            server = subprocess.Popen(server_command, stdout=stream, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            ready = wait_ready(server, url)
            save(args.output / "server-ready.json", ready)
            with (args.output / "client.stdout").open("w") as client_stream:
                subprocess.run(validation_command, stdout=client_stream, stderr=subprocess.STDOUT, check=True)
            with urlopen(url + "/health", timeout=10) as response:
                final = json.load(response)
            save(args.output / "server-final.json", final)
            assert all(final[key] == 0 for key in ("active", "waiting", "running", "kv_used")), final
            assert final["kv_free"] == final["kv_total"], final
        report["status"] = "passed" if report["all_tp1_token_equal"] else "passed_with_cross_tp_token_differences"
    except BaseException as error:
        report["status"] = "failed"
        report["errors"].append({"phase": phase, "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        try:
            if server is not None:
                stop_server(server)
                report["server_exit_code"] = server.returncode
        except BaseException as error:
            report["status"] = "failed"
            report["errors"].append({"phase": "online_shutdown", "error": repr(error),
                                     "traceback": traceback.format_exc()})
            raise
        finally:
            save(args.output / "commands.json", report)
    print(json.dumps({key: report[key] for key in ("tp", "eager", "status", "cross_tp_differences")}), flush=True)


if __name__ == "__main__":
    main()
