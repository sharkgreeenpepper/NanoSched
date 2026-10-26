import atexit
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self._rendezvous_dir = None
        if config.init_method is None:
            self._rendezvous_dir = TemporaryDirectory(prefix="nanovllm-")
            config.init_method = (Path(self._rendezvous_dir.name) / "store").as_uri()
        shared_memory_name = f"nanovllm-{uuid4().hex}"
        self._closed = False
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event, shared_memory_name))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events, shared_memory_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.exit)
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
        if self._rendezvous_dir is not None:
            self._rendezvous_dir.cleanup()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt or any(not isinstance(t, int) or not 0 <= t < self.config.hf_config.vocab_size for t in prompt):
            raise ValueError("Prompt must contain valid, nonempty token IDs")
        if len(prompt) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError("Prompt plus output exceeds max_model_len")
        # The last sampled token does not need a KV slot; all preceding tokens do.
        if len(prompt) + sampling_params.max_tokens - 1 > self.config.num_kvcache_blocks * self.config.kvcache_block_size:
            raise ValueError("Prompt plus output cannot fit in the KV pool")
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def cancel_request(self, seq_id: int) -> bool:
        return self.scheduler.cancel(seq_id)

    def step(self):
        result = self.step_with_events()
        return result["outputs"], result["num_tokens"]

    def step_with_events(self):
        """Run one batch and expose only tokens actually appended by postprocess."""
        started_at = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        scheduled_at = perf_counter()
        before = [seq.num_completion_tokens for seq in seqs]
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        produced_at = perf_counter()
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        processed_at = perf_counter()
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        events = []
        for seq, previous in zip(seqs, before):
            if seq.num_completion_tokens == previous:
                continue
            reason = None
            if seq.is_finished:
                reason = "stop" if not seq.ignore_eos and seq.last_token == self.config.eos else "length"
            events.append({"seq_id": seq.seq_id, "token_id": seq.last_token,
                           "index": seq.num_completion_tokens - 1, "produced_at": produced_at,
                           "finish_reason": reason})
        return {"outputs": outputs, "num_tokens": num_tokens, "tokens": events,
                "scheduled": [seq.seq_id for seq in seqs], "scheduled_at": scheduled_at,
                "timings": {"scheduler_s": scheduled_at - started_at,
                            "model_runner_s": produced_at - scheduled_at,
                            "postprocess_s": processed_at - produced_at}}

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("Each prompt needs one SamplingParams")
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
