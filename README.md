# NanoSched — SLO-Aware LLM Inference Scheduling

基于 Xingkai Yu（GeeeekExplorer）的 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)，沿用 MIT License。在 NVIDIA H20 上研究可配置 TPOT 目标下的 Prefill/Decode 调度取舍，提供可运行推理入口、持续到达回放和真实性能报告。

## 来源与独立贡献

上游已有 Paged KV、Prefix Cache、基础 Chunked Prefill、抢占重算、FlashAttention、CUDA Graph 和 Tensor Parallelism。它们不是本项目的新贡献。

NanoSched 的核心增量是：

- **Decode-Aware**：有界交替服务 Decode 与 Prefill，保留固定 Chunk 作为消融。
- **Adaptive Prefill Budget**：结合最近 Decode Token 间隔、Prefill 每 Token 耗时、活跃 Decode 数和队首 Prefill 等待时间，选择 256/512/1024/2048 Token Budget。TPOT 目标可配置；控制器不承诺硬实时 SLO。
- **可复现测量**：独立到达线程、并发准入上限、逐请求/Token/Step 时间、相同请求集三策略对照及调度 CPU 耗时。
- Qwen2.5/Qwen3 显式架构分派、Greedy 采样和必要的模型/调度/KV 正确性检查。原版在当前 Qwen2.5 checkpoint 上也能生成；不将其表述为首次支持该模型。

上游 [PR #276](https://github.com/GeeeekExplorer/nano-vllm/pull/276) 涉及混合 Chunked Prefill/Decode、Decode 优先和 KV 水位控制。本项目保持独立的 Prefill/Decode 批次，贡献是 **SLO 反馈选取 Budget 与阶段时机，以及对固定策略的 H20 消融**；没有实现混合 Attention 或 KV 水位准入。

## 推理

在已具备 CUDA/PyTorch 的开发环境中，从仓库根目录安装 Python 依赖：`pip install -e . pytest`。本机已有依赖，可直接使用下面的入口。

本次实测模型为 Qwen2.5-7B-Instruct，BF16、TP=1、单张 H20。已有环境使用 `scripts/run.sh`；其他机器可在安装好项目依赖的环境中直接用 `python` 替代该入口。脚本支持 `NANOSCHED_PYTHON`、`NANOSCHED_FLASH_ROOT`、`NANOSCHED_CUDA_COMPAT` 三个本机路径覆盖项。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh example.py \
  --model /path/to/Qwen2.5-7B-Instruct \
  --policy adaptive --target-tpot 0.1 \
  --prompt 'What is 17 + 25? Answer with only the number.'
```

`--policy` 可选 `original`、`decode_aware`、`adaptive`。Python 接口保持 `LLM.generate()`：

```python
from nanovllm import LLM, SamplingParams

llm = LLM('/path/to/Qwen2.5-7B-Instruct', scheduler_policy='adaptive',
          target_tpot_s=0.1, prefill_max_wait_s=0.5,
          max_num_batched_tokens=2048, max_model_len=32768,
          max_num_seqs=32, gpu_memory_utilization=0.45, tensor_parallel_size=1)
outputs = llm.generate(['Explain continuous batching.'], SamplingParams(temperature=0, max_tokens=64))
llm.exit()
```

## H20 Benchmark

相同种子和参数产生完全相同的请求集。`original` 在共同模型实现上使用上游 Prefill-First 策略，便于隔离调度影响；逐字节原版另有生成验证。

```bash
for policy in original decode_aware adaptive; do
  CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh -m benchmarks.replay \
    --model /path/to/Qwen2.5-7B-Instruct --policy "$policy" \
    --workload competition --input-length 4096 --output-length 64 \
    --arrival-rate 4 --concurrency 16 --num-requests 12 \
    --budget 2048 --decode-prefill-budget 1024 --target-tpot 0.1 \
    --seed 20261023 --repeats 3 --output "reports/my-run-$policy"
done
```

`target-tpot` 单位为秒。`concurrency` 限制引擎内未完成请求数，超额到达在外部排队，其等待计入延迟。`max-seqs` 限制单次模型批次。默认开启 CUDA Graph，`--eager` 可关闭。可通过 `--workload-file` 重放保存的请求集。

完整矩阵涵盖 1K/4K/8K/16K、短请求、长短混合、Prefill 竞争、并发与到达率消融、KV 压力，每配置预热后测量三次：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh -m benchmarks.suite \
  --model /path/to/Qwen2.5-7B-Instruct --output reports/my-matrix
```

原始结果包括 `workload.json`、`environment.json`、`trace-*.json`、`requests-*.csv`、`tokens-*.csv`、`summary.json`。Suite 自动输出 `results.json`、`results.csv`、`table.md`。单独重新汇总：

```bash
bash scripts/run.sh -m benchmarks.report --suite reports/my-matrix
```

真实结果与收益/退化分析见 [Performance Report](reports/adaptive/performance.md)，完整数表见 [三策略矩阵](reports/adaptive/matrix/table.md)。其中 8K / 2 req/s / 并发 16 的对照中，自适应将 P95 TPOT 从 161 ms 降到 75 ms；吞吐从 74.79 降到 66.91 tok/s，TTFT P95 从 5.59 增到 7.88 s。低速负载没有明显收益。此前 Decode-Aware 实验保留在 `reports/decode_aware`，早期诊断保留在 `reports/phase01_20261022`。

## 正确性与限制

```bash
bash scripts/run.sh -m pytest tests -q
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh -m benchmarks.validate_generation \
  --model /path/to/Qwen2.5-7B-Instruct --output generation-validation.json
```

GPU 生成检查覆盖串行与并发、1K–16K Prompt、多种 Chunk 和三种策略，40 组答案及完整 Token ID 对照通过；CPU 测试覆盖调度反馈、服务边界、Block 引用计数、Prefix Cache 和抢占恢复。

- 性能负载使用合成随机 Token、固定输出长度；生成正确性另用自然语言问题检查。测得的收益依赖到达率、长度、并发和目标 SLO。
- 动态 Batch / Chunk 会改变 BF16 舍入，不能承诺所有请求逐 Token 一致。少量问题正确和固定批次 Token 一致也不能替代完整模型质量评估。历史数值差异保留。
- KV 仍按完整 Prompt 预留，抢占仍会重算；没有增量 KV 预留或新的 KV 分配算法。
- Prefill 等待保护保证资源可用时队首获得一次 Chunk 服务机会，不保证整个请求 TTFT。GPU 批次不可中断，最小 Chunk 也可能超过目标。
- 实测仅 Qwen2.5-7B-Instruct/BF16/TP=1；Qwen3 有结构测试，本轮没有多卡或其他模型性能结论。
- 最大上下文配置为 32768，覆盖 16K 输入加输出；未增加 32K 输入实验。DCA、Hybrid、非默认 RoPE scaling 不在支持范围。
- 使用进程内 Python 推理接口，时间为 CPU 可见 Token 时刻；没有 HTTP 网络服务。上游 NCCL 端口仍为 2333，Benchmark 应串行运行。

调度细节见 [架构](docs/architecture.md)，指标定义见 [测量方法](docs/benchmark_methodology.md)。
