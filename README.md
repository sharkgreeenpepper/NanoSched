# NanoSched

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的轻量级推理调度实验项目。上游作者 **Xingkai Yu（GeeeekExplorer）**，沿用 MIT License；Paged Attention、Prefix Cache、Chunked Prefill 和 CUDA Graph 来自上游。

当前重点是 Decode-Aware Scheduling 的 H20 性能实验：在可复现的长短请求竞争下，比较 Decode 停顿、TTFT、TPOT 和吞吐。已有 Qwen2.5 适配、Greedy 路径和原版基线。

## 新增能力

- 按架构显式分派 Qwen2.5 / Qwen3；检查全部 Packed 权重和分片索引。
- `temperature=0` 的 Greedy 采样，支持同批次混合采样。
- 可选 Decode 优先、有界交替保护 Prefill、两档 Token Budget 和调度状态记录；默认保留 Prefill-First。
- 保存逐请求、逐 Token 和逐 Step 时间，持续到达负载回放、重复实验、CUDA Graph 对照及 GPU 监控。
- Scheduler、Block RefCount、Prefix Cache、Chunk 边界、抢占恢复和采样回归测试。

**尚未实现：** Adaptive Chunked Prefill、Incremental KV Reservation、KV-Aware Admission Control、混合 Prefill/Decode Attention、HTTP 服务。

## 模型状态

| 模型 | 当前状态 |
| --- | --- |
| Qwen2.5-7B-Instruct | 实测 BF16 / H20 / TP=1；339/339 权重完整加载；Greedy 对照存在数值差异，详见报告 |
| Qwen3 Dense CausalLM | 保留实现，有结构测试；本轮没有该模型 GPU 结果 |
| Qwen2.5-7B-Instruct-1M | 需要 DCA，显式拒绝 |
| Qwen2.5-7B-Instruct-1M-context256k-noDCA | 未验证，不宣称支持其长上下文能力 |
| Qwen3.5-9B / 目录名 Qwen3.8-27B | 实际为 Qwen3.5 Hybrid 架构，本轮不支持 |

## 环境与运行

本次复用服务器已有 Python 3.10、PyTorch 2.13.0+cu129、Transformers 5.17.0、Triton 3.7.1、FlashAttention 2.7.4.post1，缺少的 xxhash 只装入仓库 `.runtime-deps`。没有覆盖共享环境。

`scripts/run.sh` 是本机环境入口；换机器应设置 `NANOSCHED_PYTHON`、`NANOSCHED_FLASH_ROOT`、`NANOSCHED_CUDA_COMPAT`，或在已经具有项目依赖的环境中直接运行 Python。运行前用 `nvidia-smi` 确认所选 GPU 空闲。

如果复用的环境也缺少 xxhash，可单独安装到仓库目录：`bash scripts/run.sh -m pip install --no-deps --target .runtime-deps xxhash==4.0.2`。测试另需 pytest；本轮复用已安装的 9.1.1。

```bash
# CPU 回归，不初始化 GPU 引擎
bash scripts/run.sh -m pytest tests -q

# 指定一张空闲 GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh -m benchmarks.replay \
  --policy decode_aware --workload competition --eager \
  --repeats 3 --output reports/replay-new
```

结果路径必须不存在，防止覆盖实验记录。默认模型目录为 `/data1/fangdongqing/infra/models/Qwen/Qwen2.5-7B-Instruct`，可用 `--model` 覆盖。前期 Transformers 数值对照和原始失败记录保留在历史报告目录中。

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen2.5-7B-Instruct", tensor_parallel_size=1,
          enforce_eager=True, scheduler_policy="decode_aware",
          max_consecutive_decodes=1,
          max_num_batched_tokens=1024, decode_prefill_token_budget=1024)
outputs = llm.generate(["What is 17 + 25?"], SamplingParams(temperature=0, max_tokens=32))
llm.exit()
```

## 实验与复现

原始版本 `bb823b3`、适配基线 `2e1e4e6` 均有独立 detached worktree。正式比较使用相同模型、请求集合、采样参数与 Budget，每配置预热后重复三次。

```bash
for policy in prefill_first decode_aware; do
  CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh -m benchmarks.replay \
    --policy "$policy" --decode-limit 1 --repeats 3 \
    --output "reports/${policy}-new"
done
```

实验数据与失败记录保存在 [reports/phase01_20261022](reports/phase01_20261022)。正式性能报告、版本、统计和实际限制在该目录的 `report.md`。工程细节见 [架构](docs/architecture.md) 和 [测量方法](docs/benchmark_methodology.md)。

当前有界交替策略默认每次 Prefill 之间服务 1 次 Decode，删除了时间老化分支。H20 / Qwen2.5-7B / BF16 / TP=1 / CUDA Graph On 的竞争回放，三次运行均值（同一版本 `ca15b43`，原始数据：[Prefill-First](reports/decode_aware/prefill_first_graph/summary.json)、[Decode-Aware](reports/decode_aware/decode_aware_graph/summary.json)）：

|指标|适配基线 Prefill-First|Decode-Aware|
|---|---:|---:|
|最长 Token 间隔|5.774 s|0.143 s|
|每请求 TPOT P95|332.0 ms|103.3 ms|
|TTFT P95|5.452 s|5.865 s|
|E2E P95|6.210 s|6.274 s|
|输出吞吐|125.58 tok/s|125.46 tok/s|

Graph 吞吐基本持平，TTFT P95 增加约 7.6%，E2E P95 增加约 1.0%。Eager 吞吐下降约 1.4%。调度、Chunk、Block 引用和抢占的 13 项相关回归通过。Prefill-First 原版连续抢占 Decode 的问题可以复现；本轮通过有界交替消除了多秒停顿，没有修改 Attention 或 KV 分配算法。

## 已知限制

BF16 Greedy 输出在不同 Batch 下可能不同。首个计时回放的逐 Token 一致数为 Eager 17/21、Graph 18/21；固定前缀、保持 Prefill-First 只改变 Batch 大小的实验复现了其中一个分歧，详见 [复现代码](reports/decode_aware/batch_shape_repro.py)。另将两个策略的 Batch 均固定为 1，同一组 21 请求的全部输出 Token 一致。前期设置的 2% logits 阈值仅是实验记录，不作为 Phase 0 的额外验收条件。Decode 优先也可能增加 TTFT 或降低吞吐，必须按负载报告退化。

KV 仍按完整 Prompt 预留，极端压力可能无法推进。仅 TP=1 有实测，多卡和更多模型未经验证。等待请求可分配 KV 时，队首 Prefill 在最多 N 个 Decode 调度步后获得服务；该界限不是墙钟延迟承诺。同步回放计入计划到达至实际入队的延迟，不包含网络服务开销。固定 NCCL 端口仍为 2333。
