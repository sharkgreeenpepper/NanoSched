# NanoSched

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的轻量级推理调度实验项目。上游作者 **Xingkai Yu（GeeeekExplorer）**，沿用 MIT License；Paged Attention、Prefix Cache、Chunked Prefill 和 CUDA Graph 来自上游。

本轮完成 Phase 0 的模型适配、严格权重校验、Greedy 路径和实测基线，并实现 Phase 1 的最小 Decode-Aware Scheduler。模型数值对照存在未通过项，不能将本项目视为已完成生产正确性验证。

## 新增能力

- 按架构显式分派 Qwen2.5 / Qwen3；检查全部 Packed 权重和分片索引。
- `temperature=0` 的 Greedy 采样，支持同批次混合采样。
- 可选 Decode 优先、连续 Decode 限制、Prefill 等待时间保护、两档 Token Budget 和调度状态监控；默认保留 Prefill-First。
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

# 指定一张空闲 GPU；不要照搬其他用户占用的索引
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh -m benchmarks.correctness \
  --eager --output reports/correctness-new.json

CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh -m benchmarks.replay \
  --policy decode_aware --workload competition --eager \
  --repeats 3 --output reports/replay-new
```

结果路径必须不存在，防止覆盖实验记录。正确性门槛失败会保存证据并返回非零。默认模型目录为 `/data1/fangdongqing/infra/models/Qwen/Qwen2.5-7B-Instruct`，可用 `--model` 覆盖。

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen2.5-7B-Instruct", tensor_parallel_size=1,
          enforce_eager=True, scheduler_policy="decode_aware",
          max_consecutive_decodes=4, prefill_max_wait_ms=100,
          max_num_batched_tokens=1024, decode_prefill_token_budget=1024)
outputs = llm.generate(["What is 17 + 25?"], SamplingParams(temperature=0, max_tokens=32))
llm.exit()
```

## 实验与复现

原始版本 `bb823b3`、适配基线 `2e1e4e6` 均有独立 detached worktree。正式比较使用相同模型、请求 Hash、Greedy 采样、Budget 与环境，每配置预热后重复三次。不要直接比较原版随机采样和优化版 Greedy 的吞吐后宣称调度收益。

```bash
git worktree add --detach ../NanoSched-baseline-phase0 2e1e4e6
CUDA_VISIBLE_DEVICES=0 bash benchmarks/run_suite.sh \
  reports/suite-new ../NanoSched-baseline-phase0
```

实验数据与失败记录保存在 [reports/phase01_20261022](reports/phase01_20261022)。正式性能报告、版本、统计和实际限制在该目录的 `report.md`。工程细节见 [架构](docs/architecture.md) 和 [测量方法](docs/benchmark_methodology.md)。

H20 / Qwen2.5-7B / BF16 / TP=1 / CUDA Graph On 的竞争回放，三次运行均值（具体标准差、Commit 和原始数据见 [报告](reports/phase01_20261022/report.md)）：

|指标|适配基线 Prefill-First|Decode-Aware|
|---|---:|---:|
|最长 Token 间隔|5.773 s|0.137 s|
|每请求 TPOT P95|331.8 ms|40.6 ms|
|TTFT P95|5.450 s|6.543 s|
|输出吞吐|125.64 tok/s|114.37 tok/s|

因此本轮改善了 Decode 停顿，也付出了 TTFT 与吞吐退化的代价。CPU 回归 24 项通过；真实 BF16 输出对参考 3/4 提示精确匹配、重复生成 4/4 可重现，严格 logits 门槛仍有失败项。

## 已知限制

相同历史的部分 BF16 logits 未通过固定误差门槛，中文 Greedy 输出与 Transformers 存在分歧；重复生成可重现，但这不等于跨后端精确一致。Decode 优先也可能增加 TTFT、调度次数或降低吞吐，必须按负载报告退化。

KV 仍按完整 Prompt 预留，极端压力可能无法推进。仅 TP=1 有实测，多卡和更多模型未经验证。Prefill 等待保护是批次边界的软阈值；同步回放计入计划到达至实际入队的延迟，不包含网络服务开销。固定 NCCL 端口仍为 2333。
