# H20 Benchmark 测量方法

## 对照配置

本轮复用 Python 3.10、PyTorch 2.13.0+cu129、Transformers 5.17.0、Triton 3.7.1、FlashAttention 2.7.4.post1。Qwen2.5-7B-Instruct、BF16、TP=1、单张 NVIDIA H20，CUDA Graph 开启。总 Token Budget=2048、模型批次上限=32、上下文上限=32768。普通场景 GPU memory utilization=0.45，KV 压力场景为 0.22；同一场景的所有策略完全一致。

`original`、`decode_aware`、`adaptive` 使用相同模型实现，仅切换调度参数。`original` 表示上游 Prefill-First 顺序，不代表字节级相同的上游源码。独立 checkout `bb823b3` 的正常生成另行保存。

每个配置执行算子预热和一次完整负载预热，再独立测量三次。每次清空 Prefix Cache 元数据、重置控制器和随机种子。保存完整请求集，汇总时直接比较内容及公共配置。没有文件 Hash 或 Checkpoint 指纹框架。

## 持续到达与并发

固定种子生成 Prompt Token ID，按配置到达率等间隔到达。独立线程按计划产生请求，与模型执行线程解耦。`concurrency` 限制引擎内未完成请求，超额请求在外部排队，不能因为引擎忙而暂停负载源。

- `short`：输入长度统一为 1K。
- `mixed`：每四个请求中一个使用指定长 Prompt，其余最多 1K；包含 1K 同质负载作为对照。
- `competition`：首个请求 128 输入、至少 256 输出；后续为指定长度 Prompt、64 输出，观察持续 Prefill 到达对已开始生成的请求的影响。
- `pressure`：16 个 16K 输入请求，在较小 KV Pool 下测试；不是硬件默认显存配置，结果单列。

完整长度、到达率、并发和数量见 `benchmarks/suite.py:CASES`，每次运行的实际命令保存在 `commands.json`。低负载、不同并发以及短请求对照与竞争场景一起保留，不能只报告有收益的场景。

## 时间与指标

全部时间相对该次回放起点，使用 `perf_counter()`：

- `submitted_at`：计划到达时间。
- `arrived_at`：独立线程实际产生请求时间。
- `enqueued_at`：进入引擎等待队列的时间。
- `token_times`：每个输出 Token 在 CPU 可见的时刻；ModelRunner 的 `.tolist()` 已同步 GPU。中间 Prefill Chunk 不计作输出。
- `finished_at`：最后一个输出 Token 的时刻。

TTFT 与 E2E 从计划到达开始计时，包含发生器迟到和外部准入等待；两部分另报 `arrival_lag_p95_s`、`admission_wait_p95_s`，不隐藏排队。

每请求 TPOT 为首 Token 到末 Token 的平均间隔，再在请求之间计算 P50/P95/P99。ITL 对实际相邻 Token 间隔计算分位数，并报告最大间隔。两者不同，SLO 控制器使用最近最大间隔的 EWMA，结果同时给出每请求平均 TPOT 和逐 Token ITL 达标率。

吞吐为总输出 Token 除以从负载开始到全部完成的时间。主矩阵 TPOT 目标为 100 ms、TTFT 目标为 5 s；目标是用户配置，不是性能结论。Goodput 要求两个目标同时满足，TPOT 达标率单独计算。

`Scheduler.schedule()` 通过 CPU 单调时钟计时，排除 Trace 构造；单独记录 `postprocess()`（包含自适应观测、Block Hash 和状态更新）。报告每次 Schedule 均值/P95，以及两者总时间占回放时间比例。它是调度/状态管理整体耗时，不能将策略间差值全部解释为反馈控制器自身耗时。

KV 使用率是被引用 Blocks / Pool Blocks，不等于 GPU 总显存使用率。记录抢占次数、重算 Token 和实际 Chunk 分布。

汇总为三次运行指标的均值及样本标准差，保留每次原始数据。三次重复不支持广泛统计显著性或所有生产负载结论。

## 正确性与限制

自然语言生成验证覆盖 1K/4K/8K/16K Prompt、多种 Chunk、三策略，比较明确答案及完整 Token ID。EOS 在语义答案解码时去除，完整 Token 比较仍包含 EOS。性能负载使用随机 Token 并忽略 EOS，用来控制计算量，不代表模型任务质量。

动态 Batch 可能改变 BF16 舍入，历史数据存在输出分歧；不得将小规模语义验证泛化为所有 Batch 严格一致。旧 Transformers 诊断中的 2% logits 阈值不是本项目新增验收门槛，历史失败保留。

没有网络服务测量，也未测试 32K 输入、多卡或新的模型架构。低显存压力结果不代表所有显存配置。
