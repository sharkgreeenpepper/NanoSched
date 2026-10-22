# Phase 0 / Phase 1 基准方法

## 版本与控制变量

- 逐字节上游：`bb823b3`，独立只读 detached worktree `NanoSched-baseline-bb823b3`。
- 适配基线：`2e1e4e6`，独立 detached worktree `NanoSched-baseline-phase0`；保留原版 Scheduler，包含 Greedy、严格权重校验和 BF16 数值路径调整。
- 正式比较记录各个 `environment.json` 的引擎 Commit、tracked diff Hash、Harness Hash、模型 Config Hash、依赖版本、GPU 和运行参数。修改模型数学路径后必须重新测量，不能将它的收益归于调度。
- Qwen2.5-7B-Instruct、BF16、TP=1、单 H20。Token Budget=1024、序列 Batch 上限=32、最大上下文=8192、GPU Memory Utilization=0.45。
- 随机种子 20261022；请求列表保存为 `workload.json`，对照必须具有相同 SHA256。
- 每配置执行一次算子/Decode 预热、一次完整负载回放预热，随后测量三次。每次重置 Prefix Cache 元数据及随机种子。保留初次不完整预热的探索结果，不纳入正式对照。
- GPU 独占检查在运行前执行；工具只启动到指定的单张空闲 GPU，不终止其他用户进程。`gpu_telemetry.csv` 每 500 ms 记录 GPU 利用率、功耗和显存。

## 工作负载

| 名称 | 请求与到达 |
| --- | --- |
| short | 8 请求同时到达，128 输入 / 128 输出 |
| long | 4 请求同时到达，4096 输入 / 128 输出 |
| competition | 1 个 128 输入 / 128 输出的已有请求；50 ms 后开始每 25 ms 到达一个 4096 输入 / 16 输出请求，共 12 个；另有 8 个 128 输入 / 64 输出短请求，每 40 ms 到达 |

Prompt 为固定随机 Token ID，不包含视觉或特殊长上下文模型能力。忽略 EOS，固定输出长度。重复 Prefix 命中通过模型正确性对照和 CPU 回归验证，不在本轮吞吐比较里人为制造缓存收益。long 为静态批次对照，competition 为持续到达的回放。

## 时间戳与指标

`trace-*.json` 保留请求、每个 Token 事件和调度步骤：

- 回放的 `submitted_at` 是相对负载起点的**计划到达/逻辑提交时间**，不是网络发送时间。引擎在每次 GPU Step 返回后实际将到期请求加入队列，记为 `enqueued_at`。两者差值是同步 Step 粒度产生的注入延迟，计入 TTFT，避免只从入队开始计时而隐藏到达等待。没有声称运行了独立 HTTP 客户端。
- `token_times[0]` 是首个 CPU 可见输出 Token 的时间，最后一个也是请求 `finished_at`。ModelRunner 的 `.tolist()` 已完成 GPU 同步；Prefill 中间 Chunk 的样本不算输出。
- `TTFT = first_token - submitted_at`。
- 每请求 `TPOT = (last_token - first_token)/(output_tokens-1)`，只有一个输出 Token 的请求没有 TPOT。先计算每请求 TPOT，再在请求之间取 P50/P95/P99。
- `inter_token_gap` 对所有实际相邻输出 Token 间隔取分位数，另报最坏间隔。它与每请求平均 TPOT 不同，二者同时保留。
- `E2E = finished_at - submitted_at`。
- Output Throughput = 完成的输出 Token / 整段回放耗时；Request Throughput = 完成请求数 / 耗时；SLO Goodput = 同时满足 TTFT 和平均 TPOT SLO 的请求数 / 耗时。不能互相替代。
- 本轮预设 SLO：TTFT <= 0.5 s 且平均 TPOT <= 0.02 s；不把所有配置 Goodput=0 改写成优化收益。
- KV 利用率是调度时刻已被引用的 Block / 已分配 Pool 的 Block；不等同于 CUDA 总显存利用率。Prefill Chunk 分布逐请求记录，不是批次总 Token 数。

正式报告给出三次运行的均值和样本标准差（ddof=1），不是把三次的请求混在一起计算一个分位数。只有三次重复，不能据此断言统计显著性。Profiler 的诊断运行单独保存，绝不混入计时结果。

## 正确性与限制

`benchmarks.correctness` 实际加载 Transformers 与 NanoSched，比较自然 EOS、Greedy Serial、Batch、重复 Prefix 命中和跨 Chunk 生成；并强制使用同一 Token 历史比较缓存 Decode logits。独立保留逐 Token 精确匹配与固定的数值门槛（相对 L2 <= 0.02 且参考 Top-1 位于引擎 Top-5），门槛未通过时程序返回非零。

不同 Batch/Attention Kernel 的 BF16 输出可能在近似并列 Token 处出现分歧；这必须报告，不能通过忽略参数、删除失败样本或放宽原门槛宣称通过。完整 Phase 4 的 1K–32K、多并发、KV 高压、Prefix 命中与 SLO 曲线矩阵均不属于本轮交付。
