# NanoSched 调度实现

基于 nano-vLLM `bb823b3`，复用 Paged KV、Prefix Cache、Chunked Prefill、FlashAttention 和 CUDA Graph。核心策略在 `nanovllm/engine/scheduler.py`，参数在 `nanovllm/config.py`。

## 三种策略

- `prefill_first`：保持上游 Prefill 优先顺序；CLI 别名为 `original`。公共实现包含已有模型适配、计数和空批次恢复，因此是原版调度策略对照，不等于逐字节上游代码。
- `decode_aware`：有待处理 Prefill 时，最多连续执行 N 个 Decode 批次，然后服务一个 Prefill 批次，默认 N=1。活跃 Decode 存在时使用固定 `decode_prefill_token_budget`；无 Decode 时使用完整 Budget。Decode 请求按轮转次序推进。
- `adaptive`：使用相同分阶段执行路径，动态选择 Budget 和最多连续 Decode 次数。没有混合 Attention。

## 自适应反馈

每个 GPU 批次结束、采样 Token 已同步到 CPU 后，`_observe_step()` 更新以下指数移动平均（新观测权重 0.2）：

1. Decode 批次内已产生过 Token 的请求中，最大实际 Token 间隔。
2. Decode 批次耗时和该批次序列数。
3. Prefill 批次耗时除以实际计算 Token 数。

目标 `target_tpot_s` 由用户配置，默认 0.1 秒只是默认控制目标。控制器从最小候选开始：Decode 间隔达到目标的 90% 时降一档，低于 70% 时升一档，形成滞回。默认候选为 256、512、1024、2048，并受总 Token Budget 限制。

下一次 Prefill 的预算还受估计时间余量约束：

```
decode_cost = recent_decode_step_time * max(1, active_decodes / recent_decode_batch_size)
headroom = max(0, 0.9 * target_tpot - decode_cost)
budget = min(feedback_candidate, largest_candidate_fitting(headroom / prefill_time_per_token))
```

无候选可满足估计时仍使用最小 Chunk，以便 Prefill 继续推进；无活跃 Decode 时使用最大候选。线性耗时估计只用于低开销控制，不作为精确 GPU 延迟模型。

通常在两个 Prefill 批次之间服务一次 Decode；最近 Decode 间隔超过目标时允许两次。队首 Prefill 等待超过 `prefill_max_wait_s` 后，恢复为一次 Decode 后服务 Prefill。每次部分 Prefill 后更新该请求最近服务时间，因此这个上限约束的是队首两次服务之间的等待，不是整个排队时间或 TTFT。

控制器在每次独立回放前重置。固定策略不使用反馈结果。`snapshot()` 与 Step Trace 暴露所选 Budget、阶段计数及 KV 使用情况。

## 正确性边界

中间 Prefill Chunk 不产生输出 Token；最终 Chunk 才产生首 Token。BlockManager 保持上游完整 Prompt 预留、引用计数和链式 Hash 的实现，抢占释放 Block Table，恢复重新计算缺失 KV。

资源可用时，有界交替防止队首 Prefill 和活跃 Decode 相互无限饿死。KV 不足、不可中断的 GPU 批次及外部排队都可能使墙钟 SLO 无法满足。没有实现硬实时保证、KV 水位准入或增量预留。

Qwen2.5 显式使用 QKV bias、无 QK Norm；Qwen3 单独配置 Norm。Greedy 对原始 logits 取 argmax。BF16 Batch/Chunk 改变可产生近似并列 Token 的舍入差异，GPU 验证和历史差异见性能报告。
