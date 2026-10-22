# NanoSched Phase 0 / Phase 1 架构

NanoSched 在 Xingkai Yu 的 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 上开发，保留原有 Paged KV、Prefix Cache、Chunked Prefill 和 CUDA Graph 内核路径。本轮没有实现 Adaptive Chunking、Incremental KV Reservation 或 KV Admission Control。

## 请求与模型

`LLMEngine.add_request` 验证 Token ID 和 Prompt + Output 的上下文上限，构造 `Sequence`，加入 `Scheduler.waiting`。`step()` 完成一个纯 Prefill 或纯 Decode 批次；同一批次没有混合 Attention。首次 Prefill 完成时产生第一个输出 Token，后续 Decode 每次产生一个 Token。未完成的 Prefill Chunk 不向调用者产生输出 Token。

模型注册表按 `model_type` 和 `architectures` 分派 Qwen2/Qwen3。Qwen2.5 的 Q/K/V 有 bias、没有 Q/K Norm，Qwen3 的 Norm 与 bias 独立配置。两者复用 Dense Decoder 组件。标准全 Attention、默认 RoPE 可用；Hybrid、Sliding Window、DCA 和非默认 RoPE scaling 显式拒绝。Qwen3 路径保留，但本轮未做该模型的 GPU 验证。

权重加载要求所有 safetensors 分片、索引及 Packed Q/K/V、Gate/Up 子权重完整。唯一允许省略的是经过实际 storage 地址校验的 tied LM Head 别名。Qwen2.5-7B-Instruct 实际加载 339/339 张量，没有跳过权重。

`temperature=0` 对原始 logits 取 argmax，tie 使用最小 Token ID；同批次允许 Greedy 和随机采样。编译的低精度算子保留显式舍入，Residual 先在激活 dtype 相加再归一化。**这不保证不同 GEMM/Attention 后端、Batch 或 Chunk 划分下 Greedy 输出逐字节相同**。真实 Transformers 对照保存在报告中，数值门槛失败必须作为失败报告。

## 调度

`scheduler_policy="prefill_first"` 为默认策略，保留上游 Prefill 优先次序。只有本轮新增的输入错误处理、统计和无空批次恢复路径不同；逐字节原版始终可从独立 worktree 运行。

`scheduler_policy="decode_aware"` 按以下顺序选择阶段：

1. 同时存在 Running Decode 与 Waiting Prefill 时，优先 Decode。
2. 已连续服务 `max_consecutive_decodes` 次 Decode 时，安排一个 Prefill 批次并重置计数。默认 N=1；有等待且资源可用时，两阶段交替推进。
3. Decode 按轮转次序服务受 Batch 上限限制的请求。Prefill 饥饿保护只依赖调度次数，不使用时间老化分支。
4. 有 Decode 时 Prefill Token Budget 为 `min(max_num_batched_tokens, decode_prefill_token_budget)`；没有 Decode 时用完整 Budget。这是 Phase 1 的两档 Budget，并非 Phase 2 的自适应 Chunk 策略。
5. 某阶段因 KV 容量不能推进时，尝试另一阶段；必要时保留原有抢占/重算路径，没有任务可推进时给出明确错误。

资源可用时，队首请求在最多 N 个 Decode 批次后得到一个 Prefill 批次；新请求只追加到等待队尾，因此不会无限推迟已有 Prefill。该界限按调度步计数，引擎不能中断正在执行的 GPU Kernel，也不承诺墙钟延迟。正式对照两种策略均用 1024 Token Budget，排除 Chunk 大小变化的影响。

## KV 生命周期与监控

BlockManager 仍为整个 Prompt 提前保留 Blocks。写完一个完整 Block 后才发布 Prefix Hash；复用时增加 RefCount，释放时递减，归零进入空闲缓存池；最后一个 Prompt Block 仍需计算以产生 logits。抢占释放 Block Table 并保留 Token 历史，恢复时重新查找 Prefix Cache。

`Scheduler.snapshot()` 提供阶段、队列长度、Decode streak、阶段切换、饥饿保护触发次数、Prefix 命中 Token、实际 Prefill/Decode Token、抢占、重复 Prefill Token 与 KV 使用量。重复 Prefill 依据每个请求已计算位置的 high watermark 计数，缓存命中部分不算重复计算。基准对上游采用外部观测，保证不修改原始源码。

## 当前边界

- 仅 TP=1、单 H20、Qwen2.5-7B-Instruct 有本轮实际实验。
- Engine 是进程内接口，仍使用上游固定 NCCL 初始化端口 2333。
- KV Admission、增量预留、KV 高压下的完整进度保证留到 Phase 3；本轮无高压 GPU 性能结论。
- 每个 GPU 批次结束才向同步调用者提供 CPU 可见 Token；没有 HTTP/SSE 服务或网络开销测量。
- BF16 数值对照存在明确未通过项，见报告。CPU 单元测试通过不能替代模型数值验证。
