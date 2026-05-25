# dispatch_combine_moe_v2 目标日志

> 本文件只追加新阶段目标；历史目标不回改。

## 2026-05-23：PTO 化主链与方案B two-kernel queue pipeline

将 `dispatch_combine_moe_v2` 从旧三段式 `dispatch -> compute_all -> combine` 收口为 PTO 范式主链，并完成 Vec/Cube target 分离：Vec 负责 dispatch/dequant/SwiGLU/combine/restore，Cube 负责 GMM1/GMM2。先完成 split-stage correctness，再推进到方案B two-kernel ready-queue pipeline：`launchCommVecQueuePipeline` + `launchComputeCubeQueuePipeline`，通过 GM queue 串接 `dispatchToGmm1Queue -> gmm1ToSwiGluQueue -> swiGluToGmm2Queue -> gmm2ToCombineQueue`，并用 public A3 `run.sh --matrix` 闭环。

## 2026-05-23：对齐 dispatch_gmm_combine 的 group/window 级 overlap

参考 `/mnt/data/ntlab/zy/code/zhangyuan/shmem_zy/examples/dispatch_gmm_combine` 的算法调度，但不照搬其非 PTO Catlass/AscendC 编程风格。当前 two-kernel queue pipeline 已具备 correctness 骨架；下一阶段目标是补齐与参考算子的性能级 overlap 差距：

- Dispatch 与 GMM1 真 overlap：不能让 `comm_vec` block0 先完整跑完 `RunDispatchRangeOnlyKernel` 再进入 dequant/queue event-loop；要按 range/window 边 dispatch、边 dequant、边 publish `dispatchToGmm1Queue`。
- GMM2 与 Combine 真 overlap：不能等 `gmm2ToCombineQueue` 到全局 `tileCount` 后再 combine；要按 GMM2 ready entry/window 边到边 combine。
- 多 producer queue：从单 `kPipelineQueueProducerId0` 推进到 per-producer queue + consumer round-robin poll + 每 producer 本地 head，避免多个 producer 写共享 completion/count。
- Cube 多 block 分摊：`compute_cube_queue_kernel` 不应长期只有 block0 工作；要按 AIC block 分摊 GMM tile/window。
- Expert group / epilogue granularity：建立类似参考算子的 group/window 推进策略，让 dispatch/routing、GMM1、SwiGLU、GMM2、combine/restore 按组交接。
- Runtime routing/capacity 语义：在需要时补齐 tokenPerExpert、cumsum、capacity/dropPad 等协议，不局限于 public generated balanced case。

后续任务必须围绕这些缺口推进：不得回退到 host 六段串行 launch，不得破坏 PTO target 分离，不得为了运行方便引入非 PTO 主链。

## 2026-05-23：下一阶段验收原则

算子开发里性能就是功能。Batch 14+ 不要过早追求 correctness 表面通过；每个阶段先做代码检视和结构检视，确认 active path 真的消除了 whole-stage wait、单 block 瓶颈、全量 queue wait 或共享 completion 风险，再选择性跑最小黑盒/public matrix 和性能采样。public correctness PASS 是必要兜底，但不能替代 overlap 证据和性能结构验收。

## 2026-05-24：PTO 风味补齐 dispatch_gmm_combine 的同步与调度差距

Batch 14-20 已完成 two-kernel PTO queue pipeline 和 public A3 correctness/结构验收；下一阶段目标是继续缩小与 `shmem_zy/examples/dispatch_gmm_combine` 的性能级调度差距，但仍只参考算法调度，不照搬其非 PTO AscendC/Catlass 编程风格。

重点缺口和目标：

- CrossCore flag PTO 化边界：调研并封装 PTO `TSync/TPush` 与 `pto::comm::Signal` 的使用边界，明确同 kernel AIC/AIV cross-core event 与跨 kernel GM signal/queue 的适用场景。
- Group-first 调度：把已有 `localExpert/expertRangeIndex/ExpertGroupTaskRange` 从 side metadata 提升为调度主轴，让 dispatch、dequant、GMM1、SwiGLU、GMM2、combine 按 group/window round-robin 推进。
- Vec 侧多 producer：拆解当前 `comm_vec` block0 的 dispatch/dequant/SwiGLU 瓶颈，让 dispatch/dequant producer、SwiGLU producer、combine worker、restore owner 各自有清晰 block ownership。
- PTO FIFO pilot：选择 GMM1->SwiGLU 或 SwiGLU->GMM2 单边界试点 `TPush/TSync`，若 PTO cross-core 只适合同 kernel同步，则先形成 fused pilot 设计，不直接破坏当前稳定 two-kernel 主链。
- Runtime routing/cumsum metadata：补齐 tokenPerExpert、cumsumMM、cumsumSend、expertTokensBeforeCapacity 的 GM metadata/read-only schedule，优先 host materialize，不直接搬参考项目 device routing kernel。
- 最后必须做代码检视和性能对标：以 fixed shape 的 kernel/e2e、producer 数、queue wait 点、Vec/Cube 利用率作为验收抓手，不能只看 correctness PASS。

## 2026-05-24：纠偏为 dispatch_gmm_combine 全链路语义、计算与 overlap 对齐

前一阶段 Batch 21-26 只完成了 PTO two-kernel queue pipeline 的同步抽象、group-first 调度骨架、Vec 多 producer、TPush/TSync pilot 边界和 runtime cumsum metadata；这些只能视为 **PTO 调度骨架/overlap 原型**，不能视为已经对齐 `shmem_zy/examples/dispatch_gmm_combine` 的完整实现。

新的阶段目标是以参考项目的 README/DESIGN/核心实现为依据，重新按 gap matrix 推进：

- 语义对齐：补齐真实 topK/expert_idx/probs、expandedRowIdx、tokenPerExpert、cumsumMM/cumsumSend、capacity/dropPad/maxOutputSize、MoeTokenUnpermute 与 probs weighted sum 语义。
- 计算对齐：从当前 `pto_mmad_shell.hpp` 小 tile/toy matmul，推进到真实 GMM1/GMM2 的 K/N shape、分组 M、int8/fixpipe/per-token/per-channel scale、GMM1 epilogue、SwiGLU sigmoid、dynamic quant 和 scale2。
- combine/unpermute 对齐：补齐 GMM2 输出 dequant、ptrD/remote peer layout、跨 rank combine 后的 token unpermute 与概率加权恢复。
- overlap 对齐：在不照搬非 PTO AscendC/Catlass 风格的前提下，重新映射参考项目 AIV Dispatch/Combine 与 AIC GMM1/GMM2 的 group/window 交叠、ready 边界和 wait 点。
- 性能对齐：public matrix 只能证明当前 toy oracle 自洽；后续必须建立 shmem README shape baseline 与 PTO same-shape 指标，性能与功能同等验收。

执行原则：不是全仓推倒重写；保留已经验证的 host runtime、workspace、PTO target separation、GM queue/sync abstraction 和 route plan 基础设施，但重写语义链、compute micro-kernel 链和 combine/unpermute 主链，避免继续在 toy compute 上做局部修补。

## 2026-05-24：重定位为 dispatch_gmm_combine 的 PTO 风味化翻译

前一阶段在 float GMM1 路径上做局部 tail/coverage 修补，方向偏离——`dispatch_combine_moe_v2` 的本质就是 `dispatch_gmm_combine` 的 PTO 风味化翻译，必须保持原文的性能路径。

新目标是将 v2 的计算链路从 float path 全面切换为 int8 path，对齐 shmem 原文的数据类型链路和性能设计：

核心改动：
- 删除独立 Vec dequant 阶段：dispatch 直接 publish int8 给 Cube，不再 dequant 到 float
- GMM1 改 int8 MMAD + TSTORE_FP (PTO 的 Fixpipe 等价)：int8×int8→int32 L0C → per-channel dequant → fp16 GM
- GMM2 加 TSTORE_FP：同上，删除 gmm2Acc int32 全量 GM + Vec dequant bridge
- weight1 改 int8 + uint64 per-channel scale（对齐 shmem B1+scale1 语义）
- 大 tile 128×256×512 + L1/L0 double-buffer + preload pipeline
- SwiGLU 输入改 fp16 C1 + perTokenScale1 乘回
- Combine 输入改 fp16 C2 + perTokenScale2，直写远端 fp16
- 删除 float 中间 buffer (dequantOut, gmm1Out float, gmm2Acc int32, localPartial float)

前置条件：Phase 1 验证 TSTORE_FP 在 A3 int8→fp16 per-channel 路径可工作。
保留基建：host runtime、HCCL window、dispatch/combine 通信、4 条 GM queue、restore、routing metadata。
不照搬 Catlass blockMmad 代码，但翻译其语义（int8 MMAD、Fixpipe per-channel dequant、大 tile double-buffer）为 PTO 指令集。
