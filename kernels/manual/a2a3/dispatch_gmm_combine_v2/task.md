# dispatch_combine_moe_v2 PTO 化任务进度

> Last refreshed: 2026-05-23

## dispatch_gmm_combine 对齐背景

参考项目：`/mnt/data/ntlab/zy/code/zhangyuan/shmem_zy/examples/dispatch_gmm_combine`。该项目是非 PTO 实现，只能参考算法调度，不能照搬 Catlass/AscendC 编程风格。

参考算子的核心模型是 fused AIC/AIV kernel：AIC 执行 `GMM1 -> GMM2`，AIV 执行 `Dispatch -> Combine`，中间通过 cross-core flag 按 expert group / epilogue granularity 推进。它的 overlap 不是简单 host 多 launch，而是 dispatch/routing/cumsum、GMM、epilogue/SwiGLU、GMM2、combine/unpermute 在 device 内按 group/window 逐步交接。

当前项目已经完成的对齐：

- [x] 计算主链齐全：`dispatch -> dequant -> GMM1 -> SwiGLU -> GMM2 -> combine -> restore`。
- [x] PTO target 分离齐全：Vec kernel 只跑 dispatch/dequant/SwiGLU/combine/restore，Cube kernel 只跑 GMM1/GMM2。
- [x] active host 主链已切为 two-kernel queue pipeline：`launchCommVecQueuePipeline` + `launchComputeCubeQueuePipeline`。
- [x] 四段 GM queue 已接通：`dispatchToGmm1Queue -> gmm1ToSwiGluQueue -> swiGluToGmm2Queue -> gmm2ToCombineQueue`。
- [x] queue entry-before-count 可见性已修复：entry fields 写完并 flush entry cacheline 后再发布 count。
- [x] public A3 `run.sh --matrix` 已通过。

当前仍缺的对齐点：

- [ ] Dispatch 与 GMM1 的真 overlap 还不完整：当前 `comm_vec` block0 仍先跑完整 `RunDispatchRangeOnlyKernel`，再进入 dequant/queue event-loop。
- [ ] Compute cube 仍是单 producer：`compute_cube_queue_kernel` 只有 block0 工作，未按多 AIC block 分摊 GMM tile。
- [ ] Queue 仍是单 producer：当前所有 queue 使用 `kPipelineQueueProducerId0`，还没有多 producer queue + consumer round-robin poll。
- [ ] Combine 仍偏粗粒度：combine work block 当前等待 `gmm2ToCombineQueue` 到 `tileCount` 后再执行，尚未按 GMM2 ready entry/window 边到边 combine。
- [ ] 还没有按参考算子的 expert group / epilogue granularity 建立分组推进策略。
- [ ] 当前 public fixture 覆盖 generated balanced case，尚未对齐参考算子的完整 runtime routing/capacity/dropPad/tokenPerExpert 输入协议。

后续目标必须围绕这些缺口推进，不要回到 host 六段串行 launch，也不要偏离 PTO 范式。

下一阶段验收原则：算子开发里性能就是功能，不要过早追求 correctness 表面通过。每个 batch 优先走 `设计 -> 编码 -> 代码检视/结构检视 -> overlap 证据检视`，确认 active path 真的消除了 whole-stage wait、单 block 瓶颈或全量 queue wait 后，再按需要跑最小黑盒/public matrix 和性能采样。

## 目标

把当前 two-kernel queue pipeline 从 correctness 骨架推进到对齐 `dispatch_gmm_combine` 的 group/window 级 overlap：补齐 dispatch 边生产边 dequant/GMM1、GMM2 边 ready 边 combine、多 producer queue、cube 多 block 分摊，以及必要的 routing/capacity 语义。

## 当前状态

- [x] 已完成本轮现状阅读：`README.md` 原先不存在，`design.md` / `task.md` 原先不存在，只有拼写错误的 `desing.md`。
- [x] 已完成 Batch 14-20：two-kernel PTO queue pipeline、runtime routing semantic guard、source segment 修复、Batch 20 代码检视与 public A3 matrix 验收。
- [ ] 下一阶段 Batch 21-26 已设计：PTO Sync abstraction、group-first scheduler、Vec-side multi-producer、TPush/TSync pilot、runtime cumsum metadata、性能对标收尾。
- [x] 初始阶段已确认代码当时是 `launchDispatchRange -> launchComputeOnly -> launchCombineOnly` 三段 host pipeline；Batch 1 已切到多 stage 主链。
- [x] 初始阶段已确认 CMake 当时是 dispatch/combine Vec kernel + compute Cube kernel 两个 device so；Batch 1 已拆成三个 device so。
- [x] 初始阶段已确认 GMM1/GMM2 已通过 PTO matmul shell，dequant/SwiGLU 当时使用 scalar bridge；Batch 2/4 已替换。
- [x] 初始阶段已确认 dispatch/combine remote path 已用 `TGET/TPUT`，local self-copy 当时使用 scalar loop；Batch 6 已替换。
- [x] 已创建并刷新 `README.md`。
- [x] 已创建并刷新 canonical `design.md`。
- [x] 已创建本 `task.md`。
- [x] 已补充 Batch 14-20 下一阶段详细设计与任务拆分，验收口径改为代码检视/结构检视优先，性能作为功能验收的一部分。

## 待办批次

### Batch 1：ABI/CMake/stage 骨架

- [x] 确认策略：不保留 `launchComputeOnly` 兼容路径，直接切到多 stage 主链。
- [x] 在 `design.md` 细化 Batch 1 ABI、buffer ownership、stage 行为、CMake target、host enqueue、验收标准。
- [x] 拆 `kernel_launch.hpp` 中的 `ComputeOnlyLaunchArgs`，新增 `DequantLaunchArgs/Gmm1LaunchArgs/SwiGluLaunchArgs/Gmm2LaunchArgs`。
- [x] 新增 `op_kernel/compute_vec/compute_vec_kernel.cpp`，承载 Batch 1 dequant/swiglu scalar stage。
- [x] 新增 `op_kernel/compute/gmm_cube_kernel.cpp`，承载 GMM1/GMM2 cube stage。
- [x] 更新 `CMakeLists.txt` 为 dispatch/combine、compute_vec、compute_cube 三个 kernel so，并移出旧 `compute_kernel.cpp` 构建路径。
- [x] 更新 `run.sh` 的 build target 名称。
- [x] 更新 `main.cpp` 为同 stream 多 stage launch。
- [x] 静态检查确认 source/build active path 无 `ComputeOnlyLaunchArgs|launchComputeOnly|dispatch_combine_moe_v2_compute_all`；历史/验收文档仍保留说明文本。
- [x] 跑 `run.sh --matrix` 黑盒验证。

### Batch 2：Dequant Vec

- [x] 已细化 Batch 2 设计到 `design.md`：文件边界、布局、view/type、PTO 主链、tail 行处理、接入点、验收标准。
- [x] 新增 `op_kernel/compute_vec/compute_vec_views.hpp`，定义 Vec dequant padded tile/view 类型和 GM view helper。
- [x] 新增 `op_kernel/compute_vec/dequant_vec.hpp`，实现 `TLOAD -> TCVT -> TCVT -> TROWEXPANDMUL -> TSTORE`，并用 UB compact helper 收口 GM `[8,2]` 布局。
- [x] 在 `compute_vec_kernel.cpp` 接入 `RunDequantTilePto`，保持 `launchDequant` ABI 不变。
- [x] 停用/删除 `RunDequantTile8Scalar` 和任何 `RunDequantTile8CubeBridge` 残留调用。
- [x] 静态检查 `git diff --check` 和 dequant scalar/bridge grep。
- [x] 跑 `run.sh --matrix` 黑盒验证。

### Batch 3：GMM1 Cube

- [x] 已细化 Batch 3 设计到 `design.md`：文件边界、输入输出布局、active PTO 调用链、tail 约束、静态/黑盒验收标准。
- [x] 已核实 `gmm_cube_kernel.cpp` 中 `launchGmm1 -> dispatch_combine_moe_v2_gmm1` 已接入独立 cube stage。
- [x] 已核实 GMM1 active path 为 `dispatch_combine_moe_v2_gmm1 -> RunGmm1Tile8 -> RunPtoMatmul<8, 2, 4> -> RunPtoMatmulStrided<8, 2, 4>`。
- [x] 已核实 `RunPtoMatmulStrided` 内部使用 `TLOAD -> TMOV -> TMATMUL -> TSTORE`。
- [x] 静态检查 `git diff --check`、GMM1 active path grep、matmul shell primitive grep、禁用模式 grep。
- [x] 跑 `run.sh --matrix` 黑盒验证。
### Batch 4：SwiGLU Vec

- [x] 已细化 Batch 4 设计到 `design.md`：文件边界、输入输出布局、view/type、PTO 主链、tail 行处理、接入点、验收标准。
- [x] 在 `compute_vec_views.hpp` 补充 SwiGLU padded tile/view 类型和 GM view helper。
- [x] 新增 `op_kernel/compute_vec/swiglu_vec.hpp`，实现 `TLOAD -> TLOAD -> TMUL -> TMULS -> row-wise TSTORE`。
- [x] 在 `compute_vec_kernel.cpp` 接入 `RunSwiGluTilePto`，保持 `launchSwiGlu` ABI 不变。
- [x] 删除 active source 中的 `RunSwiGluTile8Scalar` 和 `RunSwiGluTileCubeBridge` 残留。
- [x] 静态检查 `git diff --check`、SwiGLU scalar/bridge grep、Vec PTO path grep。
- [x] 跑 `run.sh --matrix` 黑盒验证。

### Batch 5：GMM2 Cube

- [x] 已细化 Batch 5 设计到 `design.md`：文件边界、直接输出布局、rows/N 分发 PTO 主链、publish 语义、静态/黑盒验收标准。
- [x] 在 `compute_chain.hpp` 新增 `RunGmm2TileRowsN`，按 rows 和 validN 分发到 `RunPtoMatmulStrided<ValidM, 2, ValidN>`。
- [x] 在 `gmm_cube_kernel.cpp` 中让 GMM2 直接 `TSTORE` 到最终 `gmm2Out[rowBegin, nBase]`。
- [x] 删除 GMM2 active path 中的 `ClearComputeOutput`、`StoreGmm2Tile` 和 `tileGmm2Packed` scratch alias。
- [x] 保持 `gmm2TileReady` publish 语义不变。
- [x] 静态检查 `git diff --check`、GMM2 PTO path grep、scratch/scalar helper grep、Vec primitive grep。
- [x] 跑 `run.sh --matrix` 黑盒验证。

### Batch 6：local self-copy PTO 化

- [x] 已细化 Batch 6 设计到 `design.md`：文件边界、local copy PTO 主链、remote TGET/TPUT 保持策略、静态/黑盒验收标准。
- [x] 将 dispatch local self-copy 从 scalar loop 改为 `TLOAD/TSTORE`。
- [x] 将 combine local self-copy 从 scalar loop 改为 `TLOAD/TSTORE`。
- [x] 静态检查 `git diff --check`、local scalar copy grep、TGET/TPUT/TLOAD/TSTORE grep。
- [x] 跑 `run.sh --matrix` 黑盒验证。

### Batch 7：GMM2 -> Combine windowed readiness / overlap

- [x] 已细化 Batch 7 设计到 `design.md`：GMM2 ready 从 stage 级下沉到 window/tile 级，分为 same-stream 语义验证和双 stream overlap 两步。
- [x] Batch 7A：新增/调整 GMM2 window ready 发布，GMM2 每个全局 8 行 ownership window 完成后发布对应 ready slot。
- [x] Batch 7A：保持 same-stream launch，验证 windowed ready 语义不改变 correctness。
- [x] Batch 7A：静态检查 `gmm2TileReadyIndex/WaitGmm2Ready` 与全局 8 行 window 映射。
- [x] Batch 7A：跑 `run.sh --matrix` 黑盒验证。
- [x] Batch 7B：设计并实现 compute/combine 双 stream 编排，combine 依赖 ready signal 等待具体 window。
- [x] Batch 7B：跑 `run.sh --matrix`，重点识别 hang。

### Batch 8：Dispatch -> Compute windowed readiness / overlap

- [x] 已细化 Batch 8 设计到 `design.md`：dispatch ready window、`requiredSafeRows` wait path、dispatch/compute 双 stream 演进路径。
- [x] Batch 8A：设置并验证 `requiredSafeRows` 非零路径，覆盖 dequant wait 逻辑。
- [x] Batch 8A：跑 `run.sh --matrix` 黑盒验证。
- [x] Batch 8B：设计并实现 dispatch/compute 双 stream，compute 通过 ACL event 等待 dispatch stream 完成。
- [x] Batch 8B：跑 `run.sh --matrix`，重点识别 busy-wait/hang。
- [x] Batch 8C：按 range/tile ownership 推进 dequant window pipeline，替换 8B 的全 dispatch event 依赖。
- [x] Batch 8C：dispatch range producer 改为单 block 顺序发布，避免多 block range completion 不可见导致 dequant tile wait hang。
- [x] Batch 8C：跑 `run.sh --matrix` 黑盒验证。

### Batch 9：local PTO copy ping-pong double buffer

- [x] 已细化 Batch 9 设计到 `design.md`：dispatch/combine local copy 使用两个 PTO Vec tile 和 `EVENT_ID0/EVENT_ID1` 交替。
- [x] 将 `DispatchSelfCopyContiguous` 从串行 chunk copy 改为 PTO ping-pong copy。
- [x] 将 `RunCombineCopyContiguous` 从串行 chunk copy 改为 PTO ping-pong copy。
- [x] 保持 remote `TGET/TPUT` 路径不变。
- [x] 静态检查 `EVENT_ID0/EVENT_ID1/TLOAD/TSTORE` 和无 `dstPtr[i] = srcPtr[i]`。
- [x] 跑 `run.sh --matrix` 黑盒验证。

### Batch 10：方案B ready queue protocol scaffold

- [x] 已在 `design.md` 补充方案B：gemm_ar-style dual-kernel ready-queue overlap。
- [x] 新增 `pipeline_queue.hpp`：per-producer queue、entry/count/expectedCount/capacity、enqueue/poll helper。
- [x] 在 route/host 侧建立 queue producerCount、expectedCount、capacity 计划；当前 1-producer queue 按 compute tile count 初始化，后续多 producer 再扩展。
- [x] 在 `kernel_launch.hpp`/`main.cpp` 中补充 queue pipeline launch args 和 queue workspace 分配清零。
- [x] 新增 `comm_vec_queue_kernel.cpp` 与 `compute_cube_queue_kernel.cpp`，并拆成独立 device targets 避免 AscendC `g_tilingKey` 多 source 冲突。
- [x] 静态检查 queue entry/count 32B 对齐、单 producer ownership、Vec/Cube primitive target 分离、active host 只调用两个 queue pipeline launch。
- [x] 编译通过全部 device targets 和 host executable；按验收标准，先完成代码与详细检视，再开始黑盒验证。

### Batch 11：Dispatch/Dequant -> GMM1 queue pipeline

- [x] 将 comm_vec queue kernel 改为有限 event-loop：先生产 dispatch/dequant windows，同时轮询 GMM1/GMM2 ready queue，不能在单一顺序 kernel 中先等完整 GMM1 再进入后续阶段。
- [x] 让 compute_cube queue kernel 轮询 `dispatchToGmm1Queue`，按 window 执行 GMM1 并发布 `gmm1ToSwiGluQueue`。
- [x] GMM1 完成后写入 `gmm1ToSwiGluQueue`，保持 dequantOut `[tile,8,2]` 和 GMM1 PTO matmul shell 不变。
- [x] 静态检查 producer/consumer finite count、无单 kernel 顺序等待中间阶段导致自锁。
- [x] active two-kernel 主链接通后，最小 MPI case 通过，随后 `run.sh --matrix` 通过。

### Batch 12：GMM1 -> SwiGLU -> GMM2 queue pipeline

- [x] 让 comm_vec 从 `gmm1ToSwiGluQueue` 消费 ready window 并执行 PTO SwiGLU。
- [x] 让 comm_vec 写入 `swiGluToGmm2Queue`，compute_cube 从该 queue 消费并执行 GMM2。
- [x] GMM2 仍直接 `TSTORE` 到最终 `gmm2Out`，未恢复 packed scratch。
- [x] queue entry 发布增加 entry cacheline `dcci`，修复 consumer 看到 count 但 entry payload 未完全可见导致 SwiGLU/GMM2 读空的问题。
- [x] 校验每个 queue 的 expectedCount 与实际 enqueue count 一致；最小 case 四段 queue count 均到 1 后数值闭环。
- [x] active two-kernel 主链接通后，最小 MPI case 通过，随后 `run.sh --matrix` 通过。

### Batch 13：GMM2 -> Combine queue and final two-kernel pipeline

- [x] compute_cube 写入 `gmm2ToCombineQueue`，comm_vec 消费 ready window 并执行 combine/restore。
- [x] active host 主链已切到 `launchCommVecQueuePipeline` + `launchComputeCubeQueuePipeline` 两个 kernel。
- [x] 两个 pipeline kernel 都是有限循环：comm_vec 由 tileCount/CombineWorkBlockCount/restore block 分工退出，compute_cube 由 tileCount 退出。
- [x] active host path 不再串行调用 Batch 1-9 的六段 launch 主链。
- [x] 静态检查 active path 不再串行调用 `launchDispatchRange/launchDequant/launchGmm1/launchSwiGlu/launchGmm2/launchCombineOnly`。
- [x] 跑 `run.sh --matrix` 黑盒验证，外层 `timeout 600`，三组 public A3 matrix 均 PASS。

### Batch 14：Dispatch -> Dequant/GMM1 真 window overlap

- [x] 在 `design.md` 固化 Batch 14 设计：`comm_vec` block0 不再先完整执行 `RunDispatchRangeOnlyKernel`，而是 dispatch range/window 完成后立即 dequant 并 publish `dispatchToGmm1Queue`。
- [x] 复用 `dispatch_pull.hpp` 的单 range `RunDispatchRangeTask`，保留 remote `TGET` 和 local PTO `TLOAD/TSTORE`。
- [x] 在 `comm_vec_queue_kernel.cpp` 实现 dispatch/dequant integrated event-loop：dispatchRangeHead、dequantTileHead、gmm1/swiglu/gmm2 queue polling 同 loop 推进。
- [x] 通过代码检视确认 active path 没有 whole-dispatch 前置等待，没有回退 host 六段串行 launch。
- [x] 补 `expertSourceSegments` queue ABI，dequant tile 只在 source segment/range completion 可见后发布 `dispatchToGmm1Queue`。
- [x] 最小 MPI case 防 hang/正确性验证通过；public matrix 留到阶段性结构检视后执行。

### Batch 15：GMM2 -> Combine entry/window overlap

- [x] 在 `design.md` 固化 Batch 15 设计：combine work 不再等待 `gmm2ToCombineQueue.count >= tileCount` 后才执行。
- [x] 在 `comm_vec_queue_kernel.cpp` 将 combine work block 改为按自身 row window 扫描 `gmm2ToCombineQueue` entry。
- [x] 以 combine work block 的唯一 workIndex 保持 task/range ownership，避免多个 work block 处理同一 task。
- [x] 保持 queue combine path 不等待旧 `gmm2TileReady` signal；旧 split-stage 入口继续保留旧 wait 语义。
- [x] 通过代码检视确认 active combine path 不再等待全局 tileCount 后才开始 combine。
- [x] 最小 MPI case 防 hang/正确性验证通过；public matrix 留到阶段性结构检视后执行。

### Batch 16：Multi-producer queue + round-robin consumer

- [x] 在 `pipeline_queue.hpp` 增加 producer cursor/head helpers 和 round-robin poll helper，保持 entry-before-count publish 顺序。
- [x] 在 host queue plan 中按 producerCount 分配 expectedCount，并校验 expectedTotal 等于各 producer expectedCount 之和。
- [x] 将 active queue path 从全量硬编码 `kPipelineQueueProducerId0` 推进到 producerId 参数化。
- [x] consumer 维护每 producer 本地 head，round-robin poll 多 producer queue。
- [x] 通过代码检视确认没有多个 producer 写同一个 queue count/completion slot。
- [x] 本批先以结构检视和编译为主；打开真实多 producer 前必须先完成 Batch 17 的 producer assignment 设计。

Batch 16 落地证据：`pipeline_queue.hpp` 已新增 `PipelineProducerCursor`、`PipelineConsumerHeads`、`PublishNext`、`TryPopRoundRobin`；`comm_vec_queue_kernel.cpp` 和 `compute_cube_queue_kernel.cpp` 的 consumer 侧已改为 per-producer heads + round-robin poll，active op_kernel 中 `kPipelineQueueProducerId0` 仅剩协议常量定义。编译已通过 `dispatch_combine_moe_v2_comm_vec_queue_kernel`、`dispatch_combine_moe_v2_compute_cube_queue_kernel` 和 host executable。真实多 cube producer assignment 仍留给 Batch 17，不在 Batch 16 偷开。

### Batch 17：Cube multi-block GMM producer

- [x] 在 `compute_cube_queue_kernel.cpp` 取消 active path 只有 block0 工作的结构，设计 cubeProducerCount 和 producerId。
- [x] 采用静态 assignment：`producerId = tileIndex % cubeProducerCount`，避免 device 原子 claim。
- [x] 多 cube block 分摊 GMM1/GMM2 tile/window，并通过 per-producer queue 发布 `gmm1ToSwiGluQueue` / `gmm2ToCombineQueue`。
- [x] 保证两个 cube block 不写同一 `gmm1Out/swigluOut/gmm2Out` window。
- [x] 通过代码检视确认 Cube target 只含 `TLOAD/TMOV/TMATMUL/TSTORE` 类 cube PTO 主链，不引入 Vec primitive。
- [x] 结构检视 OK 后，再跑最小黑盒、public matrix 和固定 shape 性能采样。

Batch 17 落地证据：host 将 comm 生产 queue 保持单 producer，将 cube 生产的 `gmm1ToSwiGluQueue/gmm2ToCombineQueue` 按 `cubeProducerCount` 初始化，并把 `computeCubeQueueLaunchArgs.blockDim` 打开到该数量。`compute_cube_queue_kernel.cpp` 以 `blockIdx` 作为 producerId，按 `tileIndex % producerCount` 静态分摊 GMM1/GMM2 window，每个 cube block 只写自身 producer queue。结构 grep 未命中 block0-only cube 模式和 cube target Vec primitive；编译通过 queue 两个 device target 与 host executable；最小 MPI case PASS；`timeout 600 run.sh --matrix` 三组 public A3 matrix 均 PASS。自审风险：Vec 侧 dispatch/dequant/SwiGLU bridge 仍主要由 block0 推进，Batch 17 仅解决 cube GMM producer 并行度，后续 Batch 18/20 需要继续看 group/window 调度和性能瓶颈。

### Batch 18：Expert group / epilogue granularity 推进

- [x] 在 route plan 中建立 group -> dispatch ranges / compute tiles / combine ranges 映射。
- [x] 增强 `ExpertGroupTaskRange` / `ExpertRangeMeta` 或 queue entry side metadata，使 queue entry 能标识 group/localExpert。
- [x] `comm_vec` event-loop 按 group/window 优先推进 dispatch、dequant、SwiGLU、combine。
- [x] `compute_cube` 按 group-aware queue entry 消费 GMM1/GMM2。
- [x] 通过代码检视确认 group 只是调度颗粒度，不改变 packed row layout 和 compute tile layout。
- [x] 结构检视 OK 后，再跑 public matrix 和必要性能采样。

Batch 18 落地证据：`ComputeTileMeta` 增加 `localExpert/expertRangeIndex`，`ExpertGroupTaskRange` 增加 dispatch/combine range 与 compute tile 边界；route plan 生成 `expertGroupRanges`，并在 compute tile 上标注所属 localExpert；`PipelineQueueEntry` 从 `ComputeTileMeta` 携带 `localExpert` 和 `expertRangeIndex`，让 comm/cube window 交接保留 group 侧信息。该批只增加调度 side metadata，不改变 packed row layout、compute tile layout 或 GMM/Vec PTO 主链。`git diff --check` 通过；编译通过 queue 两个 device target 与 host executable；最小 MPI case PASS；`timeout 600 run.sh --matrix` 三组 public A3 matrix 均 PASS。

### Batch 19：Runtime routing / capacity / dropPad 协议对齐

- [x] 梳理参考算子的 tokenPerExpert、cumsumMM、cumsumSend、capacity/dropPad/expertTokensBeforeCapacity 语义，明确哪些进入当前项目范围。
- [x] 优先在 host route plan 补齐非 balanced case 的显式 metadata，不直接搬参考项目非 PTO routing kernel。
- [x] 如必须新增 device routing stage，先设计 PTO 化边界和 Vec/Cube target ownership。
- [x] 为每种新增输入语义设计黑盒可观测 case；不为了白盒便利新增测试 kernel。
- [x] 确认原 public generated balanced matrix 不回退。

Batch 19 落地证据：route plan 已显式生成 `RoutingSemanticMeta`，记录 before-capacity、executable、capacity drop、sentinel/dropPad 计数，并在 host 侧做 metadata consistency guard；`ComputeTileMeta` 的 source segment 列表改为按 tile 连续重建，避免跨 expert/range 插入导致 tile 等待范围不连续；per-iteration reset 现在会清理 dispatch done signal 区、combine done 区和 summary 区，避免 warmup/measure 后续轮次把上一轮 dispatch done 当作本轮 ready。固定失败反馈环已通过：`m=2 topk=1 experts=2 max-output-size=2` 修复第二输出行归零，`m=1 topk=2 experts=2 max-output-size=2` 修复实际值只有 oracle 一半；`timeout 600 run.sh --matrix` 三组 public A3 matrix 均 PASS。Batch 19 没有新增 device routing kernel，继续保持 host route plan + PTO Vec/Cube queue pipeline 边界。

### Batch 20：代码检视与性能验收收尾

- [x] 检视 active host path 仍只有 two-kernel queue pipeline，不回退六段串行 launch。
- [x] 检视 Vec primitive 不进入 cube target，Cube primitive 不进入 vec target。
- [x] 检视 queue publish 保持 entry-before-count 可见性顺序。
- [x] 检视多 producer 不写共享 count/completion slot，consumer 退出来自 expectedCount/expectedTotal。
- [x] 检视 combine/restore owner completion 不重入、不漏发。
- [x] 刷新 `goal.md/design.md/task.md/README.md`，确保文档与 active path 一致。
- [x] 跑 `timeout 600 kernels/manual/a2a3/dispatch_combine_moe_v2/run.sh --matrix`。
- [x] 如 Batch 14-19 引入性能目标，补固定 shape before/after kernel/e2e 采样；性能数据作为功能验收的一部分。

Batch 20 落地证据：active host grep 只命中 `launchCommVecQueuePipeline` 与 `launchComputeCubeQueuePipeline`；旧六段 launch grep 在 `main.cpp` 无命中。Vec primitive grep 在 `op_kernel/compute` 无命中，cube primitive grep 在 `op_kernel/compute_vec` 无命中。`pipeline_queue.hpp` 保持 entry fields 写入、entry cacheline `dcci`、`dsb`、再发布 `queue->count` 的顺序；consumer 使用 per-producer heads + `TryPopRoundRobin`。Combine work block 以唯一 `workIndex` 处理 task/range，restore block 等 owner/incoming completion summary 后执行 restore。`README.md/design.md/task.md/todo.md` 已刷新到 Batch 19/20 口径；`timeout 600 run.sh --matrix` 三组 public A3 matrix 均 PASS，并保留 kernel/e2e profile 输出作为性能采样。

### Batch 21：PTO Sync Abstraction

- [ ] 新增 `op_kernel/protocol/pto_sync_bridge.hpp`，统一 window signal、queue visible order、TNOTIFY/TTEST/TWAIT helper。
- [ ] 将 dispatch done、GMM2 ready、owner completion 的新增同步点通过 helper 包装，避免散落裸 busy-wait 或裸 signal store。
- [ ] 明确 PTO `TSync/TPush` 只用于确认支持的同 kernel AIC/AIV cross-core handoff；跨 kernel 仍走 GM signal/queue。
- [ ] 保持 `pipeline_queue.hpp` entry-before-count 顺序不变。
- [ ] 编译并跑 Batch 19 两个 fixed repro，确认同步抽象不改变语义。

### Batch 22：Group-first Scheduler

- [ ] 固化 `ExpertGroupScheduleEntry` 或增强 `ExpertGroupTaskRange`，让 group -> dispatch ranges / compute tiles / combine ranges 成为一等调度数据。
- [ ] 将 `comm_vec` event-loop 从全局 tile linear cursor 改成 active group round-robin 推进。
- [ ] 将 `compute_cube` 消费逻辑保留 group identity，为 group/tile 二维 producer assignment 做准备。
- [ ] 代码检视确认 group-first 只改变调度顺序，不改变 packed row layout、compute tile layout、GMM/Vec PTO 主链。
- [ ] public matrix 和 fixed repro 不回退，记录 queue wait / group schedule profile。

### Batch 23：Vec-side Multi-producer

- [ ] 设计并实现 Vec block class 分工：dispatch/dequant producers、SwiGLU producers、combine workers、restore owner。
- [ ] 将 `dispatchToGmm1Queue` 和 `swiGluToGmm2Queue` 从永久 single producer 扩展为 per-producer queue。
- [ ] consumer 继续使用 per-producer heads + round-robin；如果需要 multi-consumer，先使用 group/tile static sharding，不引入 device atomic claim。
- [ ] 代码检视确认没有多个 producer 写共享 count/completion slot，且 block0 不再独占 dispatch/dequant/SwiGLU。
- [ ] 跑 fixed repro、public matrix，并记录 fixed shape kernel/e2e、producer 数和 queue wait 点。

### Batch 24：PTO TPush / TSync Pilot

- [ ] 调研 `include/pto/npu/a2a3/TSync.hpp` 与 `TPush.hpp` 的 launch/domain 约束，确认是否支持当前 two-kernel 形态。
- [ ] 选择 GMM1->SwiGLU 或 SwiGLU->GMM2 单边界做 pilot，不全量替换四段 queue。
- [ ] 如果 two-kernel 不能使用 `wait_flag_dev/ffts_cross_core_sync`，设计独立 fused pilot kernel，不破坏 active 主链。
- [ ] 如果 pilot 接入 active path，必须跑 fixed repro + public matrix；如果不接入 active path，只做编译/局部验证并明确 inactive 边界。
- [ ] 代码检视确认没有半接入主链或与 GM queue 双重同步造成死锁。

### Batch 25：Runtime routing / cumsum metadata schedule

- [ ] 新增 `RuntimeRoutingMeta` / cumsum metadata，覆盖 tokenPerExpert、cumsumMM、cumsumSend、expertTokensBeforeCapacity、drop/sentinel 计数。
- [ ] host route plan materialize metadata 并拷贝到 GM，device scheduler 只读 schedule，不重新实现非 PTO routing kernel。
- [ ] profile 输出新增 routing/cumsum 可观测字段。
- [ ] 设计至少一个非 balanced/capacity/dropPad 可观测 CLI case，不为了白盒便利新增测试 kernel。
- [ ] public generated balanced matrix 不回退。

### Batch 26：代码检视与性能对标收尾

- [ ] 检视 active path 不回退非 PTO fused AscendC 风格，Vec/Cube target 分离仍成立。
- [ ] 检视 Sync abstraction、group-first scheduler、Vec-side multi-producer、TPush/TSync pilot、runtime cumsum metadata 的 active/inactive 边界。
- [ ] 刷新 `goal.md/design.md/task.md/README.md`，确保下一阶段文档与 active path 一致。
- [ ] 跑 `timeout 600 kernels/manual/a2a3/dispatch_combine_moe_v2/run.sh --matrix`。
- [ ] 固定 shape 性能对标：kernel/e2e、routed/remote tokens、各 producerCount、queue/TPush wait 点、group schedule。
- [ ] 总结与 `dispatch_gmm_combine` 的 CrossCore/group/FIFO/cumsum 差距缩小到了哪一层。

## 当前风险

- Vec primitive 不能直接塞进 cube target；后续 review 要继续防止 `TCVT/TROWEXPANDMUL/TMUL/TMULS` 回流到 `compute_cube`。
- Dequant/SwiGLU 已完成 Vec PTO 替换，后续主要风险是维护时破坏 padded tile valid shape 和 row-wise compact store 约束。
- GMM2 active path 已移除 packed scratch alias，后续主要风险是重新引入 `gmm2Out` 尾部 scratch 语义。
- queue protocol 必须保持 entry payload 先可见再发布 count；已通过 entry cacheline `dcci + dsb` 修复一次 count/entry 可见性不同步问题。
- Batch 8C 已证明多 producer 共享 completion slot 会出现可见性/死锁风险；后续扩展多 producer 时必须坚持 per-producer queue。
- NPU/MPI 黑盒运行可能卡住，执行时需要设置合理 timeout 或及时识别卡死。

## 验证命令

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:$LD_LIBRARY_PATH
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so
kernels/manual/a2a3/dispatch_combine_moe_v2/run.sh --matrix
```
