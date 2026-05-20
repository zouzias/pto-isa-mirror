# dispatch_ffn_combine_v3 AscendC-to-PTO Design V3

## 目标

在 `task.md` / `task_v2.md` 已完成的基础上，继续降低 `dispatch_ffn_combine_v3` device 侧对 AscendC 公共 API 的直接暴露。V3 设计不追求一次性删除 `kernel_operator.h`，而是把直接依赖分为可 PTO 化、需重构、应保留三类，然后按低风险路径逐步推进。

本设计只覆盖：
- `kernels/manual/a2a3/dispatch_ffn_combine_v3/` 内部；
- CANN 8.5.0 / A3 / `ascend910_93` 当前验证环境；
- device 侧 AscendC 直接依赖收口。

本设计不覆盖：
- host runtime 的 ACL/HCCL bootstrap；
- tiling 平台头；
- 性能优化；
- 跨目录 PTO include API 改造。

## 当前事实基线

### 已完成事项

`task.md` 已记录以下收口结果：
- direct 向量算子已清零；
- 普通 GM-facing `DataCopy` 已清零；
- `DataCopyPad` 已收口为 boundary adapter；
- routing / epilogue / kernel local sync 已 helper 化；
- matmul / fixpipe 残留已集中隔离到 substrate 文件。

`task_v2.md` 已补充：
- AscendC 直接依赖仍集中在 tensor/view、pipe queue、sync/barrier、copy/fill/vector、runtime identity、cube/fixpipe substrate 六类；
- `kernel_operator.h` 仍是 device substrate 入口，不能机械删除；
- 多数文件通过上层头传递 AscendC 类型，删除 include 前必须先降低签名和 helper 对 AscendC 类型的外露。

### PTO 已有公开能力

可直接用于本项目的 PTO 公共能力包括：
- `pto::GlobalTensor`；
- `pto::Tile`；
- `pto::TASSIGN`；
- `pto::TLOAD` / `pto::TSTORE`；
- `pto::TCVT`；
- `pto::TADD` / `pto::TMUL` 等向量原语；
- `pto::RecordEvent` / `pto::TSYNC`；
- `pto::comm::Signal` / `TNOTIFY` / `TWAIT` / `TTEST`。

这些能力来自 `include/pto/pto-inst.hpp`、`include/pto/common/pto_instr.hpp`、`include/pto/common/pto_tile.hpp`、`include/pto/comm/pto_comm_inst.hpp` 和 `include/pto/comm/comm_types.hpp`。

## 依赖分类与设计决策

### A. 可直接 PTO 化的依赖

#### 1. `AscendC::GlobalTensor` 边界 view

当前多个文件重复定义：
- `PtoShapeDyn`
- `PtoStrideDyn`
- `PtoGlobalNd`
- `MakeContiguousGlobal(AscendC::GlobalTensor<T>, elemNum)`

设计：
- 新增 v3 本地 helper：`op_kernel/utils/pto_global_view.hpp`；
- 统一放置 `PtoDynamicShape` / `PtoDynamicStride` / `PtoGlobalNd<T>`；
- 提供两个入口：
  - `MakeContiguousGlobalFromPtr(__gm__ T *ptr, uint32_t elemNum)`；
  - `MakeContiguousGlobal(AscendC::GlobalTensor<T> const &tensor, uint32_t elemNum)`。

作用：
- 把 AscendC tensor 到 PTO tensor 的转换固定为 boundary adapter；
- 让后续 helper 可以逐步从 `AscendC::GlobalTensor` 参数转成 `PtoGlobalNd` 参数；
- 避免四个以上文件复制同一段 shape/stride 逻辑。

不改变：
- 原有 tile size；
- `TLOAD/TSTORE` 的调用顺序；
- sync / barrier；
- 任何实际数据布局。

#### 2. `DataCopy` / 部分 `DataCopyPad`

设计：
- 对连续、矩形、无特殊 padding 语义的 GM↔UB 搬运，使用 `pto::GlobalTensor + pto::Tile + TLOAD/TSTORE`；
- 对真实 tail/pad/stride/atomic 边界，保留 boundary adapter，并记录原因。

判断标准：
- 可以替换：源/目的连续、elem 数明确、tile valid region 可表达；
- 不替换：依赖 `DataCopyPadExtParams` 的 pad 值、非连续 stride、atomic writeback、当前验证不足的尾块边界。

### B. 需要结构性重构的依赖

#### 1. `AscendC::LocalTensor`

PTO 对应抽象是 `pto::Tile`，但不是一行替换。

设计：
- 优先在已经由 PTO primitive 消费的 helper 内，把 `LocalTensor` 转成局部 `Tile` + `TASSIGN`；
- 后续再逐步把 helper 签名从 `LocalTensor` 改成 `Tile`；
- 不先碰 routing / unpermute 中依赖 `TQue` 生命周期的 `LocalTensor`。

#### 2. `TPipe` / `TQue` / `TBuf`

PTO 没有直接等价的 queue allocator API。

设计：
- 先只列候选，不改代码；
- 只有当某条路径能完整表达为 `Tile + TASSIGN + RecordEvent/TSYNC` 后，才启动重构；
- 第一批不动 `moe_init_routing_quant_v2` 的 `TQue` 生命周期。

#### 3. `SetFlag` / `WaitFlag` / `PipeBarrier` / `SyncAll`

PTO 有 `RecordEvent` / `TSYNC`，PTO comm 有 `TNOTIFY/TWAIT/TTEST`，但这两类不能等价覆盖所有 AscendC 硬件事件。

设计：
- 相邻操作已 PTO 化时，优先尝试 `RecordEvent` 串联；
- GM signal / remote ready 使用 `pto::comm`；
- MTE/V/Cube 底层 pipe seam、cross-core 协作、cache coherence 相关路径保留 AscendC wrapper；
- 不用 `TWAIT` 替换本地 `WaitFlag`。

### C. 应保留的 substrate 依赖

以下暂不作为 V3 第一阶段替换目标：
- host 侧 `acl/acl.h`、`hccl/hccl_*`；
- tiling `platform_ascendc.h`、`tiling_base.h`；
- kernel ABI：`__aicore__`、`GM_ADDR`；
- device identity：`GetBlockIdx`、`GetBlockNum`、`GetTaskRation`；
- cross-core coordination：`CrossCoreSetFlag`、`CrossCoreWaitFlag`；
- cache coherence / hint：`DataCacheCleanAndInvalid`、`SetL2CacheHint`；
- matmul/fixpipe substrate：`LoadData`、`Fixpipe`、`Gemm::helper::*`。

这些依赖要么 PTO 当前没有公开一对一能力，要么替换需要重新设计整条 pipeline。

## 未替换原因矩阵

V3 的未替换项不是简单因为“难改”，而是按语义边界分成三类：

- **PTO 不负责这一层**：CANN kernel ABI、runtime identity、queue/buffer 生命周期、cache coherence 等仍属于 AscendC/CANN substrate。
- **PTO 有相近 primitive，但不完整等价**：例如 `TLOAD/TSTORE` 能覆盖连续 tile 搬运，但不能自动覆盖 `DataCopyPad` 的 tail/pad/stride/atomic 语义；`RecordEvent/TSYNC` 能串 PTO primitive，但不能直接替换显式 HardEvent 的完整生命周期。
- **可改但不值得本轮扩散**：存在统一可能，但改动收益低或会扩大到 routing/fixpipe 等高风险路径。

| 未替换项 | 主要原因 | 说明 |
| --- | --- | --- |
| `kernel_operator.h` | PTO 不替代 CANN kernel substrate。 | `__aicore__`、`__gm__`、`GM_ADDR`、`GlobalTensor/LocalTensor`、runtime intrinsic 仍来自 CANN；只要 kernel 还编译在 AscendC 环境里，就不能完全删除。 |
| `AscendC::LocalTensor` | PTO 有 `Tile`，但不是生命周期等价物。 | `pto::Tile` 表达已分配 local memory 上的计算/搬运 view；UB 分配、切片、队列生命周期仍由 `LocalTensor/TBuf/TQue` 管。 |
| `TPipe/TQue/TBuf` | PTO 没有 1:1 结构体。 | 这些是 AscendC 的 buffer/queue/pipeline 管理；PTO 更偏 instruction/tile abstraction，不负责完整队列生命周期。 |
| `DataCopyPad` tail/pad 路径 | PTO 有 `TLOAD/TSTORE`，但不覆盖 pad/tail 语义。 | 已对齐、连续场景可以用 PTO；非 32B 对齐、padding、row pitch、pad fill 这些语义仍保留 `DataCopyPad`。 |
| atomic `DataCopyPad` | PTO 普通 store 不等价 atomic copy。 | `SetAtomicAdd<int32_t>() + DataCopyPad` 是 atomic accumulation，不是普通 `TSTORE`。 |
| 普通 `DataCopy` flag/L1 buffer 路径 | 语义不是普通 tensor copy。 | soft-flag 初始化/写回与同步协议绑定，不能只按连续搬运替换。 |
| `SetFlag/WaitFlag/PipeBarrier/SyncAll` | PTO event 模型不能直接等价 HardEvent 生命周期。 | `RecordEvent/TSYNC` 适合 PTO primitive 链内依赖；现有代码中很多 flag 管的是 MTE/V/M/FIX 多 pipe、double-buffer 复用、Finalize 生命周期。 |
| `CrossCoreSetFlag/WaitFlag` | PTO 本地 event 不替代 cross-core sync。 | 这是跨核同步语义，不等价于本地 tile/event。 |
| `DataCacheCleanAndInvalid` / coherence | PTO 没有普通 tile 接口等价替代。 | remote-window 场景需要 cache coherence / DDR 可见性边界。 |
| `GetBlockIdx/GetBlockNum/GetTaskRation` | PTO 不负责 runtime identity。 | 这些是 CANN runtime/kernel intrinsic。 |
| matmul/fixpipe substrate | 部分 PTO 已用，但执行骨架仍依赖 AscendC pipe/fixpipe 机制。 | `TMATMUL/TSTORE_FP` 已是 PTO；但 L0/L1/FP buffer、pipe flag、fixpipe staging 仍是 CANN substrate。 |
| `moe_v2_pto_sort.h` 内局部 adapter | 可改但不值得本轮扩大改动面。 | 这个可以进一步收口到统一 helper，但属于 routing 内部已有 PTO sort helper，本轮避免扩散。 |

因此 V3 的替换原则是：有 PTO 等价物且语义安全的就替换；PTO 只有相近 primitive 但无法覆盖完整语义的保留并记录；PTO 不负责的 substrate 依赖不作为接口替换目标。

## 分阶段方案

### Phase 1：PTO view helper 收口

目标：减少重复的 AscendC→PTO `GlobalTensor` adapter。

修改范围：
- 新增 `op_kernel/utils/pto_global_view.hpp`；
- 修改 `dispatch_ffn_combine_kernel.hpp`；
- 修改 `block_epilogue_pertoken_v2.hpp`；
- 修改 `block_epilogue_pertoken_row.hpp`；
- 修改 `block_epilogue_pertoken_swiglu.hpp`；
- 视情况修改 `block_mmad_preload_async_fixpipe_quant.hpp`。

验收：
- 重复 `PtoShapeDyn/PtoStrideDyn/MakeContiguousGlobal` 定义减少；
- build 通过；
- small case PASS。

### Phase 2：连续搬运 seam PTO 化

目标：把仍可安全表达为 PTO tile load/store 的 copy seam 替成 `TLOAD/TSTORE`。

修改范围：
- 优先看 `dispatch_ffn_combine_kernel.hpp` 里的剩余 `DataCopyPad`；
- 只处理连续、无特殊 pad/atomic 的路径。

验收：
- 被替换路径输出保持 PASS；
- `DataCopyPad` 台账减少或明确标记为不可替换 boundary adapter。

### Phase 3：LocalTensor 参数外露收缩

目标：把已 PTO 化 helper 的内部签名逐步从 AscendC tensor 转向 PTO tile/view。

修改范围：
- 只处理 epilogue / kernel local helper；
- 不碰 routing `TQue` 生命周期。

验收：
- helper 外露的 `AscendC::LocalTensor` 数量减少；
- 无新增性能优化逻辑；
- small case PASS，阶段末 large case PASS。

### Phase 4：sync/event 试点

目标：在相邻 PTO primitive 链上试点 `RecordEvent/TSYNC`，减少显式 `SetFlag/WaitFlag` 调用面。

限制：
- 不碰 cross-core；
- 不碰 remote window cache coherence；
- 不碰未 PTO 化的 AscendC pipeline。

验收：
- 被替换链路功能 PASS；
- 未出现 hang / timeout；
- 如有不确定，立即回退。

### Phase 5：include shrink

目标：完成前序收口后，再清理不必要的 `kernel_operator.h` include 和 `using namespace AscendC`。

限制：
- 只删除编译器证明不再需要的 include；
- 不通过 forward declaration 绕过真实依赖；
- 不把 include 删除作为独立目标强推。

验收：
- standalone build 通过；
- small case PASS；
- `task_V3.md` 更新最新残留台账。

## 验证策略

每个代码阶段至少执行：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:$LD_LIBRARY_PATH
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so
export MPI_RUNNER=mpirun
bash kernels/manual/a2a3/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32
```

关键阶段再执行 large case：

```bash
bash kernels/manual/a2a3/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 4097 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 8194
```

## 风险与回退

| 风险 | 触发点 | 回退策略 |
| --- | --- | --- |
| shape/stride 表达错误 | `GlobalTensor` helper 收口 | 回退到原文件本地 adapter；保留 grep 基线。 |
| tail/pad 语义丢失 | `DataCopyPad` 替换 | 恢复 boundary adapter，记录不可替换原因。 |
| pipeline hang | `SetFlag/WaitFlag` 替换 | 立即回退该 sync 试点；不继续扩散。 |
| include 删除导致隐式依赖断裂 | include shrink | 恢复 include；先改签名再重试。 |
| 性能波动误判 | 非性能阶段 | 只要求 PASS；性能只记录不作为本阶段目标。 |

## 最终完成定义

V3 完成时应满足：
- PTO view helper 已集中；
- 可安全 PTO 化的连续搬运 seam 已处理；
- 不可替换的 `DataCopyPad` 均记录为 boundary adapter；
- `LocalTensor/GlobalTensor/TPipe/TQue/TBuf` 残留均有明确归因；
- `kernel_operator.h` include 不再无理由扩散；
- build、small case、large case 均 PASS。
