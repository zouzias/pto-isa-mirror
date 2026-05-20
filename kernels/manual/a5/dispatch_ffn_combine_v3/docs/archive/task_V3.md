# dispatch_ffn_combine_v3 Task V3

## 目标

按 [design_v3.md](design_v3.md) 推进 device 侧 AscendC 直接依赖收口。V3 的任务设计以“低风险、可验证、可回退”为原则：先收口重复 PTO view adapter，再处理可 PTO 化 copy seam，最后才考虑 helper 签名和 include shrink。

## 执行边界

- 只修改 `kernels/manual/a2a3/dispatch_ffn_combine_v3/` 内部。
- 默认验证环境：CANN 8.5.0 / A3 / `ascend910_93`。
- 不做性能优化。
- 每次代码任务完成后至少跑 small case。
- 卡不支持多任务并行，验证命令必须串行执行。

## 状态总览

| 任务 | 状态 | 产物 / 验收 |
| --- | --- | --- |
| Task 0 V3 设计和任务文档 | 已完成 | [design_v3.md](design_v3.md), [task_V3.md](task_V3.md) |
| Task 1 PTO GlobalTensor view helper 收口 | 已完成 | 新增 `op_kernel/utils/pto_global_view.hpp`，重复 dynamic shape/stride adapter 已收敛；small case PASS。 |
| Task 2 替换已有 PTO load/store helper 的重复 adapter | 已完成 | `dispatch_ffn_combine_kernel.hpp` 和 epilogue helper 已改用统一 helper；small case PASS。 |
| Task 3 处理 `block_mmad_preload_async_fixpipe_quant.hpp` 的 PTO view alias | 已完成 | 已收口到统一 `PtoGlobalNd` / `MakeGlobalFromPtr`，未改 matmul/fixpipe 语义；small case PASS。 |
| Task 4 剩余 `DataCopyPad` 分类复查 | 已完成 | 已列出可替换/不可替换点，本轮不改源码。 |
| Task 5 `dispatch_ffn_combine_kernel.hpp` copy seam 试点 | 已跳过 | 未发现连续、无特殊 pad/stride/atomic 语义的安全 seam。 |
| Task 6 LocalTensor→Tile 签名收缩候选评估 | 已完成 | 已列候选，本轮不碰 routing `TQue` 生命周期且不改源码。 |
| Task 7 RecordEvent/TSYNC 试点候选评估 | 已完成 | 已列相邻 PTO primitive 链路，未发现安全源码试点；不动 cross-core / cache coherence。 |
| Task 8 include shrink | 已完成 | 仅删除 `select_helper.hpp` 的冗余 `using namespace AscendC`；small/large 回归 PASS。 |
| Task 9 small/large 回归与残留台账更新 | 已完成 | small + large PASS，已更新 `task_V3.md`。 |

## Task 1 PTO GlobalTensor view helper 收口

### 目标

新增一个 v3 本地 helper，集中表达 PTO dynamic GM view，避免多个文件重复定义同一组 `PtoShapeDyn / PtoStrideDyn / PtoGlobalNd / MakeContiguousGlobal`。

### 修改文件

- Create: `op_kernel/utils/pto_global_view.hpp`
- Modify: `op_kernel/dispatch_ffn_combine_kernel.hpp`
- Modify: `op_kernel/utils/block_epilogue_pertoken_v2.hpp`
- Modify: `op_kernel/utils/block_epilogue_pertoken_row.hpp`
- Modify: `op_kernel/utils/block_epilogue_pertoken_swiglu.hpp`

### 具体步骤

- [x] 新建 `op_kernel/utils/pto_global_view.hpp`
  - 定义 `pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoShapeDyn`
  - 定义 `pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoStrideDyn`
  - 定义 `pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoGlobalNd<Element>`
  - 定义 `MakeContiguousGlobalFromPtr(__gm__ Element *ptr, uint32_t elemNum)`
  - 定义 `MakeContiguousGlobal(AscendC::GlobalTensor<Element> const &tensor, uint32_t elemNum)`

- [x] 在 `dispatch_ffn_combine_kernel.hpp` 中删除本地重复定义
  - 删除本地 `PtoShapeDyn`
  - 删除本地 `PtoStrideDyn`
  - 删除本地 `PtoGlobalNd`
  - 删除本地 `MakeContiguousGlobal`
  - include `utils/pto_global_view.hpp`
  - 将调用改为 `pto_ext::detail::MakeContiguousGlobal`

- [x] 在 `block_epilogue_pertoken_v2.hpp` 中删除本地重复定义
  - include `pto_global_view.hpp`
  - 将调用改为统一 helper

- [x] 在 `block_epilogue_pertoken_row.hpp` 中删除本地重复定义
  - include `pto_global_view.hpp`
  - 将调用改为统一 helper

- [x] 在 `block_epilogue_pertoken_swiglu.hpp` 中删除本地重复定义
  - include `pto_global_view.hpp`
  - 保留文件内 `PtoVecTile` alias；它是局部 tile alias，不属于 GM view helper
  - 将调用改为统一 helper

### 验收

- [x] `rg -n 'using PtoShapeDyn|using PtoStrideDyn|MakeContiguousGlobal' op_kernel` 只剩统一 helper 和确需保留的特殊 adapter。
- [x] build PASS。
- [x] small case PASS：`m=16,k=128,n=128,topk=2,experts=2,max-output-size=32`，rank0/rank1 PASS。

## Task 2 替换已有 PTO load/store helper 的重复 adapter

### 目标

在 Task 1 的基础上，让已有 `PtoLoadVector` / `PtoStoreVector` helper 只通过统一 GM view helper 获取 PTO `GlobalTensor`。

### 修改文件

- Modify: `op_kernel/dispatch_ffn_combine_kernel.hpp`
- Modify: `op_kernel/utils/block_epilogue_pertoken_v2.hpp`
- Modify: `op_kernel/utils/block_epilogue_pertoken_row.hpp`
- Modify: `op_kernel/utils/block_epilogue_pertoken_swiglu.hpp`

### 具体步骤

- [x] 检查每个 `PtoLoadVector` 调用中的 `srcChunk` 是否仍为 contiguous slice。
- [x] 检查每个 `PtoStoreVector` 调用中的 `dstChunk` 是否仍为 contiguous slice。
- [x] 确认 `elemNum` 对应 PTO valid cols，不改变 tile capacity。
- [x] 不改变 `TileElems` 默认值。
- [x] 不改变 `TASSIGN/TLOAD/TSTORE` 顺序。

### 验收

- [x] `rg -n 'MakeContiguousGlobal\(' op_kernel` 的调用全部落到统一 helper。
- [x] small case PASS：`m=16,k=128,n=128,topk=2,experts=2,max-output-size=32`，rank0/rank1 PASS。

## Task 3 `block_mmad_preload_async_fixpipe_quant.hpp` PTO view alias 收口

### 目标

该文件当前有自己的 `PtoShapeDyn/PtoStrideDyn`，用于 `GlobalDataOut`。本任务只收口 alias，不改 `TMATMUL/Fixpipe/DataCopy/CrossCore` 等语义。

### 修改文件

- Modify: `op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp`

### 具体步骤

- [x] include `pto_global_view.hpp`。
- [x] 将本地 `PtoShapeDyn/PtoStrideDyn` 替换为统一 alias：实际收口为统一 `PtoGlobalNd` 与 `MakeGlobalFromPtr`，不再在本文件直接构造 dynamic shape/stride。
- [x] 保留 `GlobalDataOut` 这个局部语义 alias。
- [x] 不改 `PtoTileMmad`。
- [x] 不改 `DataCopy` / `CrossCoreSetFlag` / `CrossCoreWaitFlag`。

### 验收

- [x] build PASS。
- [x] small case PASS：`m=16,k=128,n=128,topk=2,experts=2,max-output-size=32`，rank0/rank1 PASS。

## Task 4 剩余 `DataCopyPad` 分类复查

### 目标

重新检查 `DataCopyPad` 残留，区分：
- 可 PTO 化连续搬运；
- 真实 boundary adapter；
- atomic/tail/pad 特殊路径。

### 目标文件

- `op_kernel/dispatch_ffn_combine_kernel.hpp`
- `op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h`
- `op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h`
- `op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h`
- `op_kernel/unpermute/moe_token_unpermute.h`

### 具体步骤

- [x] 对每个 `DataCopyPad` 记录 source/destination、shape、tail 语义。
- [x] 标记是否有 pad 值依赖。
- [x] 标记是否有非连续 stride。
- [x] 标记是否有 atomic writeback。
- [x] 只选择一个最低风险点进入 Task 5：未发现安全 seam，Task 5 跳过。

### 验收

- [x] 在本文件追加 `DataCopyPad` 分类表。
- [x] 不改源码。

### DataCopyPad 分类表

| 位置 | source → destination | shape / tail 语义 | 分类 | Task 5 结论 |
| --- | --- | --- | --- | --- |
| `dispatch_ffn_combine_kernel.hpp:401` `StorePerTokenRows` | UB packed per-token row → GM output row | 每行只写 `hiddenSize`，UB 源行包含额外 `UB_ALIGN` / scale 区域，依赖行间跳步剥离 packed scratch。 | boundary adapter / strided row unpack | 不替换。 |
| `dispatch_ffn_combine_kernel.hpp:410` `StorePerTokenScales` | UB packed scale slice → GM scale | 每 token 只写 1 个 float，源 stride 由 `hiddenSize / 32` 表达 packed row 内 scale 位置。 | boundary adapter / scale extraction | 不替换。 |
| `dispatch_ffn_combine_kernel.hpp:422` `LoadExpertCountsPadded` | GM expert-count table → UB padded rows | 按 `expertPerRank` 拷贝并按 `paddedExpertNumAligned` 布局到 UB，后续逐行 vector cumsum 依赖 padded row。 | padded-row adapter | 不替换。 |
| `dispatch_ffn_combine_kernel.hpp:430` `StoreExpertCountsPadded` | UB padded cumsum rows → GM dense result | 与上一条成对出现，写回 dense expert-count 行。 | padded-row adapter | 不单独替换。 |
| `moe_v2_src_to_dst_and_gather.h:151` `LoadInputTile` | GM input row → UB input tile | aligned 分支已用 `PtoLoadVector`；残留仅处理非 32B 对齐 tail。 | tail adapter | 不替换。 |
| `moe_v2_src_to_dst_and_gather.h:164` `StoreExpandedXTile` | UB quant tile → GM expanded row | aligned 分支已用 `PtoStoreVector`；残留仅处理非 32B 对齐 tail。 | tail adapter | 不替换。 |
| `moe_v2_expert_token_out.h:156` `AtomicStoreCountSlice` | UB count slice → GM count output | `SetAtomicAdd<int32_t>()` 包围，语义是 atomic accumulation。 | atomic adapter | 不替换。 |
| `moe_v2_fullload_quant.h:97` `LoadXRows` | GM x rows → padded UB rows | `dstStride` 由 `inFactor - AlignBytes(cols)` 形成，后续 full-load 计算依赖 padded UB row pitch。 | padded-row adapter | 不替换。 |
| `moe_v2_fullload_quant.h:105` `StoreExpandedXRow` | UB quant row → GM expanded row | 单行输出，但 `cols` 不保证 32B 对齐；保留 DataCopyPad 处理 tail。 | tail adapter | 不替换。 |
| `unpermute/moe_token_unpermute.h:300` `LoadTokenSlice` | GM token slice → UB token slice | aligned 分支已用 `PtoLoadVector`；残留仅处理非 32B 对齐 tail。 | tail adapter | 不替换。 |
| `unpermute/moe_token_unpermute.h:313` `StoreTokenSlice` | UB token slice → GM output token slice | aligned 分支已用 `PtoStoreVector`；残留仅处理非 32B 对齐 tail。 | tail adapter | 不替换。 |

结论：本轮没有满足“连续、无 pad / stride / atomic 语义”的 `dispatch_ffn_combine_kernel.hpp` copy seam。Task 5 按设计边界跳过，不做源码替换。

## Task 5 `dispatch_ffn_combine_kernel.hpp` copy seam 试点

### 目标

在 `dispatch_ffn_combine_kernel.hpp` 内选择一个连续、无特殊 pad 语义的 copy seam，用 PTO `TLOAD/TSTORE` 替换。

### 修改文件

- Modify: `op_kernel/dispatch_ffn_combine_kernel.hpp`

### 具体步骤

- [x] 选择 Task 4 标记为最低风险的 seam：Task 4 未找到满足条件的 seam。
- [x] 用 `pto::GlobalTensor + pto::Tile` 表达 source/destination：无安全目标，本步不执行源码替换。
- [x] 使用 `TLOAD` 或 `TSTORE` 替换 direct `DataCopyPad`：按 Task 4 结论跳过。
- [x] 保持原有同步顺序：未改源码。
- [x] 保持原有 tail 行为：未改源码。

### 验收

- [x] build PASS：Task 5 未改源码，最终由 Task 9 统一回归覆盖。
- [x] small case PASS：Task 5 未改源码，最终由 Task 9 统一回归覆盖。
- [x] 如果结果异常，立即回退 Task 5 改动，保留 Task 4 分类表：无源码改动，无需回退。

## Task 6 LocalTensor→Tile 签名收缩候选评估

### 目标

减少 helper 签名中直接暴露 `AscendC::LocalTensor`，但不影响 `TQue` 生命周期。

### 候选范围

- `dispatch_ffn_combine_kernel.hpp` 中本地 PTO vector helper；
- `block_epilogue_pertoken_v2.hpp`；
- `block_epilogue_pertoken_row.hpp`；
- `block_epilogue_pertoken_swiglu.hpp`。

### 具体步骤

- [x] 列出每个 helper 的 `AscendC::LocalTensor` 参数。
- [x] 判断调用方是否已经能提供 `pto::Tile` 或可安全构造 tile view。
- [x] 标记第一批可迁移 helper：本轮无源码迁移目标。
- [x] 不修改 routing / unpermute 的 `TQue` 相关函数。

### 验收

- [x] 在本文件追加候选表。
- [x] 不改源码。

### LocalTensor→Tile 候选表

| 范围 | 当前 helper / 参数 | 是否可第一批迁移 | 结论 |
| --- | --- | --- | --- |
| `dispatch_ffn_combine_kernel.hpp` `kernel_detail::PtoLoadVector/PtoStoreVector/PtoAddVector/PtoAddScalarVector` | 参数仍是 `AscendC::LocalTensor` / `AscendC::GlobalTensor`，helper 内部临时构造 `pto::Tile`。 | 否 | 调用方大量来自 UB buffer、GM wrapper 和 padded adapter 周边；直接改签名会把 `Tile` 生命周期外溢，收益小。保留现状。 |
| `block_epilogue_pertoken_v2.hpp` `PtoLoadVector/PtoStoreVector/PtoCastVector/PtoMulVector/PtoLoadMatrixRows/PtoStoreMatrixRows` | matrix helper 逐行切 `LocalTensor` 后调用 vector helper。 | 否 | 行切片依赖 `LocalTensor` 下标/stride，若改为 Tile 签名需同步重构矩阵 helper。保留现状。 |
| `block_epilogue_pertoken_row.hpp` `PtoLoadVector/PtoStoreVector/PtoCastVector/PtoMulVector` | 单行 epilogue UB stage helper。 | 可作为后续实验 | 结构最简单，但与 v2/swiglu helper 重复，单独改会制造接口分叉。本轮不改。 |
| `block_epilogue_pertoken_swiglu.hpp` 多个 vector math helper | `PtoVecTile` alias 已局部化，但 helper 入参仍是 `LocalTensor`。 | 否 | 链路包含 reduce/max/exp/div 等多步 UB 中间张量，签名收缩需要统一 tile builder 和临时 tile 生命周期设计。本轮不改。 |

结论：本轮不做 `LocalTensor` 参数改签名。当前更合理的边界是 helper 内部继续用 `LocalTensor.GetPhyAddr()` 创建 PTO tile，避免把 AscendC UB 分配/切片生命周期暴露给上层 PTO tile 接口。

## Task 7 RecordEvent/TSYNC 试点候选评估

### 目标

找到可用 PTO event 模型替代显式 `SetFlag/WaitFlag` 的局部链路。

### 具体步骤

- [x] 列出所有 helper body 内部的 `PtoSetFlag/PtoWaitFlag/PtoPipeBarrier`。
- [x] 判断前后操作是否均为 PTO primitive。
- [x] 排除 cross-core、cache coherence、remote-window 路径。
- [x] 只保留可局部验证的候选：本轮无源码试点目标。

### 验收

- [x] 在本文件追加候选表。
- [x] 不改源码。

### RecordEvent/TSYNC 候选表

| 范围 | 当前同步 | 前后操作 | 是否可试点 | 结论 |
| --- | --- | --- | --- | --- |
| `dispatch_ffn_combine_kernel.hpp` `kernel_detail::PtoLoadVector/PtoStoreVector` 调用点周边 | `PtoWaitFlag/PtoSetFlag` 管 MTE2/MTE3 ping-pong | 前后虽有 PTO `TLOAD/TSTORE`，但 event ID 被外层 double-buffer 复用。 | 否 | `RecordEvent` 不能直接表达现有 event-id 生命周期，保留显式 HardEvent。 |
| epilogue v2/row/swiglu load-cast-mul-store 链 | `V_MTE2`, `MTE2_V`, `MTE3_V`, `V_MTE3` | PTO vector primitive 与 UB stage 复用交错。 | 否 | 多 stage UB 复用依赖显式 flag 初始化/Finalize，不能局部替换成 `TSYNC`。 |
| swiglu 内部纯 vector math 链 | `PipeBarrier<PIPE_V>` | 前后多为 PTO arithmetic tile op。 | 候选但不改 | 可后续设计一个“pure V pipe barrier folding”实验；本轮没有现成 event object 可承载。 |
| `block_mmad_preload_async_fixpipe_quant.hpp` | MTE1/MTE2/M/FIX 多 pipe flag、CrossCore flag、soft-flag DataCopy | matmul/fixpipe/cross-core/cache coherence 混合。 | 否 | 属于 matmul/fixpipe substrate，不进入本轮替换。 |
| remote-window `TGET/TPUT` 周边 | `PipeBarrier<PIPE_ALL>` / local MTE flags | PTO comm + local UB/GM scratch 搬运。 | 否 | `PIPE_ALL` 是远端传输后本地可见性边界，不用 `RecordEvent/TSYNC` 替代。 |

结论：本轮没有安全的 `RecordEvent/TSYNC` 源码试点。保留显式 AscendC HardEvent / PipeBarrier 作为本地 pipe 生命周期和 cross-core/cache coherence 边界。

## Task 8 include shrink

### 目标

在前序代码收口后，删除不再需要的 `kernel_operator.h` include 和 `using namespace AscendC`。

### 具体步骤

- [x] 对每个直接 include `kernel_operator.h` 的文件运行局部 grep。
- [x] 如果仍有 AscendC 类型/API，保留 include。
- [x] 如果仅因统一 helper 间接使用 AscendC，尝试删除 include：未发现此类安全目标；仅删除 `select_helper.hpp` 的冗余 namespace 暴露。
- [x] build 验证。
- [x] 失败则恢复 include：未失败。

### 验收

- [x] build PASS。
- [x] small case PASS：`m=16,k=128,n=128,topk=2,experts=2,max-output-size=32`，rank0/rank1 PASS。
- [x] include shrink 结果写入本文件。

### include shrink 结果

| 文件 | 处理 | 原因 |
| --- | --- | --- |
| `op_kernel/utils/select_helper.hpp` | 删除 `using namespace AscendC;` | 该头本身不使用 AscendC 名称，且唯一包含方 `dispatch_ffn_combine.h` 已有自己的 AscendC namespace 引入。 |
| `op_kernel/utils/pto_global_view.hpp` | 保留 `kernel_operator.h` | helper 签名和实现直接使用 `__gm__`、`AscendC::GlobalTensor`、`GetPhyAddr()`。 |
| `op_kernel/utils/layout3d.hpp` | 保留 `kernel_operator.h` | 当前使用 `__aicore__` 生成 device inline 宏。 |
| `op_kernel/utils/get_tensor_addr.hpp` | 保留 `kernel_operator.h` | 直接使用 `__gm__`、`GM_ADDR`、`__aicore__`。 |
| `op_kernel/utils/hccl_window.hpp` | 保留 `kernel_operator.h` | 直接使用 `GM_ADDR`、`GlobalTensor`、`DataCacheCleanAndInvalid`、`PipeBarrier/SyncAll` 与 runtime identity。 |
| `dispatch_ffn_combine_kernel.hpp` / `dispatch_ffn_combine.h` / routing / unpermute 头 | 保留 | 仍大量直接使用 AscendC tensor、queue、flag、kernel intrinsic 或 CANN tiling 类型。 |

## Task 9 回归验证与残留台账更新

### 验证命令

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

large case：

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

### 验收

- [x] build PASS。
- [x] small case PASS：`m=16,k=128,n=128,topk=2,experts=2,max-output-size=32`，rank0/rank1 PASS。
- [x] large case PASS：`m=4097,k=128,n=128,topk=2,experts=2,max-output-size=8194`，rank0/rank1 PASS。
- [x] 更新 `设计结果` 小节。

## 设计结果

当前状态：Task V3 全部完成。已新增 v3 本地 PTO GM view helper，将 `dispatch_ffn_combine_kernel.hpp`、epilogue helper 和 `block_mmad_preload_async_fixpipe_quant.hpp` 的重复 PTO GM view adapter 收口到统一 helper；`DataCopyPad`、`LocalTensor→Tile`、`RecordEvent/TSYNC` 和 include shrink 均已按风险边界完成分类或最小改动；small/large case 均已通过。

### 已确认边界

- `kernel_operator.h` 是 AscendC / CANN kernel 侧核心头，当前仍被 `__aicore__`、`__gm__`、`GM_ADDR`、`GlobalTensor/LocalTensor`、cache/coherence 和 pipe primitive 直接依赖，不能直接删除。
- `GlobalTensor` 的 PTO GM view adapter 已通过 `op_kernel/utils/pto_global_view.hpp` 收口，连续向量 helper 和 int8 fixpipe store path 复用统一 helper。
- `LocalTensor` 可通过 PTO `Tile` 模型逐步收缩，但本轮评估后不改签名，避免把 AscendC UB 分配/切片生命周期外溢到上层接口。
- `TPipe/TQue/TBuf` 没有 PTO 1:1 替代，需要结构性重构；routing/unpermute 队列生命周期本轮不动。
- `SetFlag/WaitFlag/PipeBarrier/SyncAll` 本轮未找到安全 `RecordEvent/TSYNC` 源码试点，仍作为本地 pipe 生命周期、cross-core 和 cache coherence 边界保留。
- `DataCopyPad` 残留均为 tail/padded-row/strided extraction/atomic adapter，本轮不强行替换。

## 当前进度

- 进度：9/9
- 当前任务：已完成 Task V3 全部 AscendC-to-PTO 收口任务。
