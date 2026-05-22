# PTO Tile 编程在 A5 dispatch_combine_moe 算子中的作用与优势

## 1. 结论摘要

`dispatch_combine_moe` 不是一个单纯的 GEMM 算子。它同时包含 token reorder、跨 rank token 分发、两段 GMM、Swiglu / epilogue、per-token scale、结果 combine 回写等多段逻辑。这个算子的难点不只在矩阵计算，而在多层数据搬移、计算与通信之间的协同。

PTO 编程在这个算子中的核心价值，是把这些路径统一到 **Tile + GlobalTensor + 通信原语** 的表达模型中：

- 矩阵主路径用 `TLOAD / TMOV / TMATMUL / TSTORE / TSTORE_FP` 表达 GM、L1、L0、Acc、GM 之间的流动。
- 向量路径用 `TLOAD / TSTORE / TCVT / TGATHER / TSORT32` 等原语表达 token reorder、scale、epilogue 和量化前后处理。
- 跨卡路径用 `TGET / TPUT / TNOTIFY / TWAIT` 表达 remote window 上的数据交换和同步。
- `TASSIGN` 把 tile 与 UB/L1/L0/fixpipe 物理 offset 显式绑定，使 tile 生命周期和数据位置更容易审查。

因此，这里的 PTO 化不是简单把 `DataCopy` 换成另一个 API，而是把 dispatch/combine MoE 的数据流写成可分层、可组合、可复用、可检视的 tile pipeline。

## 2. dispatch_combine_moe 的数据流特点

这个算子可以拆成几条同时存在的数据流：

1. **routing / token reorder**  
   根据 expert 分配结果重排 token，生成 expanded row、expert token count、per-token scale 等中间数据。

2. **dispatch gather**  
   每个 rank 从 remote window 中拉取属于本 rank expert 的 token 数据，并整理成本地 GMM 输入。

3. **GMM1 + Swiglu / quant**  
   第一段矩阵计算后进入向量 epilogue，涉及 cast、scale、mul、exp、div、reduce、量化等处理。

4. **GMM2 + combine**  
   第二段矩阵计算后，将结果按照 token 原始归属写回本地或远端 rank。

5. **跨 rank token count / ready 同步**  
   token 数量、prefix sum、ready flag、barrier counter 等控制信息需要在 rank 间协同。

如果用纯 Ascend C + Catlass 风格组织，这些逻辑通常会自然分裂成三套抽象：Catlass 管 GEMM block，Ascend C 管 UB/GM 搬移和 event，通信路径再额外管理 remote window / HCCL / shmem 风格接口。PTO 的优势在于把这些路径都尽量纳入统一的 tile primitive 体系。

## 3. PTO 编程模型在本算子中的落点

| 数据路径 | PTO 接口 / 原语 | 本项目代表位置 | 作用 |
|---|---|---|---|
| GM → UB 向量搬入 | `pto::TLOAD` | `PtoLoadVector`，`op_kernel/utils/pto_vector_ops.hpp` | token、scale、flag buffer 等向量数据搬入 UB |
| UB → GM 向量搬出 | `pto::TSTORE` | `PtoStoreVector`，`op_kernel/utils/pto_vector_ops.hpp` | 中间结果、token count、epilogue 输出写回 GM |
| UB → GM 原子累加 | `pto::TSTORE<AtomicAdd>` | `PtoStoreAtomicAddVector`，`op_kernel/utils/pto_vector_ops.hpp` | expert token count 这类计数更新 |
| UB 内向量 gather / sort | `TGATHER`、`TSORT32` | `op_kernel/token_reorder/routing/moe_pto_sort.h`、`op_kernel/token_reorder/routing/moe_packed_sort_merge.h` | routing 阶段的本地 UB 内排序、归并和字段抽取 |
| GM → L1 矩阵搬入 | `TLOAD` + `GlobalTensor` + `Tile<Mat>` | `PtoLoadNdGmToNzL1`、`PtoLoadNzGmToNzL1`，`op_kernel/utils/pto_mmad_ops.hpp` | GMM 输入矩阵从 GM 装入 L1 |
| L1 → L0A/L0B | `TMOV` | `PtoMoveL1ToL0A`、`PtoMoveL1ToL0B`，`op_kernel/utils/pto_mmad_ops.hpp` | cube 计算前的 L0 staging |
| L0A/L0B → L0C | `TMATMUL`、`TMATMUL_ACC` | `PtoTileMmad`，`op_kernel/utils/pto_mmad_ops.hpp` | GMM tile 计算 |
| L0C / Acc → GM | `TSTORE`、`TSTORE_FP` | `PtoStoreAccToGm`、`StoreAccumulator`，`op_kernel/utils/pto_mmad_ops.hpp` | 累加结果写回，含 fixpipe/量化输出 |
| GM scale → L1/fixpipe | `TLOAD` + `TMOV` | `StagePerChannelScale`，`op_kernel/utils/pto_mmad_ops.hpp` | per-channel scale 进入 fixpipe 路径 |
| 远端 GM → 本地 | `pto::comm::TGET` | `CopyGMToGMPerToken`，`op_kernel/dispatch_combine_moe_kernel.hpp` | dispatch 阶段从 peer window 拉 token |
| 本地 → 远端 GM | `pto::comm::TPUT` | combine epilogue、token all-gather | combine 阶段把结果写回 owner rank |
| 跨 rank 同步 | `TNOTIFY`、`TWAIT` | `PtoRemoteWindow`，`op_kernel/utils/hccl_window.hpp` | remote window 上的 ready / barrier 协同 |
| Tile 绑定 | `TASSIGN` | vector、matmul、comm tile path 均使用 | 把 tile 与 UB/L1/L0/fixpipe offset 绑定 |

这里的关键点是：PTO 的接口不是只覆盖 GEMM，而是覆盖了这个算子里最关键的几类数据流：vector、matrix、remote communication、tile binding。

## 4. 相对纯 Ascend C + Catlass 编程的优势

### 4.1 数据搬移语义更统一

纯 Ascend C 写法通常围绕 `GlobalTensor`、`LocalTensor`、`DataCopy`、event flag 和手写地址偏移展开。Catlass 能提升 GEMM block 的组织效率，但 GEMM 之外的 token reorder、remote window、per-token scale 和 epilogue 仍需要另一套手工数据搬移逻辑。

PTO 写法把不同层级的数据搬移统一成 tile primitive：

- 向量是 `Tile<Vec>` + `TLOAD/TSTORE`。
- 矩阵是 `Tile<Mat/Left/Right/Acc>` + `TLOAD/TMOV/TMATMUL/TSTORE`。
- 远端通信是 `GlobalTensor` + `comm::TGET/TPUT`。

这样开发者审查一段代码时，首先能看到数据从哪个 tile 到哪个 tensor，而不是先从底层地址和 byte offset 反推语义。

### 4.2 Tile 有效区域显式，尾块和动态 shape 更容易表达

MoE dispatch/combine 的数据量受 token 分布影响，很多维度不是固定满块：

- 每个 expert 的 token 数不同。
- 每个 rank 的 token 分布不同。
- combine 回写时每个目标 rank 的行数不同。
- 向量 epilogue 中最后一段 tile 可能不足完整 tile 长度。

PTO 的 `pto::DYNAMIC` shape、动态 valid rows/cols 和 tile 构造参数，让“这个 tile 实际有效多少行、多少列”成为接口的一部分。相比手工 `DataCopy` 参数和 offset 计算，这种写法更适合表达 MoE 的不规则性。

### 4.3 存储层级边界更清晰

这个算子里同时涉及 GM、remote window、UB、L1、L0A、L0B、L0C、fixpipe buffer。纯底层写法很容易让“地址属于哪一层、当前数据在哪一层、下一步该由哪个 pipe 消费”变得隐式。

PTO tile 类型把层级语义前置：

- `TileType::Vec` 对应向量/UB 工作集。
- `TileType::Mat` 对应 L1 / matrix staging。
- `TileLeft`、`TileRight`、`TileAcc` 对应 cube 计算的 L0A/L0B/L0C 角色。
- `TileType::Scaling` 对应 fixpipe scale 路径。

这使得矩阵路径可以按 “GM → L1 → L0A/L0B → L0C → GM” 审查，向量路径可以按 “GM → UB → vector ops → GM/remote GM” 审查。

### 4.4 Compute / Comm decouple 更自然

dispatch_combine_moe 的通信不是独立通信算子，而是与计算阶段交错：

- dispatch 阶段从 peer window 拉 token。
- token count 需要跨 rank all-gather。
- combine 阶段根据 token owner 把结果写回本地或远端。
- ready signal 和 barrier counter 控制跨 rank 可见性。

PTO 的 comm primitive 使这些通信步骤可以继续使用 tile/global tensor 表达，而不是跳出到完全不同的通信编程模型。`TGET/TPUT` 处理 payload，`TNOTIFY/TWAIT` 处理 signal，这使 compute/comm decouple 的边界更清晰。

### 4.5 更利于模式复用和代码检视

PTO 化以后，可以按模式审查代码：

- vector pipeline 是否都走 `PtoLoadVector/PtoStoreVector`。
- matrix pipeline 是否都走 `PtoLoad*GmTo*L1`、`PtoMoveL1ToL0*`、`PtoTileMmad`、`StoreAccumulator`。
- cross-rank payload 是否都走 `TGET/TPUT`。
- 同步和标量控制是否仍有 native 残留。

这比直接审查大量 `DataCopy`、`SetFlag/WaitFlag`、裸地址偏移更容易形成 checklist。

## 5. 关键工程差异

| 维度 | 纯 Ascend C + Catlass 常见组织 | PTO tile 编程在本算子中的组织 |
|---|---|---|
| GEMM 主路径 | Catlass / block GEMM 抽象负责主计算，周边搬移仍需手写 | `TLOAD/TMOV/TMATMUL/TSTORE_FP` 串起 GM/L1/L0/fixpipe/GM |
| Vector epilogue | Ascend C vector API + UB tensor + event 手写组织 | `Tile<Vec>` + PTO vector wrapper 表达 load/cast/mul/store |
| Token reorder | 手工 UB/GM 搬移、排序、scatter/gather | `TSORT32/TGATHER/TLOAD/TSTORE` 表达 UB 内排序和字段抽取 |
| 跨 rank payload | HCCL/shmem/window 地址 + 自定义协议 | `comm::TGET/TPUT` 直接表达 remote tensor 搬移 |
| 跨 rank signal | 自定义 flag/counter 协议 | `comm::TNOTIFY/TWAIT` 表达 ready/wait |
| 数据形状 | 常由 byte offset、stride、copy length 隐式表达 | `GlobalTensor`、`Shape`、`Stride`、valid tile 显式表达 |
| 检视方式 | 从底层 API 和地址偏移反推意图 | 从 tile 类型、primitive、global tensor 直接看数据流 |

PTO 并不是替代所有底层机制。它更像是在 Ascend C 之上增加了一层面向 tile pipeline 的语义层：底层仍要理解 event、pipe、buffer 容量和硬件约束，但开发者写主数据路径时可以更多围绕 tile 和 primitive 组织。

## 6. 本项目当前 PTO 化效果与边界

`dispatch_combine_moe` 的 PTO 化效果可以概括为：

1. **矩阵 bulk 路径已 PTO 化**  
   GMM 输入搬入、L1 到 L0、matmul、acc/fixpipe 输出都通过 PTO wrapper 表达，代表入口是 `op_kernel/utils/pto_mmad_ops.hpp` 和 `op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp`。

2. **向量 bulk 路径已 PTO 化**  
   token reorder、epilogue、scale、quant 前后处理等主要向量搬移和计算都使用 PTO vector wrapper 或 PTO 原语。routing 路径由 `moe_init_routing_sort.h`、`moe_init_routing_gather_dynamic_quant.h`、`moe_init_routing_fullload_dynamic_quant.h`、`moe_init_routing_expert_tokens.h`、`moe_packed_sort_merge.h` 等文件承载。

3. **跨卡 payload 路径已 PTO 化**  
   dispatch 拉取 peer token、combine 写回 owner rank、token count all-gather 等 payload 搬移使用 `TGET/TPUT`。

4. **跨 rank signal 部分 PTO 化**  
   remote token ready 和 barrier wait 使用 `TNOTIFY/TWAIT`，但 host 侧 HCCL/MPI 仍负责 root info、comm resource、remote window 初始化。

按源码口径看，排除 `build/`、`.cache/`、`out/` 后，业务侧没有 `DataCopy/DataCopyPad/LoadData/Fixpipe/Mmad` 这类 native bulk 搬移调用；主链路使用 PTO bulk/comm 搬移，native 调用集中在同步、标量控制和 substrate 边界。因此可以说“**主数据搬移和计算 payload 已经 PTO 化**”，但不宜说“所有控制路径和同步路径都已经完全 PTO 化”。

## 7. 代码例子：tile 编程优势在本算子中的体现

### 7.1 例子一：dispatch gather 用 `GlobalTensor + TGET` 表达远端 token 拉取

dispatch 阶段的关键动作，是把 peer window 中属于当前 rank expert 的 token 拉到本地。代码中先把远端地址包装成 `GlobalTensor`，再把 UB tile 绑定到临时 buffer，最后用 `pto::comm::TGET` 完成远端搬移。

代表代码：`op_kernel/dispatch_combine_moe_kernel.hpp` 中 `CopyGMToGMPerToken`：

```cpp
using PackedGlobal = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
using PackedTile = pto::Tile<pto::TileType::Vec, T, 1, 1024, pto::BLayout::RowMajor, -1, -1>;

PackedGlobal localPackedG(localPackedScratch, packedShape, packedStride);
PackedGlobal remotePackedG(src + inputOffset, packedShape, packedStride);
PackedTile packedTile(1, copyInNum < packedTileCols ? copyInNum : packedTileCols);
pto::TASSIGN(packedTile, ubOffsetBytes);
pto::comm::TGET(localPackedG, remotePackedG, packedTile);
```

这段代码体现的优势是：

- 远端数据不再只是一个裸 GM 地址，而是带 shape/stride 的 `GlobalTensor`。
- 本地接收 buffer 不再只是 UB byte offset，而是绑定到明确的 `Tile<Vec>`。
- 跨卡搬移语义集中在 `TGET`，开发者能直接看出这是 remote → local 的 payload 搬移。

如果用纯 Ascend C + 手写 remote window 方式，通常要从 `GM_ADDR + offset + length + event` 反推“拉的是几行 token、每行多宽、目标 buffer 在哪里”。PTO 写法把这些信息前置到了类型和 tile 构造里。

### 7.2 例子二：GMM 主路径按 GM → L1 → L0 → Acc → GM 分层表达

GMM 路径中，A/B 矩阵从 GM 搬到 L1，再从 L1 搬到 L0A/L0B，之后进入 `TMATMUL`，最终从 Acc / fixpipe 写回 GM。PTO 让这个层级在代码中非常清楚。

代表代码分布在 `op_kernel/utils/pto_mmad_ops.hpp`：

```cpp
pto::TLOAD(dstTile, srcGlobal);     // GM -> L1
pto::TMOV(dstTile, srcTile);        // L1 -> L0A / L0B
pto::TMATMUL(cTile, aTile, bTile);  // L0A/L0B -> L0C
pto::TSTORE(dstGlobal, accTile);    // L0C -> GM
pto::TSTORE_FP(dstGlobal, accTile, scalingTile); // L0C + scale -> GM
```

这段模式体现的优势是：

- `TileType::Mat`、`TileLeft`、`TileRight`、`TileAcc` 分别对应 L1、L0A、L0B、L0C 的角色。
- 搬移和计算接口与硬件层级一一对应，代码审查时可以按 pipeline 阶段看。
- 对 int8 / scale / fixpipe 路径，`ScalingTile + TSTORE_FP` 让量化输出成为 tile pipeline 的一部分，而不是另起一套零散 fixpipe 写回逻辑。

这比“Catlass 负责 GEMM，周边 Ascend C 手动搬移”的组织方式更连贯：GEMM 前后的数据准备、计算、写回都处在同一套 PTO tile 语义里。

### 7.3 例子三：combine 写回用 `TPUT` 统一本地/远端输出逻辑

combine 阶段需要根据 token owner 决定写回本 rank 还是远端 rank。PTO 写法把本地写和远端写都组织成 tile store：本地用 `PtoStoreVector`，远端先写 local scratch，再用 `TPUT` 推到 owner rank。

代表代码：`op_kernel/utils/block_epilogue_pertoken_row.hpp`：

```cpp
void StoreRemoteRow(...) {
    row_detail::PtoStoreVector(localScratch, ubDOffset + colOffsetBytes, chunkCols);
    TputGlobal localRowG(localScratch, rowShape, rowStride);
    TputGlobal remoteRowG(dstRowBase + colOffset, rowShape, rowStride);
    pto::comm::TPUT(remoteRowG, localRowG, tputTile);
}

if (dstRank == params.rank) {
    row_detail::PtoStoreVector(dstRowBase, ubDOffset, blockN);
} else {
    StoreRemoteRow(dstRowBase, localScratch, blockN, ubDOffset);
}
```

这段代码的优势是：

- 本地输出和远端输出都以 vector tile 为最小搬移单元。
- 远端输出不是裸地址协议，而是 `local GlobalTensor -> remote GlobalTensor`。
- 对开发者来说，combine 的核心不再是“写哪个 GM 地址”，而是“这个 tile 的 owner rank 是谁，应该 store 还是 put”。

这对 MoE combine 很重要，因为结果回写天然是不规则的：不同 token 的 owner rank 不同，每个 rank 的行数也不同。PTO 的 shape/stride/tile 写法比手写 offset 更容易维护。

### 7.4 例子四：token reorder 中用 `TSORT32/TGATHER` 表达 UB 内排序和字段抽取

routing 阶段包含排序、归并和 payload 提取逻辑：`moe_init_routing_sort.h` 负责 routing sort 主流程，`moe_packed_sort_merge.h` 负责分段排序结果归并，`moe_pto_sort.h` 提供 packed sort 和字段抽取原语封装。其中 `moe_pto_sort.h` 用 `TSORT32` 做 packed sort，用 `TGATHER` 从 packed sort 结果中抽取 key 或 payload 字段。

代表代码：`op_kernel/token_reorder/routing/moe_pto_sort.h`：

```cpp
pto::TSORT32(packedTile, srcTile, payloadTile);

pto::TGATHER<PtoSortPayloadTile, PtoPackedPayloadTile, pto::MaskPattern::P1010>(
    sortedPayloadTile, packedPayloadTile);

pto::TGATHER<PtoSortKeyTile, PtoPackedSortTile, pto::MaskPattern::P0101>(
    sortedKeyTile, packedTile);
```

这体现的优势是：

- UB 内排序由 `TSORT32` 表达，packed 数据结构的字段抽取由 mask pattern 表达，语义比手写地址步进更明确。
- `sortedPayloadTile`、`packedPayloadTile`、`sortedKeyTile` 这些 tile 名称直接表达了数据角色。
- token reorder 这种 vector-only 路径也纳入 PTO，而不是只让 PTO 覆盖 GEMM。

### 7.5 例子五：量化输出路径把 scale 显式建模为 tile

你选中的 `CopyL0CToGmQuantMode` 属于量化输出模式选择：不同芯片、源类型、目标类型、scale granularity 会映射到不同输出模式。在 PTO 路径中，真正的数据路径体现在 `StagePerChannelScale` 和 `TSTORE_FP`：scale 先从 GM 进入 `ScaleMatTile`，再 `TMOV` 到 `ScalingTile`，最后和 Acc tile 一起参与 `TSTORE_FP`。

代表代码：`op_kernel/utils/pto_mmad_ops.hpp`：

```cpp
pto::TLOAD(scaleMatTile, gmBlockSGlobal);
pto::TMOV(scalingTile, scaleMatTile);
pto::TSTORE_FP(dstGlobal, accTile, scalingTile);
```

这段代码的优势是：

- scale 不再是一个隐式参数，而是明确的 `ScalingTile`。
- per-channel scale 的搬入、格式转换、写回绑定在同一个 tile pipeline 中。
- 量化路径可以和普通 Acc 写回共用 `StoreAccumulator` 这类封装，只在 trait 和 tile 类型上区分模式。

这对于 dispatch_combine_moe 很关键，因为 GMM 输出后紧接 epilogue/quant/combine，scale 生命周期如果不清晰，很容易在纯 Ascend C 写法中变成多个 buffer、多个 event、多个 offset 的组合。


## 8. 小结

PTO tile 编程在 `dispatch_combine_moe` 中的优势，核心不是“API 名字更统一”，而是把复杂 MoE 算子的多层数据流重写成更接近硬件层级和算子语义的 tile pipeline。

对这个算子而言，PTO 的价值主要体现在三点：

1. **统一表达**：矩阵、向量、远端通信都可以用 tile/global tensor/primitive 描述。
2. **显式边界**：GM、UB、L1、L0、fixpipe、remote window 的职责更清楚。
3. **更好检视**：开发者可以按 primitive 和 tile 类型检查主路径是否 PTO 化，快速定位 native 残留。

因此，相比纯 Ascend C + Catlass 编程，PTO 更适合这个算子的原因在于：它不仅覆盖 GEMM 主计算，也覆盖了 dispatch/combine MoE 最难维护的 token 搬移、动态 shape、跨 rank payload 和 compute/comm 协同路径。
