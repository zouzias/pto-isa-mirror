# combine_tile Phase 2 Design

## 目标

Phase 2 的目标是把当前 `combine_tile` kernel 调整为更清晰的 PTO 范式实现：

```text
批量数据 / SIMD / tensor payload -> PTO Tile + GlobalTensor + primitive
标量控制 metadata / routing lookup / signal epoch -> raw GM scalar + PTO signal
PTO primitive 调用 -> 在 stage 主流程中自然直接出现，不再被纯转发 helper 包起来
```

这份设计只描述下一阶段代码整理方向，不改变当前项目的计算边界。`combine_tile` 仍然只覆盖 MoE combine：

```text
expertOutput -> all2allv-like return -> ptrD -> probs weighted restore -> outputC
```

## 设计依据

参考项目内 PTO 总结：

- `docs/PTO_Pragrame_style.md`：PTO kernel 应先拆 stage graph，再用 `Tile`、`GlobalTensor`、primitive 和 signal 表达数据流与同步。
- `/home/ntlab/zy/code/zhangyuan/pto-isa-zy/docs/pto-knowledge.md`：PTO style 不等于把所有数组 Tile 化；边界由 consumer 语义决定。
- `.ai-knowledge/writeback/decision-records/2026-05-26-pto-data-plane-vs-control-plane.md`：payload tensor、SIMD control-data、scalar control metadata 要分开处理。
- `.ai-knowledge/writeback/pitfalls/2026-05-26-pto-control-metadata-over-tiling.md`：避免把 count/index/flag/prefix/offset 这类控制 metadata 过度 Tile 化。

关键原则：

1. 涉及批量数据计算的场景必须 PTO 化，包括矩阵乘、行向量 load/store、row add、reduce、mask、scan、broadcast、compare、filter、vector update 等。
2. 控制类处理如果没有 SIMD 语义，不强制 PTO 化。标量 rank、offset、prefix、row index、signal epoch、cursor 可以保持 raw GM / scalar C++。
3. PTO primitive 应直接调用。不要写只包一层 `TLOAD/TSTORE/TPUT/TWAIT/TAXPY` 的 helper，让主流程看不到真实 PTO 数据流。

## 当前 Kernel 分层

当前 `combine_tile_kernel.cpp` 的实际阶段是：

```text
DispatchCombineTileCombine
  -> ReturnExpertRowsToOwners
  -> WaitCombinePhase
  -> RestoreOutputRows
```

Phase 2 保持这三个 stage 名称与顺序：

- `ReturnExpertRowsToOwners`：负责 `expertOutput -> peerWindow.ptrD` 的变长 return。
- `WaitCombinePhase`：负责等待所有 peer 的 combine done signal。
- `RestoreOutputRows`：负责 `ptrD + probs -> outputC` 的加权还原。

但 stage 内部需要更直接地表达 PTO primitive，不再通过纯包装 helper 隐藏数据流。

## 数据分类

### Payload Tensor

这些对象承载真实 tensor 数据或批量向量计算，应 PTO 化：

- `expertOutput`
- `ptrD`
- `outputC`
- `dispatchedA`
- `packedA`
- 后续若引入真实 expert 计算，其输入/输出矩阵或向量 tile

Phase 2 要求：

- GM 边界用 `GlobalTensor` 表达 shape、stride、layout。
- 片上数据用 `Tile<TileType::Vec, ...>` 或后续矩阵场景的 Mat/Left/Right/Acc tile 表达。
- 搬运与计算直接使用 `TLOAD`、`TSTORE`、`TPUT`、`TGET`、`TAXPY`、`TMATMUL*` 等 primitive。

### SIMD Control-Data

这些对象虽然服务控制流程，但如果作为整段数组参与批量计算，也必须 PTO 化：

- 批量 mask
- 批量 route compare / filter / compact
- device 侧 count row reduce
- device 侧 prefix/scan
- 批量 broadcast/update

Phase 2 不新增这些能力；但后续如果把 `expandedRowIdx`、`peerTokenPerExpert`、`cumsumPerExpert`
改成 device 侧批量 compare、filter、scan 或 reduce，就必须升级为 `Vec Tile + PTO primitive`。

### Scalar Control Metadata

这些对象当前只做标量 lookup、调度或同步，不需要 Tile 化：

- `peerTokenPerExpert[src, globalExpert]`
- `dispatchOffset[localExpert]`
- `prevSumBeforeRank[src, localExpert]`
- `cumsumPerExpert[src, globalExpert - 1]`
- `expandedRowIdx[token, slot]`
- `combineDoneSignal[rank]`
- `signalValue`
- `rowChunk`
- `chunkBase`
- `blockId / blockNum`

Phase 2 要求：

- 标量 index/count/offset/prefix lookup 保持 raw GM scalar load。
- signal 使用 `pto::comm::Signal` + `TWAIT` / notify primitive。
- 不为这些标量 metadata 引入 `Tile`、L1/L0/UB staging 或双 buffer。

## Primitive 调用风格

Phase 2 的核心代码风格是：primitive 在 stage 中直接可见。

推荐：

```cpp
GlobalNd<half> src(...);
GlobalNd<half> dst(...);
VecTile<half> tile(1, cols);
TASSIGN(tile, ubAddr);
TLOAD(tile, src);
TSTORE(dst, tile);
```

推荐：

```cpp
GlobalNd<half> remoteDst(...);
GlobalNd<half> localSrc(...);
VecTile<half> ping(1, cols);
VecTile<half> pong(1, cols);
TASSIGN(ping, pingAddr);
TASSIGN(pong, pongAddr);
pto::comm::TPUT(remoteDst, localSrc, ping, pong);
```

不推荐：

```cpp
CopyRowHalf(...);       // 内部只做 TLOAD + TSTORE
TPutRowsHalf(...);      // 内部只做 GlobalTensor 构造 + TPUT
AddWeightedRowHalf(...); // 内部只做 TLOAD + TAXPY + TSTORE
```

允许保留的函数类型：

- layout 计算：`MakeWorkspaceLayout`、`MakePeerWindowLayout`。
- view 构造：`MakeLocalWorkspaceView`、`MakeLocalPeerWindowView`。
- remote pointer 映射：`RemotePtr`、`MakeRemotePeerWindowView`。
- 纯标量边界函数：`ExpertNumPaddedDevice`、`TokenShardBegin`、`TokenShardEnd`。

需要收敛或移除的函数类型：

- 只隐藏一个 PTO primitive 的 wrapper。
- 只隐藏 `GlobalTensor + Tile + primitive` 的 wrapper。
- 让 stage 主流程看不到 `TLOAD/TSTORE/TPUT/TAXPY/TWAIT` 的 wrapper。

## Stage 设计

### Stage 1: ReturnExpertRowsToOwners

职责：

```text
local expertOutput segment -> owner rank peerWindow.ptrD segment
```

控制面：

- 遍历 `src rank` 和 `localExpert`。
- 标量读取 `rows`、`srcStart`、`dstStart`。
- 根据 `rowChunk` 和 `blockId` 分配 chunk。

数据面：

- `src == myRank`：本地 `expertOutput -> local ptrD`，直接在当前分支内构造 `GlobalTensor`，声明 `VecTile`，
  调用 `TLOAD` 和 `TSTORE`。
- `src != myRank`：本地 `expertOutput -> remote ptrD`，直接在当前分支内构造 local/remote `GlobalTensor`，
  声明 ping/pong `VecTile`，调用 `pto::comm::TPUT`。

Phase 2 代码形态：

```text
for each return segment:
  rows/srcStart/dstStart = raw scalar metadata lookup
  for each rowChunk assigned to this block:
    if local owner:
      GlobalTensor src/dst
      VecTile tile
      TLOAD
      TSTORE
    else:
      GlobalTensor src/remoteDst
      VecTile ping/pong
      pto::comm::TPUT
```

注意：

- `peerTokenPerExpert`、`dispatchOffset`、`prevSumBeforeRank`、`cumsumPerExpert` 不做 Tile 化。
- `TPUT` 所需 `GlobalTensor` 应在调用点附近构造，避免裸指针偏移跨多层传播。
- 如果未来 return payload 从 half 扩展到其他 dtype，优先用类型别名组织，不恢复纯 wrapper。

### Stage 2: WaitCombinePhase

职责：

```text
wait local peerWindow.combineDoneSignal[peer] >= signalValue
```

控制面：

- signal 是同步 metadata，不是 payload tensor。
- 保持 `pto::comm::Signal` + `TWAIT`。

Phase 2 代码形态：

```text
for peer assigned to this block:
  pto::comm::Signal signal(...)
  pto::comm::TWAIT(signal, signalValue, WaitCmp::GE)
```

注意：

- 不引入 signal tile staging。
- 不把 signal wait 包成看不到 `TWAIT` 的多层 helper。

### Stage 3: RestoreOutputRows

职责：

```text
outputC[token, :] = sum(slot in topK) probs[token, slot] * ptrD[expandedRowIdx[token, slot], :]
```

控制面：

- token/block 切分是 scalar control。
- `expandedRowIdx[token, slot]` 当前是 scalar lookup。
- `probs[token, slot]` 当前是 scalar weight。

数据面：

- `outputC[token, :]` 清零是批量 vector store，应 PTO 化。
- `ptrD[ptrDRow, :]` load 是批量 payload load，应 PTO 化。
- `outputC += prob * ptrD` 是 SIMD vector update，应 PTO 化，用 `TAXPY` 或等价 primitive。
- `outputC` store 是 payload store，应 PTO 化。

Phase 2 代码形态：

```text
for token assigned to this block:
  GlobalTensor output row
  VecTile output tile
  TEXPANDS zero
  TSTORE output

  for each topK slot:
    ptrDRow = raw scalar expandedRowIdx lookup
    prob = scalar probs lookup
    if ptrDRow valid:
      GlobalTensor ptrD row
      GlobalTensor output row
      VecTile ptrTile/outputTile
      TLOAD ptrTile
      TLOAD outputTile
      TAXPY outputTile, ptrTile, prob
      TSTORE outputTile
```

注意：

- `expandedRowIdx` 不因为是数组就自动 Tile 化。
- 如果后续把 route validity 做批量 mask/filter，再把相关逻辑改成 Vec Tile。
- `TAXPY` 调用必须在 restore stage 主流程中直接可见。

## 需要调整的当前函数

当前代码中建议保留：

- `MakeWorkspaceLayout`
- `MakePeerWindowLayout`
- `MakeLocalWorkspaceView`
- `MakeLocalPeerWindowView`
- `RemotePtr`
- `MakeRemotePeerWindowView`
- `ExpertNumPaddedDevice`
- `TokenShardBegin`
- `TokenShardEnd`
- `SoftSyncAiv`

当前代码中建议内联到 stage，或改成只保留类型别名/局部 lambda：

- `CopyRowHalf`
- `TPutRows`
- `TPutRowsHalf`
- `StoreZeroRowHalf`
- `AddWeightedRowHalf`
- `NotifySignal`
- `WaitSignal`

当前代码中可保留为极薄 scalar 操作，也可直接展开：

- `LoadScalarI32`
- `StoreScalarI32`
- `InvalidateGmCacheLines`

判断规则：

- 如果函数名称隐藏了 PTO primitive，则移除。
- 如果函数名称表达的是 layout、remote view、block 切分、cache 管理，可以保留。
- 如果函数只是为了少写几行 primitive 调用，不保留。

## 文件级改造计划

### `combine_tile_kernel.cpp`

1. 保留当前 ABI 和 kernel 入口。
2. 在文件顶部明确三类对象：
   - payload tensor aliases；
   - scalar control metadata；
   - synchronization signal。
3. 在 `ReturnExpertRowsToOwners` 中直接展开 local copy 和 remote TPUT。
4. 在 `WaitCombinePhase` 中直接构造 `pto::comm::Signal` 并调用 `TWAIT`。
5. 在 `RestoreOutputRows` 中直接展开 zero store 和 weighted `TAXPY`。
6. 删除只包 primitive 的 helper，或把它们降级为局部代码块。

### `README.md`

README 可保留当前项目说明，只需在后续实现完成后补充 Phase 2 编码规则入口。

### `DESIGN.md`

如果后续要把 Phase 2 作为长期设计基线，可将本文件的原则合并进 `DESIGN.md`，但不要在本阶段混改。

## 验收标准

静态验收：

- `combine_tile_kernel.cpp` 中 `TLOAD`、`TSTORE`、`TPUT`、`TWAIT`、`TAXPY` 在 stage 主流程中可见。
- 不再存在只包 primitive 的 `CopyRowHalf`、`TPutRowsHalf`、`AddWeightedRowHalf`、`StoreZeroRowHalf`。
- scalar metadata 没有新增 `Tile` staging。
- signal wait 没有被包装成隐藏 `TWAIT` 的多层 helper。
- forbidden API grep 仍无 AscendC/Catlass/SHMEM 依赖。

行为验收：

```bash
bash run.sh --skip-run 1
```

硬件空闲时再跑：

```bash
timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --combine-return-only 1
timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --verify 1
```

日志仍应包含：

```text
rank=<r> combine_return_done
rank=<r> combine_wait_done peers=<ep>
rank=<r> restore_done
rank=<r> verify=PASS mismatch_count=0
```

## 非目标

- 不新增 dispatch。
- 不新增真实 expert FFN/GMM。
- 不把所有 metadata Tile 化。
- 不引入 HCCL collective `AllToAllV`。
- 不改变 host fixture 的语义。
- 不做公开 ABI 命名重构。
