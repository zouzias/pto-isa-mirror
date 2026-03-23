# TREDUCE

## 简介

归约操作：从多个远程 NPU 收集数据并在本地执行逐元素归约。

只有根节点需要执行 `TREDUCE`。非根节点只需确保其源缓冲区已准备好且在操作期间保持有效。在非根节点上调用 `TREDUCE` 是未定义行为。

**大 Tile 支持**：当 GlobalTensor 在行和/或列方向上超过 UB tile 容量时，归约会通过 2D 滑窗自动分块。

## 数学表示

对有效区域中的每个元素 `(i, j)`：

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \bigoplus_{r=0}^{N-1} \mathrm{src}^{(r)}_{i,j} $$

其中 $N$ 为 rank 数量，$\oplus$ 为归约操作（求和、取最大值、取最小值等）。

## 汇编语法

PTO-AS 形式：见 [PTO-AS 规范](../../assembly/PTO-AS.md)。

同步形式：

```text
treduce %group, %dst {op = #pto.reduce_op<Sum>} : (!pto.group<...>, !pto.memref<...>)
treduce %group, %dst {op = #pto.reduce_op<Max>} : (!pto.group<...>, !pto.memref<...>)
```
下降后引入内部累加器和接收 tile 用于归约流水线；C++ intrinsic 需要显式的 `accTileData`、`recvTileData`（或 `accTileData`、`pingTileData`、`pongTileData`）操作数。

## C++ Intrinsic

声明在 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// 基本归约（累加器 + 接收 tile）
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData, 
                              TileData &accTileData, TileData &recvTileData, ReduceOp op, WaitEvents&... events);

// Ping-pong 归约（累加器 + ping + pong tile 双缓冲）
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                              TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                              ReduceOp op, WaitEvents&... events);
```

## 约束

- **类型约束**：
  - `ParallelGroup::value_type::RawDType` 必须等于 `GlobalDstData::RawDType`。
  - `TileData::DType` 必须等于 `GlobalDstData::RawDType`。
- **内存约束**：
  - `dstGlobalData` 必须指向本地地址（当前 NPU 上）。
  - `accTileData`、`recvTileData`（或 `accTileData`、`pingTileData`、`pongTileData`）必须为预分配的 UB tile。
- **ParallelGroup 约束**：
  - `parallelGroup.tensors[r]` 必须指向 rank `r` 的源缓冲区（根节点视角的远程 GM）。
  - `parallelGroup.GetRootIdx()` 标识调用方 NPU 为归约根节点。
  - 所有源 tensor 假定具有相同的 shape 和 stride。
- **分块模式约束**（数据超过单个 UB tile 时）：
  - 如果 `TileData` 具有静态 `ValidRow`，`GetShape(DIM_3)` 必须能被 `ValidRow` 整除。使用 `DYNAMIC` ValidRow 的 Tile 支持非整除行数。
  - 如果 `TileData` 具有静态 `ValidCol`，`GetShape(DIM_4)` 必须能被 `ValidCol` 整除。使用 `DYNAMIC` ValidCol 的 Tile 支持非整除列数。

## 示例

### 基本 Reduce Sum

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_sum(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, 
                                 BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GTensor(group_addrs[i]);
    }
    
    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;

    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Sum);
}
```

### Max Reduce

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_max(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, 
                                 BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GTensor(group_addrs[i]);
    }
    
    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;

    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Max);
}
```
