# TBROADCAST

## 简介

广播操作：从当前 NPU 向并行组中的所有 rank 广播数据。调用方 NPU 为根节点，其数据被复制到所有其他 NPU。

只有根节点需要执行 `TBROADCAST`。非根节点只需确保其目标缓冲区已分配且在操作期间可写。在非根节点上调用 `TBROADCAST` 是未定义行为。

**大 Tile 支持**：当 GlobalTensor 在行和/或列方向上超过 UB（Unified Buffer）tile 容量时，传输会通过 2D 滑窗自动分块。

## 数学表示

操作完成后：

$$ \mathrm{dst}^{(k)}_{i,j} = \mathrm{src}^{(\text{root})}_{i,j} \quad \forall k \in [0, N) $$

其中 $N$ 为 rank 数量，`root` 为调用方 NPU。

## 汇编语法

PTO-AS 形式：见 [PTO-AS 规范](../../assembly/PTO-AS.md)。

同步形式：

```text
tbroadcast %group, %src : (!pto.group<...>, !pto.memref<...>)
```
下降后引入 UB 暂存 tile 用于 GM→UB→GM 数据路径；C++ intrinsic 需要显式的 `stagingTileData`（或 `pingTile` / `pongTile`）操作数。

## C++ Intrinsic

声明在 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// 基本广播（单暂存 tile）
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &stagingTileData, WaitEvents&... events);

// Ping-pong 广播（双暂存 tile 双缓冲）
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## 约束

- **类型约束**：
  - `ParallelGroup::value_type::RawDType` 必须等于 `GlobalSrcData::RawDType`。
  - `TileData::DType` 必须等于 `GlobalSrcData::RawDType`。
- **内存约束**：
  - `srcGlobalData` 必须指向本地内存（当前 NPU）。
  - `stagingTileData`（或 `pingTile` / `pongTile`）必须在 UB 中预分配。
- **ParallelGroup 约束**：
  - `parallelGroup.tensors[k]` 必须指向 rank `k` 的目标缓冲区（根节点视角的远程 GM）。
  - `parallelGroup.GetRootIdx()` 标识调用方 NPU 为广播根节点。
  - 所有目标 tensor 假定具有相同的 shape 和 stride。
- **分块模式约束**（数据超过单个 UB tile 时）：
  - 如果 `TileData` 具有静态 `ValidRow`，`GetShape(DIM_3)` 必须能被 `ValidRow` 整除。使用 `DYNAMIC` ValidRow 的 Tile 支持非整除行数。
  - 如果 `TileData` 具有静态 `ValidCol`，`GetShape(DIM_4)` 必须能被 `ValidCol` 整除。使用 `DYNAMIC` ValidCol 的 Tile 支持非整除列数。

## 示例

### 基本广播

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                 BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GTensor(group_addrs[i]);
    }

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor srcG(my_data);
    TileT stagingTile(TILE_ROWS, TILE_COLS);

    // 当前 NPU 将其数据广播到所有其他 NPU
    comm::TBROADCAST(group, srcG, stagingTile);
}
```

### Ping-Pong 广播（双缓冲）

使用两个 UB tile 重叠下一块的 TLOAD 和当前块的 TSTORE。

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast_pingpong(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {

    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GPerRank srcG(my_data);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);

    // Ping-pong：重叠 TLOAD 和 TSTORE 以提高吞吐量
    comm::TBROADCAST(group, srcG, pingTile, pongTile);
}
```
