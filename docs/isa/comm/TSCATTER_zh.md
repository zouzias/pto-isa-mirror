# TSCATTER

## 简介

分发操作：调用方 NPU（根节点）通过沿 **DIM_3**（行维度）拆分本地源 tensor，将数据分发到并行组中的所有 rank。这是 `TGATHER` 的逆操作。

只有根节点需要执行 `TSCATTER`。非根节点只需确保其目标缓冲区已分配且在操作期间可写。在非根节点上调用 `TSCATTER` 是未定义行为。

**大 Tile 支持**：当每个 rank 的数据在行和/或列方向上超过 UB tile 容量时，传输会通过 2D 滑窗自动分块。

## 数学表示

本地源 tensor 的 shape 为 $(D_0, D_1, D_2, N \times H, W)$，其中 $N$ 为 rank 数量，每个 rank 接收 $H$ 行。操作完成后：

$$\mathrm{dst}^{(r)}_{d_0, d_1, d_2,\; i,\; j} = \mathrm{src}^{\mathrm{local}}_{d_0, d_1, d_2,\; r \cdot H + i,\; j} \quad \forall\, r \in [0, N),\; i \in [0, H),\; j \in [0, W)$$

## 汇编语法

PTO-AS 形式：见 [PTO-AS 规范](../../assembly/PTO-AS.md)。

同步形式：

```text
tscatter %group, %src : (!pto.group<...>, !pto.memref<...>)
```
下降后引入 UB 暂存 tile 用于 GM→UB→GM 数据路径；C++ intrinsic 需要显式的 `stagingTileData`（或 `pingTile` / `pongTile`）操作数。

## C++ Intrinsic

声明在 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// 基本分发（单暂存 tile）
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                              TileData &stagingTileData, WaitEvents&... events);

// Ping-pong 分发（双暂存 tile 双缓冲）
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                              TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## 约束

- **类型约束**：
  - `ParallelGroup::value_type::RawDType` 必须等于 `GlobalSrcData::RawDType`。
  - `TileData::DType` 必须等于 `GlobalSrcData::RawDType`。
- **内存约束**：
  - `srcGlobalData` 必须指向本地内存（当前 NPU），且足够容纳所有 rank 的数据。具体而言，`srcGlobalData.GetShape(DIM_3)` 必须 $\geq N \times H$，其中 $H$ 为每个 rank 的 `GetShape(DIM_3)`。
  - 如果 `srcGlobalData.GetShape(DIM_3) > N × H`，则仅读取前 `N × H` 行；剩余行被忽略。
  - `stagingTileData`（或 `pingTile` / `pongTile`）必须在 UB 中预分配。
- **ParallelGroup 约束**：
  - `parallelGroup.tensors[r]` 必须指向 rank `r` 的目标缓冲区（根节点视角的远程 GM）。
  - `parallelGroup.GetRootIdx()` 标识调用方 NPU 为分发根节点。
  - 所有目标 tensor 假定具有相同的 shape 和 stride；如果不同则行为未定义。
- **分块模式约束**（每个 rank 的数据超过单个 UB tile 时）：
  - 如果 `TileData` 具有静态 `ValidRow`，每个 rank 的目标 `GetShape(DIM_3)` 必须能被 `ValidRow` 整除。使用 `DYNAMIC` ValidRow 的 Tile 支持非整除行数。
  - 如果 `TileData` 具有静态 `ValidCol`，`GetShape(DIM_4)` 必须能被 `ValidCol` 整除。使用 `DYNAMIC` ValidCol 的 Tile 支持非整除列数。

## 示例

### 基本分发（单暂存 Tile）

根节点有 `NRANKS * ROWS` 行宽度为 `COLS` 的数据。每个 rank 接收 `ROWS × COLS`，沿 DIM_3 拆分。tile 大小（`TILE_ROWS × TILE_COLS`）可以小于每个 rank 的数据 — 此时实现会通过 2D 滑窗自动沿 DIM_3 和 DIM_4 分块传输。

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void scatter(__gm__ T* local_data, __gm__ T* group_addrs[NRANKS], int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GSource = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GSource srcG(local_data);
    TileT stagingTile(TILE_ROWS, TILE_COLS);

    comm::TSCATTER(group, srcG, stagingTile);
}
```

### Ping-Pong 分发（双缓冲）

使用两个 UB tile 重叠下一块的 TLOAD（MTE2）和当前块的 TSTORE（MTE3）。

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void scatter_pingpong(__gm__ T* local_data, __gm__ T* group_addrs[NRANKS], int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GSource = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GSource srcG(local_data);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);

    // Ping-pong：重叠 TLOAD 和 TSTORE 以提高吞吐量
    comm::TSCATTER(group, srcG, pingTile, pongTile);
}
```
