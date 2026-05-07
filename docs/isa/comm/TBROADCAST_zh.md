# TBROADCAST

## 简介

`TBROADCAST` 把当前 NPU 作为根节点的数据广播到并行组中的所有 rank。

只有根节点执行 `TBROADCAST`。非根节点只需要保证目标缓冲区在操作期间已分配且可写；在非根节点上主动调用该指令属于未定义行为。

当数据超过单个 UB Tile 容量时，传输会自动按二维滑动方式分块。

## 数学语义

广播完成后：

$$ \mathrm{dst}^{(k)}_{i,j} = \mathrm{src}^{(\text{root})}_{i,j} \quad \forall k \in [0, N) $$

其中 `N` 为 rank 总数。

## 汇编语法

PTO-AS 形式：

```text
tbroadcast %group, %src : (!pto.group<...>, !pto.memref<...>)
```

lowering 会引入 UB 暂存 Tile，因此 C++ 接口要求显式传入 `stagingTileData`，或在双缓冲模式下传入 `pingTile` / `pongTile`。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &stagingTileData, WaitEvents&... events);

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## 约束

!!! warning "约束"
    ### 类型约束

    - `ParallelGroup::value_type::RawDType` 必须等于 `GlobalSrcData::RawDType`
    - `TileData::DType` 必须等于 `GlobalSrcData::RawDType`

    ### 内存与并行组约束

    - `srcGlobalData` 必须指向根节点本地内存
    - `stagingTileData`、`pingTile`、`pongTile` 必须预先在 UB 中分配
    - `parallelGroup.tensors[k]` 必须指向 rank `k` 的目标缓冲区
    - `parallelGroup.GetRootIdx()` 必须标识当前调用方是广播根节点

    ### 分块约束

    - 静态 `ValidRow` / `ValidCol` 场景下，对应维度必须能整除
    - 若要支持不足整行或整列的边界情况，应使用动态 valid region 的 Tile

## 示例

### 基础广播

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                 BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) tensors[i] = GTensor(group_addrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor srcG(my_data);
    TileT stagingTile(TILE_ROWS, TILE_COLS);

    comm::TBROADCAST(group, srcG, stagingTile);
}
```

### 乒乓双缓冲

```cpp
comm::TBROADCAST(group, srcG, pingTile, pongTile);
```

## 相关页面

- [通信与运行时](communication-runtime_zh.md)
- [TGATHER](./TGATHER_zh.md)
- [TSCATTER](./TSCATTER_zh.md)
