# pto.tgather

## 概要

`pto.tgather` 使用一个或两个显式 UB 暂存 tile，把并行组中各 rank 的缓冲区收集到一个目标 GlobalTensor 中。

## 语义

已核实的 public wrapper 接受：

- `parallelGroup`，
- 目标 GlobalTensor，
- 一个暂存 tile 或一对 ping/pong tile，
- 可选的等待事件 token。

wrapper 会先等待所有传入事件，然后再委托给后端 gather 实现。

## 汇编语法

```text
pto.tgather %group, %dst : (!pto.group<...>, !pto.memref<...>)
```

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`。

```cpp
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &stagingTileData, WaitEvents &... events);

template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &pingTile, TileData &pongTile, WaitEvents &... events);
```

## 约束

!!! warning "约束"
    - 并行组缓冲区描述、目标 tensor 与暂存 tile 的元素类型必须兼容。
    - 暂存 tile 必须位于 UB。
    - ping/pong tile 应位于互不重叠的 UB 区域。
    - wrapper 自身并不编码 root-only 校验；使用者需要遵守后端 collective 实现所期望的语义契约。

## 示例

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void example_gather(auto &group, auto &dstG, auto &stagingTile) {
    comm::TGATHER(group, dstG, stagingTile);
}
```
