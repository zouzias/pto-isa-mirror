# pto.treduce

## 概要

`pto.treduce` 使用显式 accumulator / receive 暂存 tile，把并行组中各 rank 的缓冲区归约到一个目标 GlobalTensor 中。

## 语义

已核实的 public wrapper 接受：

- `parallelGroup`，
- 目标 GlobalTensor，
- 一个 accumulator tile 加一个 receive tile，或 accumulator tile 加一对 ping/pong receive tile，
- `ReduceOp`，
- 可选的等待事件 token。

wrapper 会先等待所有传入事件，然后再委托给后端 reduce 实现。

## 汇编语法

```text
pto.treduce %group, %dst {op = #pto.reduce_op<Sum>} : (!pto.group<...>, !pto.memref<...>)
```

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`。

```cpp
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &accTileData, TileData &recvTileData, ReduceOp op, WaitEvents &... events);

template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                             ReduceOp op, WaitEvents &... events);
```

## 约束

!!! warning "约束"
    - 并行组缓冲区描述、目标 tensor 与暂存 tile 的元素类型必须兼容。
    - accumulator / receive tile 必须位于 UB。
    - ping/pong receive tile 应位于互不重叠的 UB 区域。
    - wrapper 自身并不编码 root-only 校验；使用者需要遵守后端 collective 实现所期望的语义契约。

## 示例

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void example_reduce(auto &group, auto &dstG, auto &accTile, auto &recvTile) {
    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Sum);
}
```
