# pto.tget

## 概要

`pto.tget` 通过一个或两个显式 UB 暂存 tile，把远端 GlobalTensor 的数据远程读入到本地 GlobalTensor。

## 语义

从概念上看，`TGET` 将远端源 tensor 复制到本地目标 tensor：

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \mathrm{src}^{\mathrm{remote}}_{i,j} $$

公共 API 支持：

- 单暂存 tile 形式，
- 乒乓双缓冲形式。

## 汇编语法

```text
pto.tget %dst_local, %src_remote : (!pto.memref<...>, !pto.memref<...>)
```

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`。

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &stagingTileData,
                          WaitEvents &... events);

template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &pingTile,
                          TileData &pongTile, WaitEvents &... events);
```

## 约束

!!! warning "约束"
    - `dstGlobalData`、`srcGlobalData` 与暂存 tile 的元素类型必须兼容。
    - 源与目标的 layout 必须兼容。
    - 暂存 tile 必须位于 UB，且大小适合所选分块策略。
    - 乒乓 tile 应位于互不重叠的 UB 区域。

## 面向目标的说明

- wrapper 在进入实现前会先等待所有传入事件 token。
- 双缓冲是显式 API 选择，不是单 tile overload 的隐式优化。

## 示例

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void example_get(auto &dstG, auto &srcG, auto &stagingTile) {
    comm::TGET(dstG, srcG, stagingTile);
}
```
