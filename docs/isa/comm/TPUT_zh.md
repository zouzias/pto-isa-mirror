# TPUT

## 简介

远程写操作：将本地数据写入远程 NPU 的内存。数据通过 UB tile 作为中间暂存缓冲区进行传输。

当 GlobalTensor 超过 UB tile 容量时，TPUT 自动执行 **2D 滑窗** — 沿行（DIM_3）和列（DIM_4）分块以适应 tile 大小，并遍历所有外层维度（DIM_0、DIM_1、DIM_2）。

## 数学表示

对有效区域中的每个元素 `(i, j)`：

$$ \mathrm{dst}^{\mathrm{remote}}_{i,j} = \mathrm{src}^{\mathrm{local}}_{i,j} $$

数据流：`srcGlobalData（本地 GM）` → `stagingTileData（UB）` → `dstGlobalData（远程 GM）`

## 汇编语法

PTO-AS 形式：见 [PTO-AS 规范](../../assembly/PTO-AS.md)。

同步形式：

```text
tput %dst_remote, %src_local : (!pto.memref<...>, !pto.memref<...>)
```
下降后引入 UB 暂存 tile 用于 GM→UB→GM 数据路径；C++ intrinsic 需要显式的 `stagingTileData`（或 `pingTile` / `pongTile`）操作数。

## C++ Intrinsic

声明在 `include/pto/comm/pto_comm_inst.hpp`

### 单 tile（自动分块）

```cpp
template <AtomicType atomicType = AtomicType::AtomicNone,
          typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData, WaitEvents&... events);
```

### Ping-pong 双缓冲

使用两个暂存 tile 重叠相邻块的 TLOAD 和 TSTORE，将一次 DMA 传输隐藏在另一次之后。

```cpp
template <AtomicType atomicType = AtomicType::AtomicNone,
          typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

### 运行时原子类型

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData, AtomicType atomicType, WaitEvents&... events);
```

## 约束

- **类型约束**：
  - `GlobalSrcData::RawDType` 必须等于 `GlobalDstData::RawDType`。
  - `TileData::DType` 必须等于 `GlobalSrcData::RawDType`。
  - `GlobalSrcData::layout` 必须等于 `GlobalDstData::layout`。
- **内存约束**：
  - `dstGlobalData` 必须指向远程地址（目标 NPU 上）。
  - `srcGlobalData` 必须指向本地地址（当前 NPU 上）。
  - `stagingTileData` / `pingTile` / `pongTile` 必须在 Unified Buffer 中预分配。
- **有效区域**：
  - 传输大小由 `GlobalTensor` shape 决定（自动分块以适应 tile）。
- **原子操作**：
  - `atomicType` 支持 `AtomicNone` 和 `AtomicAdd`。
- **Ping-pong**：
  - `pingTile` 和 `pongTile` 必须具有相同的类型和维度。
  - 必须位于不重叠的 UB 偏移地址。

## 示例

### 基本用法

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_tput(__gm__ T* local_data, __gm__ T* remote_addr) {
    using TileT = Tile<TileType::Vec, T, 16, 16>;
    using GShape = Shape<1, 1, 1, 16, 16>;
    using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
    /* 
    如果 globalTensor 大于 UB Tile，TPUT 将自动执行 2D 滑窗。
    using GShape = Shape<1, 1, 1, 4096, 4096>;
    using GStride = BaseShape2D<T, 4096, 4096, Layout::ND>;
    */
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    GTensor srcG(local_data);
    GTensor dstG(remote_addr);
    TileT stagingTile;
    TASSIGN(stagingTile, 0);

    // 基本远程写
    comm::TPUT(dstG, srcG, stagingTile);

    // 带原子加的远程写
    comm::TPUT<AtomicType::AtomicAdd>(dstG, srcG, stagingTile);
}
```

### Ping-pong 双缓冲

```cpp
constexpr size_t tileUBBytes = ((64 * 64 * sizeof(float) + 1023) / 1024) * 1024;
TileT pingTile(64, 64);
TileT pongTile(64, 64);
TASSIGN(pingTile, 0);
TASSIGN(pongTile, tileUBBytes);  // 不重叠的 UB 区域

// 重叠 TLOAD[i+1] 和 TSTORE[i] 以提高流水线利用率
comm::TPUT(dstG, srcG, pingTile, pongTile);
```

### 运行时原子类型

```cpp
// 在运行时（而非编译时模板参数）选择原子类型
comm::TPUT(dstG, srcG, stagingTile, AtomicType::AtomicAdd);
```
