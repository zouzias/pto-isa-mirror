# TGET_ASYNC

## 简介

`TGET_ASYNC` 是异步远程读原语。它启动从远程 GM 到本地 GM 的传输，并立即返回一个 `AsyncEvent`。

数据流：

`srcGlobalData（远程 GM）-> DMA 引擎 -> dstGlobalData（本地 GM）`

## 模板参数

- `engine`：
  - `DmaEngine::SDMA`（默认）
  - `DmaEngine::URMA`（待实现）

> **重要提示（SDMA 路径）**  
> 使用 `DmaEngine::SDMA` 的 `TGET_ASYNC` 当前仅支持**扁平连续的逻辑 1D tensor**。  
> 当前 SDMA 异步实现不支持非 1D 或非连续布局。

## C++ Intrinsic

声明在 `include/pto/comm/pto_comm_inst.hpp`。

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);
```

`AsyncSession` 是引擎无关的会话对象。使用 `BuildAsyncSession<engine>()` 构建一次，然后传递给所有异步调用和事件等待。模板 `engine` 参数在编译时选择 DMA 后端，使代码对未来引擎（URMA、CCU 等）保持前向兼容。

## AsyncSession 构造

使用 `include/pto/comm/async/async_event_impl.hpp` 中的 `BuildAsyncSession`：

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {32 * 1024, 0, 1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

engine 模板参数选择后端（当前仅支持 SDMA）。

带默认值的参数：

| 参数 | 默认值 | 描述 |
|---|---|---|
| `syncId` | `0` | MTE3/MTE2 流水线同步事件 id（0-7）。如果 kernel 在相同 id 上使用了其他流水线屏障，需要覆盖此值。 |
| `baseConfig` | `{32*1024, 0, 1}` | `{block_bytes, comm_block_offset, queue_num}`。适用于大多数单队列传输场景。 |
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA 通道组索引。默认内部使用 `get_block_idx()`，映射到当前 AI core。在多 block 或自定义通道映射场景下需要覆盖。 |

## 约束

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- SDMA 路径要求源 tensor 为**扁平连续的逻辑 1D**
- workspace 必须是由 host 端 `SdmaWorkspaceManager` 分配的有效 GM 指针

如果不满足 1D 连续性要求，当前实现返回无效的 async event（`handle == 0`）。

## scratchTile 的作用

`scratchTile` **不**用于承载传输的载荷数据。  
它被转换为 `TmpBuffer`，用作临时 UB 工作空间：

- 写入/读取 SDMA 控制字（flag、sq_tail、channel_info）
- 轮询事件完成标志
- 完成时提交队列尾部

实际载荷路径为 远程 GM -> DMA 引擎 -> 本地 GM；`scratchTile` 仅用于控制/同步元数据。

## scratchTile 类型和大小约束

- 必须是 `pto::Tile` 类型
- 必须是 UB/Vec tile（`ScratchTile::Loc == TileType::Vec`）
- 可用字节数必须至少为 `sizeof(uint64_t)`（8 字节）

推荐：`Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>`（256B）。

## 完成语义

使用 `AsyncEvent` 进行同步：

- `event.Wait(session)` — 阻塞直到传输完成

等待成功后，对 `dstGlobalData` 的读取已完成。

## 示例

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimpleGet(__gm__ T *localDst, __gm__ T *remoteSrc,
                                 __gm__ uint8_t *sdmaWorkspace)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(localDst, shape, stride);
    GT srcG(remoteSrc, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    auto event = comm::TGET_ASYNC<comm::DmaEngine::SDMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```
