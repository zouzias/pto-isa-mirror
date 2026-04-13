# TPUT_ASYNC

## 简介

`TPUT_ASYNC` 是异步远程写原语。它启动一次从本地 GM 到远端 GM 的传输，并立即返回 `AsyncEvent`。

数据流：

`srcGlobalData（本地 GM）` → DMA 引擎 → `dstGlobalData（远端 GM）`

## 模板参数

- `engine`：
  - `DmaEngine::SDMA`（默认）
  - `DmaEngine::URMA`（待实现）

> **注意（SDMA 路径）**
> `TPUT_ASYNC` 配合 `DmaEngine::SDMA` 目前**仅支持扁平连续的逻辑一维 tensor**。
> 当前 SDMA 异步实现不支持非一维或非连续布局。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);
```

`AsyncSession` 是引擎无关的会话对象。使用 `BuildAsyncSession<engine>()` 构建一次后，传递给所有异步调用和事件等待。模板参数 `engine` 在编译期选择 DMA 后端，使代码对未来引擎（URMA、CCU 等）保持前向兼容。

## AsyncSession 构建

使用 `include/pto/comm/async/async_event_impl.hpp` 中的 `BuildAsyncSession`：

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {1024 * 1024, 0, 1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

### 参数一览


| 参数                             | 类型                                               | 是否必填  | 默认值                                                       | 合法范围                  | 说明                                                                                                                                                               |
| ------------------------------ | ------------------------------------------------ | ----- | --------------------------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scratchTile`                  | `Tile<TileType::Vec, uint8_t, 1, UB_ALIGN_SIZE>` | **是** | —                                                         | 可用字节 >= 8             | UB Vec tile，用作 SDMA 控制面读写的临时缓冲（doorbell、event record、tail 回写等）。**不承载**用户数据。异步期间不可挪作他用。                                                                           |
| `workspace`                    | `__gm__ uint8_t` *                               | **是** | —                                                         | 非 `nullptr`           | 设备侧 SDMA 上下文 GM 区域（channel 信息、SQ ring 地址、workspace 标志位）。由 host 侧 `SdmaWorkspaceManager` + AICPU 算子初始化。布局：`BatchWriteFlagInfo(64B)`                               |
| `syncId`                       | `uint32_t`                                       | 否     | `0`                                                       | 0–7（MTE 事件号）          | MTE3/MTE2 管道同步事件编号，供 `SetValue`/`GetValue` 的 `set_flag`/`wait_flag` 配对使用。仅当 kernel 在同一编号上有其他管道屏障时才需覆盖。                                                           |
| `baseConfig.block_bytes`       | `uint64_t`                                       | 否     | `1048576`（1 MiB）                                          | > 0                   | 每条 data SQE 的最大搬运字节数。总传输量拆分为 `ceil(transfer_size / block_bytes)` 条 SQE。值越大控制面开销越低但粒度越粗；单 queue 上的 SQE 总数（`ceil(iter_num / queue_num) + 1`）不得超过 `kSqDepth`（2048）。 |
| `baseConfig.comm_block_offset` | `uint64_t`                                       | 否     | `0`                                                       | >= 0                  | 字节偏移，叠加到 src 和 dst 基地址上：`addr = data() + offset + idx * block_bytes`。典型场景：多核分片传输同一大 buffer。调用方须保证 `offset + transfer_size` 不越界（无运行时检查）。                          |
| `baseConfig.queue_num`         | `uint32_t`                                       | 否     | `1`                                                       | 1 – 48                | 本 block 占用的并行 SDMA channel（SQ ring）数量。data SQE 按 `idx % queue_num` 轮转分发。并发 block 数上限 = `48 / queue_num`。建议从 1 起步。                                                      |
| `channelGroupIdx`              | `uint32_t`                                       | 否     | `kAutoChannelGroupIdx`（`UINT32_MAX`）→ 取 `get_block_idx()` | `[0, 48 / queue_num)` | 选择本 block 使用的连续 channel 组，从 `idx * queue_num` 起占 `queue_num` 个槽位。越界时静默返回 `AsyncEvent(handle=0)`。                                                                 |


## 约束

- src 和 dst tensor 的数据类型和内存布局必须一致，且均须为**扁平连续的逻辑一维**
- dst tensor 的元素数量须大于等于 src tensor 的元素数量
- src 和 dst 的 GM 指针须非 `nullptr`，即 HCCL window 须已初始化且远端地址有效
- `blockDim * queue_num` 须小于等于48（即 `channelGroupIdx < 48 / queue_num`）
- 单 queue 上的 SQE 总数不得超过2048
- workspace 必须是由主机侧 `SdmaWorkspaceManager` 分配的有效 GM 指针（非 `nullptr`）
- 多 block 并发时，各 block 的 **`channelGroupIdx` 必须互不重叠**，否则 SQ ring 和 workspace 会互相踩踏。使用默认值 `kAutoChannelGroupIdx`时天然满足；手动指定时调用方须自行保证唯一性
- 异步操作 in-flight 期间，必须先调用 **`event.Wait(session)`** 等待全部完成，才能销毁 session / scratchTile 或开启新一轮批量提交。否则 SQ ring 可能溢出覆盖未消费的 SQE
- **`comm_block_offset + transfer_size`** 不得超过 src/dst buffer 的实际大小，否则产生内存越界

> **A5 平台说明**
>
> 在 A5（Atlas 推理系列）架构上，SDMA 硬件不支持 PUT（写）方向的远端传输。因此 `TPUT_ASYNC` 的 SDMA 实现在 A5 上是**伪实现**：实际使用 MTE2/MTE3 管道进行 GM → UB → GM 的分块同步搬运。该回退路径在功能上等价，但操作是**同步完成**的——返回的 `AsyncEvent` 的 `handle` 始终为 0（表示已完成）。这意味着在 A5 上调用 `TPUT_ASYNC` 后无需等待，但也无法获得异步流水线重叠的性能收益。

## 完成语义（Quiet 语义）

`TPUT_ASYNC` 仅提交数据传输 SQE，不提交 flag SQE。flag SQE 的提交延迟到 `Wait` 调用时进行。

- `event.Wait(session)` — 提交 flag SQE 并阻塞，直到**自上次 Wait 以来所有已发出的异步操作**全部完成

这意味着多次 `TPUT_ASYNC` 调用后，只需对最后一个返回的 `AsyncEvent` 调用一次 `Wait`，即可等待所有 pending 操作完成（类似 shmem 的 quiet 语义）。

wait 成功后，所有已发出的 `dstGlobalData` 写入均已全部完成。

## 示例

### 单次传输

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimplePut(__gm__ T *remoteDst, __gm__ T *localSrc,
                                 __gm__ uint8_t *sdmaWorkspace)
{
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(remoteDst, shape, stride);
    GT srcG(localSrc,  shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    auto event = comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```

### 批量传输（Quiet 语义）

```cpp
template <typename T>
__global__ AICORE void BatchPut(__gm__ T *remoteDstBase, __gm__ T *localSrc,
                                __gm__ uint8_t *sdmaWorkspace, int nranks)
{
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT srcG(localSrc, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    comm::AsyncEvent lastEvent;
    for (int rank = 0; rank < nranks; ++rank) {
        GT dstG(remoteDstBase + rank * 1024, shape, stride);
        lastEvent = comm::TPUT_ASYNC(dstG, srcG, session);
    }
    (void)lastEvent.Wait(session);  // 一次 Wait 等待所有 pending 操作
}
```

