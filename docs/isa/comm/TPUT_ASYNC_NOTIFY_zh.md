# TPUT_ASYNC_NOTIFY

## 简介

`TPUT_ASYNC_NOTIFY` 启动一次从本端 GM 到远端 GM 的数据传输，并在本次数据传输完成后更新远端
`int32_t` signal，用于通知远端数据已就绪。signal 支持 `Set` 和 `AtomicAdd`，可分别用于状态发布
和完成计数。

数据流：

```text
srcGlobalData（本地 GM） -> dstGlobalData（远端 GM）
                                  |
                                  v
                       dstSignalData（远端 int32 signal）
```

调用者必须提前准备好 payload 目的地址和 signal 的远端地址。该指令不根据 rank 自动进行地址转换，
也不负责分配 signal 内存。

## 操作语义

对 payload 有效区域内的每个元素：

$$\mathrm{dst}^{\mathrm{remote}}_i = \mathrm{src}^{\mathrm{local}}_i$$

payload 操作之后，根据 `notifyOp` 更新 signal：

- `NotifyOp::Set`

  $$\mathrm{signal}^{\mathrm{remote}} = \mathrm{signalValue}$$

- `NotifyOp::AtomicAdd`

  $$\mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{signalValue}$$

`AtomicAdd` 的 signal 更新是原子的；它不使不同发送者写入的 payload 自动具备原子性。

## 模板参数

- `engine`：当前仅支持 `DmaEngine::SDMA`，默认值也是 `DmaEngine::SDMA`。
- `GlobalDstData`：远端 payload 目的 tensor 类型。
- `GlobalSrcData`：本地 payload 源 tensor 类型。
- `GlobalSignalData`：远端 signal tensor 类型，其 `RawDType` 必须是 `int32_t`。
- `WaitEvents...`：调用本指令前必须完成、且提供零参数 `Wait()` 的 PTO pipeline event 类型。当前
  `AsyncEvent` 需要 `Wait(session)`，不能直接作为该参数传入。

当前 A2/A3 和 A5 的 URMA notify 路径以及 A5 的 RoCE notify 路径均未实现。显式指定
`TPUT_ASYNC_NOTIFY<DmaEngine::URMA>` 或 `TPUT_ASYNC_NOTIFY<DmaEngine::ROCE>` 会在编译期被拒绝。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`。

### 通用接口

A2/A3 和 A5 均提供：

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData,
          typename GlobalSrcData,
          typename GlobalSignalData,
          typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC_NOTIFY(
    GlobalDstData &dstGlobalData,
    GlobalSrcData &srcGlobalData,
    GlobalSignalData &dstSignalData,
    int32_t signalValue,
    NotifyOp notifyOp,
    const AsyncSession &session,
    WaitEvents&... events);
```

### A5 explicit-peer overload

A5 还提供与 `TPUT_ASYNC` 一致的 overload：

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData,
          typename GlobalSrcData,
          typename GlobalSignalData,
          typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC_NOTIFY(
    GlobalDstData &dstGlobalData,
    GlobalSrcData &srcGlobalData,
    GlobalSignalData &dstSignalData,
    int32_t signalValue,
    NotifyOp notifyOp,
    const AsyncSession &session,
    uint32_t peer,
    WaitEvents&... events);
```

当前 A5 SDMA/MTE 路径通过 `dstGlobalData` 和 `dstSignalData` 中已经转换好的远端虚拟地址确定目标，
因此 `peer` 仅用于保持接口一致，当前不会参与寻址。

## 参数说明

| 参数 | 地址归属 | 说明 |
|---|---|---|
| `dstGlobalData` | 远端 GM | payload 目的 tensor，必须包含已转换的远端地址 |
| `srcGlobalData` | 本地 GM | payload 源 tensor |
| `dstSignalData` | 远端 GM | signal tensor，通常使用 `comm::Signal` 包装远端 `int32_t*` |
| `signalValue` | 本地标量 | SET 的目标值，或 AtomicAdd 的增量 |
| `notifyOp` | 本地枚举 | `NotifyOp::Set` 或 `NotifyOp::AtomicAdd` |
| `session` | 本地上下文 | 通过 `BuildAsyncSession<DmaEngine::SDMA>` 构建 |
| `peer` | rank id | 仅 A5 overload 存在；当前 SDMA/MTE 路径忽略 |
| `events...` | 本地 event | 指令开始前自动等待这些提供零参数 `Wait()` 的依赖完成 |

`comm::Signal` 是以下单元素 tensor 的别名：

```cpp
using Signal =
    GlobalTensor<int32_t,
                 Shape<1, 1, 1, 1, 1>,
                 Stride<1, 1, 1, 1, 1>,
                 Layout::ND>;
```

构造 `Signal` 只会包装地址，不会分配内存或执行远端地址转换。

## AsyncSession 构建

公共用法应由 Host 侧 `SdmaWorkspaceManager` 准备 workspace，再在 AICore kernel 中使用
`BuildAsyncSession<DmaEngine::SDMA>` 构建 session。不要手工填写 `AsyncSession`。session、
workspace 和 scratch tile 的生命周期必须覆盖本次调用及 event 等待。

A2/A3 必须显式传入 `channelGroupIdx = 0`，不能使用会根据 `get_block_idx()` 自动选择 channel group
的默认值：

```cpp
comm::sdma::SdmaBaseConfig config{
    comm::sdma::kDefaultSdmaBlockBytes, 0, 1};

bool ok = comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
    scratchTile, sdmaWorkspace, session,
    0,       // syncId
    config,
    0);      // channelGroupIdx
```

这也意味着当前 A2/A3 notify 只支持每个 rank 一个发送 AICore、一个 active session。A5 使用相同的
session 构建接口，但本指令在 A5 上会在调用返回前完成。

workspace 初始化和其他构建参数参见 [TPUT_ASYNC](TPUT_ASYNC_zh.md)。

## 约束

- **引擎约束**
  - 当前仅支持 `DmaEngine::SDMA`。
- **payload 类型与布局**
  - `GlobalSrcData::RawDType` 必须等于 `GlobalDstData::RawDType`。
  - `GlobalSrcData::layout` 必须等于 `GlobalDstData::layout`。
  - `srcGlobalData` 和 `dstGlobalData` 必须是扁平连续的逻辑一维 tensor。
  - `dstGlobalData` 的元素容量必须不小于 `srcGlobalData`。
  - payload 元素数必须大于 0。A2/A3 的零长度提交不会更新 signal。
- **signal**
  - `GlobalSignalData::RawDType` 必须是 `int32_t`。
  - signal 地址必须非空并按 4 字节对齐。
  - signal 内存必须由调用者预先分配并初始化。
- **地址**
  - `srcGlobalData` 必须指向当前 NPU 的本地 GM。
  - 常规用法中，`dstGlobalData` 和 `dstSignalData` 必须指向同一目标 rank 的远端 GM。
- **session**
  - `session.valid` 必须为 `true`，且 `session.engine == DmaEngine::SDMA`。
  - A2/A3 要求 `channelGroupIdx == 0`，且每个 rank 只能有一个发送 AICore 和一个 active session。
  - A5 上同一 session 也不应由多个独立执行流并发修改。

## 完成语义

### A2/A3

返回的 `AsyncEvent` 覆盖本次 payload、signal 更新和完成标志。成功提交时
`event.valid() == true`。若 `event.valid() == false`，表示本次 A2/A3 提交失败，不能仅通过
`Wait()` 的返回值把它当作成功。

提交通常会在操作完成前返回。当 SDMA 发送队列接近环回时，提交过程可能等待同一 session
中的最新 event 完成并回收已消费的 SQ entry，然后再继续提交新任务。

成功提交后调用：

```cpp
event.Wait(session);
```

成功返回后，本次组合操作以及同一 session 中由该 event 覆盖的此前操作已经完成。

### A5

A5 路径在调用返回前完成 payload 和 signal 更新。成功返回的 event 当前也使用 `handle == 0`，
所以 `event.valid() == false` 不能在 A5 上解释为提交失败；`Wait(session)` 和 `Test(session)`
会将该 event 视为已完成。

因此 A5 的该路径保持了异步接口形状，但当前不会在函数返回后继续与调用者并发执行。
由于成功 event 和默认无效 event 都可能表现为 `handle == 0`，A5 上
`Wait/Test == true` 本身也不能作为独立的成功诊断依据；参数错误由当前实现的断言机制处理。

接收端的 `TWAIT/TTEST` 只维护 signal cache line。接收端读取远端生产者写入的 payload 前，还必须
按目标平台要求维护 payload cache；不能把“signal 条件满足”理解为自动刷新全部 payload cache。
对接收端即时轮询可见性有严格要求的协议，应在目标软硬件版本上进行验证。

## 同步与并发限制

- 单个生产者发布状态时，优先使用 `Set`。
- A5 上，多个生产者向同一 signal 报告完成次数时，可以将 `AtomicAdd` 用作计数协议，前提是每个
  生产者使用独立 session/资源且 payload 地址不冲突；该并发组合仍需按目标环境验证。
- A2/A3 当前固定使用 channel group 0，只支持每 rank 一个发送 AICore/active session，不能把它
  作为同一 rank 多 AICore 并发计数接口使用。
- 多个生产者对同一 signal 使用 `Set` 时是覆盖语义，不能通过最终值判断所有生产者均已完成。
- 使用共享 AtomicAdd 前必须初始化 signal，并确保不存在并发 SET 或普通 store。
- signal 更新的原子性不保护 payload；不同发送者必须使用互不冲突的 payload 目的区域，或自行提供
  额外同步协议。
- 一个 signal 只证明协议明确关联的 payload，不自动代表发往同一 rank 的其他操作均已完成。

## 示例

以下示例假定：

- `remoteDst` 和 `remoteSignal` 已经由通信运行时转换为目标 rank 的远端地址；
- Host 已通过 `SdmaWorkspaceManager` 初始化 `sdmaWorkspace`；
- 示例 kernel 只启动一个 AICore。

### SET：发送 payload 并发布就绪标志

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void put_and_set(__gm__ T *remoteDst,
                                   __gm__ T *localSrc,
                                   __gm__ int32_t *remoteSignal,
                                   __gm__ uint8_t *sdmaWorkspace)
{
    using GShape = Shape<1, 1, 1, 1, 1024>;
    using GStride = Stride<1024, 1024, 1024, 1024, 1>;
    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;
    using ScratchTile =
        Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    GT dst(remoteDst);
    GT src(localSrc);
    comm::Signal signal(remoteSignal);
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0);

    comm::AsyncSession session;
    comm::sdma::SdmaBaseConfig config{
        comm::sdma::kDefaultSdmaBlockBytes, 0, 1};
    if (!comm::BuildAsyncSession(
            scratchTile, sdmaWorkspace, session, 0, config, 0)) {
        return;
    }
    comm::AsyncEvent event = comm::TPUT_ASYNC_NOTIFY(
        dst, src, signal, 1, comm::NotifyOp::Set, session);
#ifdef PTO_NPU_ARCH_A2A3
    if (!event.valid()) {
        return;
    }
#endif
    if (!event.Wait(session)) {
        return;
    }
}
```

接收端可使用本地 signal 地址等待：

```cpp
__global__ AICORE void wait_until_ready(__gm__ int32_t *localPayload,
                                        __gm__ int32_t *localSignal)
{
    comm::Signal signal(localSignal);
    comm::TWAIT(signal, 1, comm::WaitCmp::EQ);

    // TWAIT only maintains the signal cache line. Invalidate payload cache
    // before consuming data written by the remote producer.
    dcci((__gm__ void *)localPayload, cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
    // 在此读取与 signal 关联的 payload。
}
```

### AtomicAdd：累加完成计数

以下片段复用前一示例已经构建的有效 session。A2/A3 当前仅用于单发送 AICore 场景。

```cpp
template <typename GT>
PTO_INTERNAL comm::AsyncEvent put_and_count(
    GT &remoteDst,
    GT &localSrc,
    __gm__ int32_t *remoteCounter,
    comm::AsyncSession &session)
{
    comm::Signal counter(remoteCounter);
    return comm::TPUT_ASYNC_NOTIFY(
        remoteDst, localSrc, counter, 1,
        comm::NotifyOp::AtomicAdd, session);
}
```

接收端等待 `expectedProducers` 个生产者，并在读取 payload 前执行相应 cache 维护：

```cpp
comm::Signal counter(localCounter);
comm::TWAIT(counter, expectedProducers, comm::WaitCmp::GE);
dcci((__gm__ void *)localPayload, cache_line_t::ENTIRE_DATA_CACHE);
dsb(DSB_DDR);
```

### 等待前序 AsyncEvent 后再发送通知

以下片段中的 `remoteDst0/1`、`localSrc0/1` 均为已经构造好的 `GlobalTensor`：

```cpp
comm::AsyncEvent previous = comm::TPUT_ASYNC(remoteDst0, localSrc0, session);
comm::Signal signal(remoteSignal);

bool previousReady = previous.Wait(session);
#ifdef PTO_NPU_ARCH_A2A3
previousReady = previous.valid() && previousReady;
#endif
if (previousReady) {
    comm::AsyncEvent notified = comm::TPUT_ASYNC_NOTIFY(
        remoteDst1, localSrc1, signal, 2,
        comm::NotifyOp::Set, session);
#ifdef PTO_NPU_ARCH_A2A3
    if (!notified.valid()) {
        return;
    }
#endif
    if (!notified.Wait(session)) {
        return;
    }
}
```

`AsyncEvent` 必须像上例一样显式传入 session 等待。只有提供零参数 `Wait()` 的 PTO pipeline event
才能直接放入 `events...`。
