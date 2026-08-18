# TPUT_ASYNC_NOTIFY

## 简介

`TPUT_ASYNC_NOTIFY` 是带通知的异步远程写原语。它先将非空payload从本地GM传输到远端GM，再更新一个
远端32位signal。返回的 `AsyncEvent` 同时表示payload传输和signal更新两部分操作的完成状态。

数据流：

```text
srcGlobalData（本地 GM） -> DMA引擎 -> dstGlobalData（远端 GM）
                                           |
                                      payload完成后
                                           v
                                  dstSignalData（远端 GM）
```

payload目的地址和signal地址都必须由调用方提前转换为远端地址；该指令不会根据rank转换地址，也不会
分配signal内存。如果只需要更新signal、不需要传输payload，应使用 `TNOTIFY`。

## 模板参数与后端支持

- `engine` 在编译期选择DMA后端：

| 引擎 | 平台或后端限制 | 支持的 `NotifyOp` |
|---|---|---|
| `DmaEngine::SDMA`（默认） | A2/A3和A5 | `Set`、`AtomicAdd` |
| `DmaEngine::URMA` | 仅A5（Ascend 950PR/Ascend 950DT）、NPU_ARCH 3510；要求CANN Toolkit >= 9.1.0 | `Set`、`AtomicAdd` |
| `DmaEngine::RDMA` | 仅A5（Ascend 950PR/Ascend 950DT）、NPU_ARCH 3510；HNS1825 RoCE网卡平台 | 仅 `Set`；不支持 `AtomicAdd` |

引擎在编译期选择。构建RDMA版本时，必须在CMake配置阶段设置 `PTO_RDMA_BACKEND=HNS_1825`；该选项
不是运行时的引擎选择参数。

## C++内建接口

公开接口声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData,
          typename GlobalSrcData,
          typename GlobalSignalData,
          typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC_NOTIFY(GlobalDstData &dstGlobalData,
                                      GlobalSrcData &srcGlobalData,
                                      GlobalSignalData &dstSignalData,
                                      int32_t signalValue,
                                      NotifyOp notifyOp,
                                      const AsyncSession &session,
                                      WaitEvents &... events);
```

A5上还提供显式目标peer重载：

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData,
          typename GlobalSrcData,
          typename GlobalSignalData,
          typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC_NOTIFY(GlobalDstData &dstGlobalData,
                                      GlobalSrcData &srcGlobalData,
                                      GlobalSignalData &dstSignalData,
                                      int32_t signalValue,
                                      NotifyOp notifyOp,
                                      const AsyncSession &session,
                                      uint32_t peer,
                                      WaitEvents &... events);
```

对于URMA和RDMA，`peer` 用于选择该peer对应的队列和远端内存元数据，`dstGlobalData` 与
`dstSignalData` 必须都指向该peer。对于SDMA，远端虚拟地址已包含在 `GlobalTensor` 中，因此忽略 `peer`。

可选的 `events` 参数包接受零个或多个带零参数 `Wait()` 方法的PTO流水事件。发起payload传输前，接口会
等待所有传入事件。`AsyncEvent` 需要调用 `Wait(session)`，因此不能直接放入这个参数包；若前序操作返回
`AsyncEvent`，应先用对应Session显式等待。不传依赖事件也是合法用法。

简写形式可能把末尾参数写作 `session [, peer] [, events...]`。中括号表示其中参数可选，只是文档记法，
C++调用时不写中括号。

## 参数说明

| 参数 | 说明 |
|---|---|
| `dstGlobalData` | payload在远端GM中的目的地址。 |
| `srcGlobalData` | payload在本地GM中的源地址。 |
| `dstSignalData` | 远端GM中的一个 `int32_t` signal。推荐使用 `comm::Signal`；它只封装调用方传入的地址，不负责分配内存。 |
| `signalValue` | `Set` 写入的值，或 `AtomicAdd` 使用的加数。 |
| `notifyOp` | signal更新操作：`NotifyOp::Set` 或 `NotifyOp::AtomicAdd`。 |
| `session` | 为所选 `engine` 构建的现有异步Session。 |
| `peer` | A5上可选的显式目标rank；URMA和RDMA多peer场景推荐使用。 |
| `events` | 零个或多个带零参数 `Wait()` 方法的前置PTO流水事件。 |

返回值为 `AsyncEvent`，其完成范围同时覆盖payload传输和signal更新。

### Signal与 `signalValue`

`comm::Signal` 是只含一个 `int32_t` 元素的全局tensor别名：

```cpp
using Signal = GlobalTensor<int32_t,
                            Shape<1, 1, 1, 1, 1>,
                            Stride<1, 1, 1, 1, 1>,
                            Layout::ND>;
```

构造 `Signal` 只会封装调用方提供的GM地址，不会分配或初始化底层 `int32_t`。调用方应按照应用协议
初始化signal。`signalValue` 是 `Set` 发布的值，或 `AtomicAdd` 使用的有符号增量。接收端通常先用
`TWAIT` 或 `TTEST` 观察这个值，再消费对应payload。

## 操作语义

单次调用的可观察顺序为：

1. 等待所有传入的依赖事件。
2. 将完整payload从 `srcGlobalData` 传输到 `dstGlobalData`。
3. payload到达远端GM后，按照 `notifyOp` 更新 `dstSignalData`。

`NotifyOp::Set` 的语义为：

$$
\mathrm{signal}^{\mathrm{remote}} = \mathrm{signalValue}
$$

`NotifyOp::AtomicAdd` 的语义为：

$$
\mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{signalValue} \quad (\text{原子操作})
$$

`DmaEngine::RDMA` 不支持 `NotifyOp::AtomicAdd`。

原子性只作用于 `AtomicAdd` 对signal的更新，不会使payload写入变成原子操作。`Set` 只定义单次调用
写入的值；多个生产者并发更新同一个signal时，接口不承诺胜出者或计数语义。除非应用另有同步协议，
不要在同一个signal上并发混用 `Set`、`AtomicAdd` 或普通store。

上述顺序只约束同一次调用中的payload与signal更新，不定义不同Session或不同执行流独立发起的操作之间的顺序。

## AsyncSession构建

在AICore kernel中使用所选引擎对应的重载构建 `AsyncSession`，不要手工填写Session字段。发起操作前
必须检查构建函数的布尔返回值，并保证Session及其引用的全部资源存活到相关Event完成。

### SDMA构建（默认）

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile &scratchTile,
    __gm__ uint8_t *workspace,
    AsyncSession &session,
    uint32_t syncId = 0,
    const sdma::SdmaBaseConfig &baseConfig = {
        sdma::kDefaultSdmaBlockBytes, 0, 1},
    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `scratchTile` | - | SDMA控制和同步元数据使用的UB/Vec临时tile。 |
| `workspace` | - | 由Host侧 `SdmaWorkspaceManager` 初始化的GM指针。 |
| `session` | - | 输出Session。 |
| `syncId` | `0` | 取值0-7的MTE3/MTE2同步事件ID；kernel已占用该ID时应改用其他ID。 |
| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | SDMA块字节数、通信块偏移和队列数。 |
| `channelGroupIdx` | `kAutoChannelGroupIdx` | 通道组。默认根据 `get_block_idx()` 推导；并发Session必须分配相互隔离的通道组。 |

Host侧管理器声明于 `include/pto/comm/async/sdma/sdma_workspace_manager.hpp`。

### URMA构建（仅NPU_ARCH 3510）

```cpp
#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(
    __gm__ uint8_t *workspace,
    AsyncSession &session);

template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(
    __gm__ uint8_t *workspace,
    uint32_t destRankId,
    AsyncSession &session);
#endif
```

| 参数 | 说明 |
|---|---|
| `workspace` | 由Host侧 `UrmaWorkspaceManager` 初始化的GM指针。 |
| `destRankId` | 兼容Session重载所绑定的目的rank。 |
| `session` | 输出Session。 |

A5新代码建议使用不带 `destRankId` 的peer无关重载，再把目的rank传给带显式 `peer` 的
`TPUT_ASYNC_NOTIFY` 重载。这样一个Session可以与多个peer通信。为兼容已有用法，仍保留绑定peer的构建
函数，可与不带显式 `peer` 的指令重载配合使用。

URMA不需要 `scratchTile`，要求CANN Toolkit >= 9.1.0。传给 `UrmaWorkspaceManager::Init()` 的对称数据
buffer必须使用大页内存，应通过 `ACL_MEM_MALLOC_HUGE_ONLY` 分配；小块内存若使用
`ACL_MEM_MALLOC_HUGE_FIRST`，可能回退到4 KB页并导致内存注册失败。Host侧管理器声明于
`include/pto/comm/async/urma/urma_workspace_manager.hpp`。

### RDMA构建（仅NPU_ARCH 3510）

```cpp
#ifdef PTO_RDMA_SUPPORTED
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile &scratchTile,
    __gm__ uint8_t *workspace,
    uint32_t myPe,
    AsyncSession &session,
    uint32_t syncId = 0);

template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile &scratchTile,
    __gm__ uint8_t *workspace,
    uint32_t destRankId,
    uint32_t myPe,
    AsyncSession &session,
    uint32_t syncId = 0);
#endif
```

| 参数 | 说明 |
|---|---|
| `scratchTile` | RDMA控制数据使用的UB/Vec tile，至少64字节。 |
| `workspace` | Host侧RDMA初始化流程返回的GM指针。 |
| `destRankId` | 兼容Session重载所绑定的目的rank。 |
| `myPe` | 用于选择本地已注册内存元数据的本地rank。 |
| `session` | 输出Session。 |
| `syncId` | 取值0-7的MTE/Scalar同步事件ID。 |

A5新代码建议使用不带 `destRankId` 的peer无关重载，再使用带显式 `peer` 的指令重载。为兼容已有
用法，仍保留绑定peer的构建函数，可与不带显式 `peer` 的指令重载配合使用。Host侧接口声明于
`include/pto/comm/async/rdma/rdma_workspace_manager.hpp`。

### `scratchTile` 的作用和大小

`scratchTile` 不存放用户payload。SDMA用它保存队列控制和Event元数据，RDMA用它保存WQE/CQE控制数据；
payload在GM buffer之间直接传输。

- 必须使用UB/Vec内存中的 `pto::Tile`（`ScratchTile::Loc == TileType::Vec`）。
- SDMA要求至少8字节可用空间，推荐类型为
  `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>`（256字节）。
- RDMA要求至少64字节可用空间。
- 通过该tile构建的Session所发Event全部完成前，必须保持tile有效。

## 约束

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`。
- `GlobalSrcData::layout == GlobalDstData::layout`。
- 源和目的payload tensor必须是扁平、连续的逻辑一维tensor。
- 目的tensor的元素容量不得小于源tensor的元素数。payload字节数由源tensor的逻辑元素数和元素宽度确定。
- payload大小必须大于0；仅更新signal时应使用 `TNOTIFY`。
- `dstSignalData` 必须表示远端GM中恰好一个非空、4字节对齐的 `int32_t`；调用方必须提前分配并初始化。
- payload目的地址范围与 `dstSignalData` 不得重叠。
- 对于URMA和RDMA，payload目的地址和signal必须属于Session或显式 `peer` 参数选择的同一个远端peer。
- 对于URMA和RDMA，完整的本地payload、远端payload和远端signal地址范围都必须位于Host初始化阶段
  注册的内存区域内。
- 单次RDMA payload不能超过 `0x7fffffff` 字节。
- RDMA仅支持 `NotifyOp::Set`，不得使用 `NotifyOp::AtomicAdd`。
- `session.valid` 必须为 `true`，Session引擎必须与模板参数 `engine` 一致。
- Session、workspace和所有scratch tile的生命周期必须覆盖相关Event的完成阶段。
- 每个 `events` 参数都必须提供零参数 `Wait()` 方法。

## 完成与顺序语义

返回Event记录的引擎与发起操作的Session一致。下面两个方法都必须传入同一个Session：

- `event.Wait(session)` 阻塞等待；覆盖的操作完成时返回 `true`，检测到完成失败时返回 `false`。
- `event.Test(session)` 对相同完成状态进行非阻塞检测。
- 成功完成同时覆盖完整payload传输及其后的signal更新。

面向用户的完成规则是quiet语义：若多个异步操作按顺序提交到同一个Session、同一个后端队列，等待最新
Event也会覆盖该有序队列中更早的未完成操作。URMA和RDMA发往不同peer的队列相互独立，必须分别等待
每个peer的最后一个Event。URMA或RDMA Event中已编码peer，`Wait`/`Test` 会据此选择完成队列，调用方
无需再次传入 `peer`。

接收端可观察到signal更新时，对应payload已经到达远端GM。该保证只作用于同一次
`TPUT_ASYNC_NOTIFY` 调用。

## Session所有权与并发

- 不要从相互独立的执行流并发使用同一个Session发起操作。
- 共享SDMA通道组的操作也必须共享同一个Session；并发SDMA kernel或相互独立的Session必须使用隔离的
  通道组。
- 重建Session、释放workspace或scratch tile、复用SDMA通道组前，必须完成全部未完成Event。
- 队列或未完成Event达到容量时，后端可以施加提交反压；调用方不得假设可以无限积压异步操作。
- 仅当引擎支持、各生产者的payload范围不冲突，并且Session和后端资源按平台要求隔离时，多个生产者
  才能用 `AtomicAdd` 实现完成计数。

## 接收端payload可见性

该指令不会使接收端可能缓存的 `dstGlobalData` 旧副本失效，也不会主动刷新该缓存。`TWAIT` 和 `TTEST`
只负责观察signal，不会额外维护payload缓存。

接收端观察到signal后、读取payload前，必须按照目标平台和运行时的内存规则保证payload可见。观察到
signal代表payload已经到达远端GM，但不代表此前缓存的payload旧副本已经自动刷新。

## 示例

以下发送端示例假设Host通信运行时已完成远端payload和signal地址转换，并初始化了所选后端的workspace。

### SDMA `Set`：完整发送端kernel

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void PutAndNotifySdma(__gm__ T *remoteDst,
                                       __gm__ T *localSrc,
                                       __gm__ int32_t *remoteSignal,
                                       __gm__ uint8_t *sdmaWorkspace)
{
    using GShape = Shape<1, 1, 1, 1, 1024>;
    using GStride = Stride<1024, 1024, 1024, 1024, 1>;
    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;
    using ScratchTile =
        Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    GT dstGlobalData(remoteDst);
    GT srcGlobalData(localSrc);
    comm::Signal remoteReady(remoteSignal);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
            scratchTile, sdmaWorkspace, session)) {
        return;
    }

    comm::AsyncEvent event =
        comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::SDMA>(
            dstGlobalData, srcGlobalData, remoteReady, 1,
            comm::NotifyOp::Set, session);
    if (!event.Wait(session)) {
        return;
    }
}
```

### URMA `AtomicAdd`显式peer

URMA不需要scratch tile。运行该kernel前必须初始化signal；每次加 `1` 可让接收端统计已完成的生产者数。

```cpp
template <typename T>
__global__ AICORE void PutAndCountUrma(__gm__ T *remoteDst,
                                      __gm__ T *localSrc,
                                      __gm__ int32_t *remoteCounter,
                                      __gm__ uint8_t *urmaWorkspace,
                                      uint32_t peer)
{
    using GShape = Shape<1, 1, 1, 1, 1024>;
    using GStride = Stride<1024, 1024, 1024, 1024, 1>;
    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;

    GT dstGlobalData(remoteDst);
    GT srcGlobalData(localSrc);
    comm::Signal counter(remoteCounter);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(
            urmaWorkspace, session)) {
        return;
    }

    comm::AsyncEvent event =
        comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::URMA>(
            dstGlobalData, srcGlobalData, counter, 1,
            comm::NotifyOp::AtomicAdd, session, peer);
    if (!event.Wait(session)) {
        return;
    }
}
```

### RDMA `Set`显式peer

RDMA仅支持 `Set`。本例使用256字节Vec tile，满足至少64字节的要求。

```cpp
template <typename T>
__global__ AICORE void PutAndNotifyRdma(__gm__ T *remoteDst,
                                       __gm__ T *localSrc,
                                       __gm__ int32_t *remoteSignal,
                                       __gm__ uint8_t *rdmaWorkspace,
                                       uint32_t myPe,
                                       uint32_t peer)
{
    using GShape = Shape<1, 1, 1, 1, 1024>;
    using GStride = Stride<1024, 1024, 1024, 1024, 1>;
    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;
    using ScratchTile =
        Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    GT dstGlobalData(remoteDst);
    GT srcGlobalData(localSrc);
    comm::Signal remoteReady(remoteSignal);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::RDMA>(
            scratchTile, rdmaWorkspace, myPe, session)) {
        return;
    }

    comm::AsyncEvent event =
        comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::RDMA>(
            dstGlobalData, srcGlobalData, remoteReady, 1,
            comm::NotifyOp::Set, session, peer);
    if (!event.Wait(session)) {
        return;
    }
}
```

### 接收端

接收端等待被远端更新的signal所对应的本地地址，然后按目标平台要求完成payload缓存维护，再读取payload：

```cpp
comm::Signal ready(localSignalPtr);
comm::TWAIT(ready, 1, comm::WaitCmp::EQ);

// 执行目标平台和运行时要求的payload可见性操作。
// 满足要求后再消费localPayloadPtr。
```
