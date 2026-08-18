# TPUT_ASYNC_NOTIFY

## 简介

`TPUT_ASYNC_NOTIFY` 是带通知的异步远程写原语。它先将非空payload从本地GM传输到远端GM，再更新一个
远端32位signal。该接口沿用 `TPUT_ASYNC` 的 `AsyncEvent`，不新增事件类型。

数据流：

```text
srcGlobalData（本地 GM） -> DMA引擎 -> dstGlobalData（远端 GM）
                                           |
                                      payload完成后
                                           v
                                  dstSignalData（远端 GM）
```

如果只需要更新signal，不需要传输payload，应使用 `TNOTIFY`。

## 模板参数与后端支持

- `engine` 在编译期选择DMA后端：

| 引擎 | 平台或后端限制 | 支持的 `NotifyOp` |
|---|---|---|
| `DmaEngine::SDMA`（默认） | `TPUT_ASYNC` 支持的平台 | `Set`、`AtomicAdd` |
| `DmaEngine::URMA` | 仅 Ascend 950PR/Ascend 950DT、NPU_ARCH 3510；要求CANN Toolkit >= 9.1.0 | `Set`、`AtomicAdd` |
| `DmaEngine::RDMA` | 仅 Ascend 950PR/Ascend 950DT、NPU_ARCH 3510；HNS1825网卡平台 | 仅 `Set`；不支持 `AtomicAdd` |

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

A5上提供与 `TPUT_ASYNC` 一致的显式目标peer重载：

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

可选的 `events` 参数包沿用PTO现有的依赖事件约定。发起payload传输前，接口会等待所有传入的依赖事件；
不传依赖事件也是合法用法。

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
| `events` | 零个或多个前置PTO流水依赖事件。 |

返回值为 `AsyncEvent`，其完成范围同时覆盖payload传输和signal更新。

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

上述顺序只约束同一次调用中的payload与signal更新，不定义不同Session或不同执行流独立发起的操作之间的顺序。

## AsyncSession构建

`TPUT_ASYNC_NOTIFY` 沿用 `TPUT_ASYNC` 的 `AsyncSession`、workspace和 `AsyncEvent` 约定。各引擎的构建参数、
workspace要求、生命周期及Session所有权详见 [TPUT_ASYNC](TPUT_ASYNC_zh.md)。

A5上与多个peer通信时，应构建一个不绑定peer的URMA或RDMA Session，并通过显式 `peer` 重载传入目标
rank。为兼容已有调用方式，仍保留绑定peer的Session用法。

## 约束

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`。
- `GlobalSrcData::layout == GlobalDstData::layout`。
- 源和目的payload tensor必须是扁平、连续的逻辑一维tensor。
- payload大小必须大于0；仅更新signal时应使用 `TNOTIFY`。
- `dstSignalData` 必须表示远端GM中恰好一个 `int32_t`，并满足4字节对齐。
- payload目的地址范围与 `dstSignalData` 不得重叠。
- 对于URMA和RDMA，payload目的地址和signal必须属于Session或显式 `peer` 参数选择的同一个远端peer。
- 对于URMA和RDMA，完整的本地payload、远端payload和远端signal地址范围都必须位于Host初始化阶段
  注册的内存区域内。
- 单次RDMA payload不能超过 `0x7fffffff` 字节。
- RDMA仅支持 `NotifyOp::Set`，不得使用 `NotifyOp::AtomicAdd`。
- Session的引擎必须与模板参数 `engine` 一致。Session和workspace的生命周期必须覆盖所有相关Event的完成阶段。

## 完成与顺序语义

返回的Event沿用 `TPUT_ASYNC` 的quiet语义：

- `event.Wait(session)` 阻塞等待完成。
- `event.Test(session)` 非阻塞检测完成状态。
- 成功完成同时覆盖payload传输及其后的signal更新。
- RDMA操作涉及不同peer时，必须分别等待每个peer的最后一个Event。

接收端可观察到signal更新时，对应payload已经到达远端GM。该保证只作用于同一次
`TPUT_ASYNC_NOTIFY` 调用。

## 接收端payload可见性

该指令不会使接收端可能缓存的 `dstGlobalData` 旧副本失效，也不会主动刷新该缓存。`TWAIT` 和 `TTEST`
只负责观察signal，不会额外维护payload缓存。

接收端观察到signal后、读取payload前，必须按照目标平台和运行时的内存规则保证payload可见。观察到
signal代表payload已经到达远端GM，但不代表此前缓存的payload旧副本已经自动刷新。

## 示例

### Set通知

下面的Session和payload tensor与 `TPUT_ASYNC` 使用相同方式构建：

```cpp
comm::Signal remoteReady(remoteSignalPtr);

auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::SDMA>(
    dstGlobalData, srcGlobalData, remoteReady, 1, comm::NotifyOp::Set, session);
(void)event.Wait(session);
```

接收端可以先等待signal，再消费payload：

```cpp
comm::Signal ready(localSignalPtr);
comm::TWAIT(ready, 1, comm::WaitCmp::EQ);

// 读取localPayloadPtr前，先按目标平台要求保证payload可见。
// 满足该要求后再消费payload。
```

### RDMA显式peer

RDMA仅支持 `Set`：

```cpp
comm::Signal remoteReady(remoteSignalPtr);

auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::RDMA>(
    dstGlobalData, srcGlobalData, remoteReady, 1, comm::NotifyOp::Set, session, peer);
(void)event.Wait(session);
```
