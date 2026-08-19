# TNOTIFY

## 简介

`TNOTIFY` 更新一个32位signal，用于NPU之间的轻量级通知。它不传输payload，也不返回
`AsyncEvent`。

显式传入 `peer` 的重载用于远端通知。通信运行时根据该peer的可达能力选择直访GM、URMA或RDMA路径；
调用方不需要在接口上指定引擎。接口返回时，本次signal更新已经完成。

不带 `peer` 的原有重载保留用于兼容，但它只直接访问 `dstSignalData` 指向的GM地址，不查询peer可达
信息，也不会自动切换到URMA或RDMA；本文只说明新增的显式peer接口。

## 数学语义

`NotifyOp::Set` 时：

$$
\mathrm{signal}^{\mathrm{peer}} = \mathrm{value}
$$

`NotifyOp::AtomicAdd` 时：

$$
\mathrm{signal}^{\mathrm{peer}} \mathrel{+}= \mathrm{value}
\quad (\text{原子操作})
$$

## 汇编语法

```text
tnotify %signal_remote, %value, %peer {op = #pto.notify_op<Set>} : (!pto.memref<i32>, i32, i32)
tnotify %signal_remote, %value, %peer {op = #pto.notify_op<AtomicAdd>} : (!pto.memref<i32>, i32, i32)
```

## C++内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`。

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignalData,
                      int32_t value,
                      NotifyOp op,
                      uint32_t peer,
                      WaitEvents &... events);
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `dstSignalData` | 要更新的signal。显式peer形式中，它必须表示 `peer` 上的目标GM地址。 |
| `value` | `Set` 写入的值，或 `AtomicAdd` 使用的有符号增量。 |
| `op` | signal更新操作：`NotifyOp::Set` 或 `NotifyOp::AtomicAdd`。 |
| `peer` | 目标rank id。用于查询peer可达能力并选择通信路径，不负责把本地地址转换成远端地址。 |
| `events` | 零个或多个前置PTO流水事件。接口在更新signal前依次等待这些事件。 |

`events` 中的每个对象必须提供无参 `Wait()`。`AsyncEvent` 需要通过 `Wait(session)` 等待，不能直接作为
`events` 参数传入。

### Signal对象

`comm::Signal` 是单个 `int32_t` signal对应的GlobalTensor别名：

```cpp
using Signal = GlobalTensor<int32_t,
                            Shape<1, 1, 1, 1, 1>,
                            Stride<1, 1, 1, 1, 1>,
                            Layout::ND>;
```

构造 `Signal` 只封装调用方提供的GM地址，不分配或初始化底层内存。调用方必须提前分配signal，并按
通信协议设置初始值。

## 路由与能力

- 显式peer重载读取通信运行时初始化的peer可达信息，并选择支持当前 `NotifyOp` 的路径。具体路径属于
  实现细节，调用方不应依赖某一次调用实际选择了哪种引擎。
- `dstSignalData` 的地址和 `peer` 必须指向同一个目标。`peer` 只选择目标通信资源，不修正错误地址。

当前接口能力为：

| 目标可达路径 | `NotifyOp::Set` | `NotifyOp::AtomicAdd` | 说明 |
|---|---|---|---|
| 直访GM | 支持 | 支持 | 适用于signal地址可直接访问的本机或P2P场景。 |
| URMA | 支持 | 支持 | 仅NPU_ARCH 3510；要求通信运行时已初始化该peer的URMA资源。 |
| RDMA | 支持 | 不支持 | 仅A5；当前RDMA后端为HNS1825 RoCE。 |

SDMA数据路径可达不等于signal地址一定能由Scalar直接访问。`TNOTIFY` 不单独提交SDMA payload传输；
只有运行时确认目标signal可直访时，才使用直访GM路径。

## 操作与并发语义

- `Set` 将signal写为 `value`。多个执行流并发 `Set` 同一个signal时，不保证最终值。
- `AtomicAdd` 对signal更新本身具有原子性；多个生产者的增量不会相互覆盖。
- 不应在缺少应用层同步时，对同一个signal并发混用 `Set`、`AtomicAdd` 或普通store。
- RDMA路径不支持 `AtomicAdd`，不得对仅能通过RDMA到达的peer使用该操作。

## 完成、顺序与可见性

单次调用按以下顺序执行：

1. 等待所有 `events`。
2. 根据 `peer` 选择可用路径。
3. 更新signal，并在该更新完成后返回。

对于URMA或RDMA路径，接口内部必须等待对应远端操作完成，因此 `void` 返回不表示异步提交。

`TNOTIFY` 只覆盖本次signal更新，不会自动等待其他Session或其他队列中此前提交的异步操作。若signal用于
发布 `TPUT_ASYNC` 写入的payload，调用方必须先对该PUT返回的Event执行 `Wait(session)`，再调用
`TNOTIFY`。

接收端使用 `TWAIT` 等待signal时，`TWAIT` 会在轮询过程中刷新signal所在cache line。若接收端通过普通
load读取signal或相关payload，调用方仍需遵守目标平台和运行时的内存一致性要求。

## 约束

- `GlobalSignalData::RawDType` 必须为 `int32_t`。
- `dstSignalData.data()` 必须非空且按4字节对齐。
- `peer` 必须小于通信运行时初始化的rank总数，且该peer必须存在支持当前操作的可达路径。
- URMA和RDMA路径要求signal完整地址范围位于Host初始化阶段注册的远端内存区域内。
- 显式peer重载要求peer可达信息及对应通信资源已经在kernel启动前初始化，并在调用期间保持有效。
- RDMA仅支持 `NotifyOp::Set`。

## 示例

以下示例假设Host通信运行时已初始化peer可达信息和相应通信资源，并已提供属于目标peer的远端signal
地址。

### 显式peer Set

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

__global__ AICORE void NotifyPeerSet(__gm__ int32_t *remoteSignalPtr,
                                    uint32_t peer)
{
    comm::Signal remoteSignal(remoteSignalPtr);
    comm::TNOTIFY(remoteSignal, 1, comm::NotifyOp::Set, peer);
}
```

同一调用形式可用于直访GM、URMA或RDMA目标，路径由运行时选择。

### 显式peer AtomicAdd

```cpp
__global__ AICORE void NotifyPeerAdd(__gm__ int32_t *remoteCounterPtr,
                                    uint32_t peer)
{
    comm::Signal remoteCounter(remoteCounterPtr);

    // 仅在该peer存在支持AtomicAdd的直访GM或URMA路径时调用。
    comm::TNOTIFY(remoteCounter, 1, comm::NotifyOp::AtomicAdd, peer);
}
```

### 在异步PUT完成后发送独立通知

```cpp
auto event = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(
    remoteDst, localSrc, session, peer);

if (event.Wait(session)) {
    comm::TNOTIFY(remoteSignal, 1, comm::NotifyOp::Set, peer);
}
```

必须显式等待PUT的Event；仅按程序顺序紧接着调用 `TNOTIFY`，不能保证异步payload先完成。
