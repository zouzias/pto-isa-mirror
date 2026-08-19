# `TPUT_ASYNC_NOTIFY` 设计评审说明

> 本文用于配合 `docs/isa/comm/TPUT_ASYNC_NOTIFY_zh.md` 进行接口评审，说明接口必要性、后端实现思路与
> 评审边界，不作为用户接口规范。用户可见的参数、约束和支持范围以 ISA 接口文档为准。

## 1. 设计结论

`TPUT_ASYNC_NOTIFY` 将“向指定 peer 写入 payload”和“随后更新该 peer 的 signal”作为一个异步提交：

```cpp
auto event = comm::TPUT_ASYNC_NOTIFY<engine>(
    dstGlobalData, srcGlobalData, dstSignalData,
    signalValue, notifyOp, session, peer, waitEvents...);
```

接口需要保证：

1. 前置 `waitEvents` 完成后才开始本次操作；
2. 对应 payload 先于 signal 更新到达远端；
3. 返回的 `AsyncEvent` 同时覆盖 payload 和 signal；
4. `Set` 写入 signal，`AtomicAdd` 原子累加 signal；
5. `peer` 明确指定目标 rank；SDMA 的地址不依赖 `peer`，因此忽略该参数。

该操作提供的是“有序的远端数据发布”，不是 payload 与 signal 的事务。若 signal 阶段失败，已经完成的
payload 不会回滚。

## 2. 为什么需要新增接口

### 2.1 现有接口不能无等待地安全组合

基于现有公开接口，用户要保证顺序，只能执行：

```cpp
auto putEvent = comm::TPUT_ASYNC<engine>(dst, src, session, peer);
putEvent.Wait(session);
comm::TNOTIFY(signal, value, op);
```

URMA/RDMA 需要等待 CQ，SDMA 需要等待完成标志。若省略中间等待，`TNOTIFY` 与异步 DMA 不一定处于
同一个引擎和有序队列，signal 可能先于 payload 对接收端可见。

因此，用户虽然能通过“等待后再通知”实现相同最终结果，却无法用现有接口表达“无需中间等待、同时保证
payload-before-signal”的操作。

### 2.2 接口带来的主要价值

- **明确正确性语义**：接收端观察到 signal 后，才可以把对应 payload 视为已经到达远端 GM。
- **保持异步流水**：发送端不必在 payload 与 signal 之间轮询完成状态，可以继续计算或提交后续工作。
- **缩短通知延迟**：后端可在 payload 满足顺序条件后直接执行 signal，省去用户侧完成检测与再次发起操作的
  间隔。
- **开放后端优化空间**：后端可利用同一 SQ、批量发布、doorbell 合并或原生 atomic WQE；两个独立接口无法
  被库安全地自动识别为一组操作。
- **统一完成管理**：一个 Event 表示 payload 与 signal 的整体完成，简化 buffer、session 和临时资源的
  生命周期管理。
- **支持常见通信协议**：可直接实现 mailbox、ring buffer、ping-pong slot，以及多个生产者通过
  `AtomicAdd` 汇聚完成计数的协议。

OpenSHMEM/NVSHMEM 的非阻塞 put-with-signal 也采用相同的接口形态和“数据先于对应 signal”的基本语义，
说明该组合原语具有通用性，而不是特定后端的临时接口。

## 3. 公共语义与支持状态

| 引擎 | `Set` | `AtomicAdd` | 主要顺序域 | 当前评审口径 |
|---|---|---|---|---|
| SDMA | 支持 | 支持 | 同一 SDMA queue | 纳入接口范围 |
| URMA | 支持 | 支持 | 同一 peer/QP/SQ | 纳入接口范围 |
| RDMA | 支持 | 设计支持 | 同一 peer/QP/SQ，通知 WQE 设置 `fence` | 当前仅开放 `Set`；`AtomicAdd` 等待固件支持 |

公共 signal 固定为 4-byte 对齐的 `int32_t`。`AtomicAdd` 的原子性只覆盖 signal 更新，不会使互相重叠的
payload 写入变得安全。多个生产者共享一个计数 signal 时，其 payload 目标范围仍应互不冲突或由上层协议
保护。

## 4. 后端实现方案

### 4.1 SDMA

SDMA 将 payload、signal 与完成标志编排在同一有序 queue：

```text
payload SQE
  -> signal SQE（Set: 4-byte copy；AtomicAdd: int32 atomic-add）
  -> postDone SQE
  -> AsyncEvent
```

参考实现先发布 payload，使数据传输尽早开始，再构造并发布 signal 与 postDone SQE。由于 signal SQE
排在同一 queue 的 payload SQE 之后，postDone 又排在 signal 之后，返回 Event 可以覆盖整个组合操作。

异步硬件读取的 `signalValue` 不能引用短生命周期栈变量，需要放入 workspace 中可跟踪复用的稳定 slot。
A5 上若使用同步 MTE fallback，则按“完成 payload、执行 pipeline barrier、更新 signal”的顺序实现相同
语义；该路径不主张异步性能收益。

### 4.2 URMA

URMA 在指定 peer 对应的同一 SQ 中连续构造 payload WQE 和 signal WQE：

```text
Set:       payload WRITE -> signal WRITE
AtomicAdd: payload WRITE -> signal FAA
                         -> 一次 PublishHead/doorbell
                         -> AsyncEvent
```

参考实现中，`Set` 合计占用 2 BB / 2 CQE；`AtomicAdd` 的 FAA 占用 2 BB，因此组合操作合计为
3 BB / 2 CQE。优化点不是减少为一个 CQE，而是无需在 payload 与 signal 之间轮询 CQ，并可一次性发布
已经构造好的 WQE。

SET 的本地 signal value、FAA 的 operand/result sink 都必须在硬件完成前保持有效，因此使用可注册、可按
完成水位复用的 resource slot。Event 同时记录最终 BB 和 CQE 目标，避免把“WQE 占用 BB 数”错误地等同
于“产生 CQE 数”。

### 4.3 RDMA

RDMA 方案在 `peer` 对应的同一 QP/SQ 上构造：

```text
payload RDMA WRITE
  -> fenced signal WQE
       Set:       4-byte RDMA WRITE
       AtomicAdd: NIC masked FAA WQE
  -> 发布 SQ head / ring doorbell
  -> 返回指向最终完成水位的 AsyncEvent
```

顺序保证包含两个条件：

1. payload 与 signal 必须提交到同一 peer、同一 QP、同一 SQ；
2. signal WQE 设置 `fence`，要求其等待前序 payload WQE 满足完成顺序后再执行。

因此，不能把 payload 与 signal 分发到不同 QP 后仅依赖提交先后。`fence` 设置在后一个 signal WQE 上，
它约束的是“signal 不越过前序 payload”，不是等待 signal 自身完成。

`AtomicAdd` 在接口和队列编排上已经纳入 RDMA 设计，使用 masked FAA 实现 `int32_t` 原子加：以 signal
所在的 8-byte 对齐字为 atomic 目标，根据 signal 地址选择低或高 32-bit lane，并通过 mask/field-boundary
配置把加法和进位限制在该 lane，避免修改相邻 4 bytes。对应 8-byte 范围必须完整位于远端已注册内存中，
FAA 返回的旧值写入可按完成水位复用的本地 sink，但 PTO 接口不返回该值。

当前固件阻塞了所需 atomic 能力，所以用户接口现阶段仍标为不支持。固件能力具备后，还需完成 mask 位序、
大小端、正负增量、WQE `fence`、CQE、旧值 sink 和远端结果的硬件验证，才能开放该能力。

## 5. 完成、错误与可见性边界

- Event 完成表示本次 payload 和 signal 都已达到后端定义的完成点；等待最终 Event 不需要在中间等待
  payload Event。
- Event 必须携带或可推导目标 peer 与最终队列水位，使 `Wait/Test` 能选择正确的 CQ 或完成标志。
- 提交路径应同时检查 payload WQE、signal WQE、CQE 和临时 resource slot 的容量，不能提交一半后才发现
  后半段没有资源。
- 接口不是 all-or-nothing 事务。提交或完成失败时，发送端仍需检查 Event 错误；接收端 signal 不能代替
  transport 错误处理。
- signal 可见只证明与它绑定的 payload 已到达远端 GM，不自动排序其他 Session、QP、engine 或其他发送者
  的操作。
- 接收端已有的 payload cache 副本是否需要失效或刷新，仍由调用方按照平台内存一致性规则保证；接口只负责
  传输与 signal 的先后关系。

## 6. 评审建议重点

| 关注点 | 建议评审结论 |
|---|---|
| 是否只是语法糖 | 不是；现有接口只有在中间等待后才安全，无法表达非阻塞的有序发布。 |
| signal 的含义 | 观察到 signal 表示本次对应 payload 已到达，不代表其他传输或缓存副本已经同步。 |
| 三后端是否同构 | 只要求公共语义一致；SDMA SQE、URMA WQE 和 RDMA WQE/fence 的内部实现可以不同。 |
| RDMA 是否只靠 `fence` | 不是；同 peer/QP/SQ 是前提，`fence` 用于约束后一个 signal WQE 不越过前序 payload。 |
| Event 覆盖范围 | 必须覆盖 payload 与 signal，且能定位实际操作使用的 peer/完成队列。 |
| AtomicAdd 并发 | 只保证 signal 更新原子；payload 冲突、计数阈值和 signal 复用由上层协议负责。 |
| RDMA AtomicAdd 状态 | 设计支持但当前固件阻塞；在固件与硬件验证完成前不作为用户可用能力。 |
| 性能收益 | 设计允许消除强制中间等待和利用后端批量提交；具体收益应通过基准测试量化。 |

## 7. 建议验证项

1. 各已支持引擎分别验证 `Set`、`AtomicAdd` 的 payload、signal 与 Event 完成语义；
2. 接收端在观察到 signal 后立即读取 payload，覆盖不同长度、边界和连续多次提交；
3. 多个 outstanding 操作只等待最后 Event，验证队列水位、slot 回绕和背压；
4. 多生产者写入不重叠 payload，并通过 `AtomicAdd` 汇聚完成计数；
5. RDMA 检查 payload/signal WQE 位于同一 SQ、signal WQE 的 `fence` 位和最终 CQE 水位；
6. 固件支持后，单独验证 RDMA masked FAA 的高/低 32-bit lane、正负增量、旧值 sink、相邻 4 bytes、
   远端结果与运行后 NIC 状态；
7. 性能对比 `TPUT_ASYNC_NOTIFY` 与 `TPUT_ASYNC + Wait/Test + TNOTIFY`，分别观察发送端等待时间、
   CQ/完成标志轮询次数、doorbell 次数和端到端通知延迟。
