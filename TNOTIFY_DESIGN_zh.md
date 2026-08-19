# `TNOTIFY` 显式 peer 重载设计评审说明

> 本文用于配合 `docs/isa/comm/TNOTIFY_zh.md` 进行接口评审，说明新增 peer 重载的必要性、路由与后端
> 实现边界，不作为用户接口规范。用户可见的参数、约束和支持范围以 ISA 接口文档为准。

## 1. 设计结论

新增接口保持 `TNOTIFY` 的同步通知语义，只增加目标 peer：

```cpp
comm::TNOTIFY(dstSignalData, value, notifyOp, peer, waitEvents...);
```

接口需要保证：

1. 等待前置 `waitEvents` 后再更新 signal；
2. 根据 `peer` 的已就绪可达能力选择直访 GM、URMA 或 RDMA 路径；
3. 接口返回时，本次 signal 更新已经完成；
4. `Set` 写入 signal，`AtomicAdd` 原子累加 signal；
5. `dstSignalData` 已经表示目标 peer 的远端地址，`peer` 只选择通信资源，不负责地址换算。

不带 `peer` 的原有重载继续用于已知地址可直访的兼容场景，不自动选择 URMA/RDMA。

## 2. 为什么需要新增重载

现有接口只接收 signal 地址：

```cpp
comm::TNOTIFY(signal, value, op);
```

当前实现可以直接对该地址执行 Scalar store 或 `st_atomic`，但它不知道目标 rank，也无法取得远端路径需要的
QP、SQ/CQ、rkey/lkey 和已注册内存信息。对于只能通过 URMA/RDMA 到达的 peer，一个远端地址不足以构造
正确的通知操作。

显式 `peer` 使实现能够查询通信运行时准备好的可达信息和每个 peer 的资源。接口不增加 `DmaEngine`
模板参数，因为用户表达的是“通知 peer N”；实际路径应随部署拓扑变化，而不是由 kernel 固定。

该重载只完成独立通知，不负责排序此前的异步数据传输：

```cpp
auto event = comm::TPUT_ASYNC<engine>(dst, src, session, peer);
event.Wait(session);
comm::TNOTIFY(signal, value, op, peer);
```

省略 `Wait(session)` 时，独立 `TNOTIFY` 不保证 signal 晚于另一个 Session 或队列中的 payload。需要无中间
等待的 payload-before-signal 语义时，应使用 `TPUT_ASYNC_NOTIFY`。

## 3. 公共语义与支持状态

| 可达路径 | `Set` | `AtomicAdd` | 当前评审口径 |
|---|---|---|---|
| 直访 GM | 支持 | 支持 | 复用当前 Scalar signal 实现 |
| URMA | 支持 | 支持 | 需要同步等待对应 CQ 完成 |
| RDMA | 支持 | 设计支持 | 当前仅开放 `Set`；`AtomicAdd` 等待固件支持 |
| 仅 SDMA 可达且 GM 不可直访 | 不支持 | 不支持 | SDMA 可达不能推导 Scalar 可达 |

signal 固定为 4-byte 对齐的 `int32_t`。`AtomicAdd` 只保证 signal 更新本身的原子性；并发写同一 signal 的
payload、协议计数和生命周期仍由上层保证。并发 `Set` 不承诺最终值，也不应无同步地混用普通 store、
`Set` 和 `AtomicAdd`。

## 4. peer 可达信息与路由

建议为每个 peer 保存一项内部能力位图，至少区分：

```text
DirectGm  // Scalar/MTE 可直接访问该 peer 的 GM 地址
Sdma      // 存在 SDMA 数据通路
Urma      // URMA 资源已初始化并可用
Rdma      // RDMA/RoCE 资源已初始化并可用
```

`DirectGm` 必须独立于 `Sdma`。SDMA 描述的是 DMA 搬运能力，当前直接通知使用的是 Scalar GM store 或
Scalar atomic；仅有 SDMA 位时不能直接执行现有 `TNOTIFY`。

位图应表示“路径已经初始化成功”，而不只是硬件理论可达。建议使用确定性的选择顺序：

```text
DirectGm 支持当前 op -> URMA 支持当前 op -> RDMA 支持当前 op -> 明确报错
```

不能把不支持的 `AtomicAdd` 静默降级为普通 Set 或非原子的 read-modify-write。

## 5. 隐藏运行时上下文

peer-only 签名没有 `AsyncSession`，因此设备侧必须能发现一个 PTO 内部通信上下文。它至少需要承载：

- rank 数量和 peer 能力位图；
- URMA/RDMA workspace 及每个 peer 的队列、QP 和 MR 信息；
- 按执行流或 QP 隔离的本地 signal staging、FAA operand/result slot；
- UB scratch、同步事件和队列所有权策略；
- 用于拒绝未初始化或版本不匹配状态的 magic/version。

这些结构属于运行时与后端实现，不应冻结成公共 ISA ABI。Host 必须在 kernel 启动前完成初始化，并保证
上下文及资源在调用期间有效。

## 6. 各路径实现方案

### 6.1 直访 GM

复用当前 `TNOTIFY_IMPL`：

```text
Set:       cache maintenance -> Scalar store -> DSB/barrier
AtomicAdd: cache maintenance -> int32 st_atomic -> DSB/barrier
```

只有运行时确认 `DirectGm` 时才可选择该路径。

### 6.2 SDMA

第一版不为 standalone `TNOTIFY` 单独提交 SDMA 描述符。4-byte Set 需要稳定的本地 GM staging、精确小包
写和同步完成验证；普通 SDMA copy 也不能实现原子加。这些能力不能从“SDMA 可达”自然推导。

若后续必须支持 SDMA-only peer，应把 4-byte Set 作为独立能力验证；`AtomicAdd` 仍必须依赖真实远端
atomic，不能用软件读改写模拟。

### 6.3 URMA

```text
Set:       value -> 已注册 staging -> 4-byte WRITE -> poll CQ -> return
AtomicAdd: operand/result slot -> FAA -> poll CQ -> return
```

即使 PTO 不向用户返回 FAA 旧值，后端仍需提供硬件要求的 result buffer。staging、operand 和 result slot
必须保持到 CQ 完成，并按执行流/QP 隔离，避免多个核在设备读取前覆盖同一 slot。

### 6.4 RDMA

`Set` 复用 HNS1825 的 4-byte RDMA WRITE：从内部上下文选择 peer 的 QP、rkey 和 CQ，本地 value 来自已
注册且隔离的 staging slot，CQ 成功后才返回。

`AtomicAdd` 设计使用 masked FAA：以 signal 所在的 8-byte 对齐字为 atomic 目标，根据地址选择低或高
32-bit lane，并通过 mask/field-boundary 把修改限制在该 lane；旧值写入内部 result sink。对应 8-byte
范围必须完整位于远端已注册内存中。

当前固件阻塞所需 atomic 能力，因此用户接口仍标为不支持。固件能力具备后，需验证 mask 位序、大小端、
正负增量、相邻 4 bytes、旧值 sink、CQE 和远端结果后再开放。

standalone `TNOTIFY` 没有同一次调用中的前序 payload WQE；它不能靠 RDMA fence 排序另一 Session/QP
中的异步 PUT。此场景仍需先等待 PUT Event。

## 7. 完成、错误与可见性边界

- 接口返回类型为 `void`，因此 URMA/RDMA 路径不能只提交 WQE 后立即返回，必须等待本次远端更新完成。
- peer 越界、上下文未初始化、没有兼容路径、MR 越界或 CQE 错误，应走 PTO 统一的确定性错误上报。
- 接口只完成 signal 更新，不隐式等待其他 Session、队列、engine 或发送者的操作。
- `TWAIT` 会在轮询 signal 时维护 signal cache line；普通 load 以及 signal 关联 payload 的 cache
  maintenance 仍由调用方按平台内存一致性规则保证。
- 同一 QP/SQ 和 staging slot 的并发所有权必须由运行时隔离或串行化。

## 8. 评审建议重点

| 关注点 | 建议评审结论 |
|---|---|
| 是否只需增加一个参数 | 否；peer-only 接口还依赖可被 kernel 发现、生命周期明确的内部通信上下文。 |
| 为什么不用引擎模板参数 | 路由取决于目标 peer 和部署拓扑，用户不应固定硬件路径。 |
| SDMA 位是否等于可直访 | 不等于；必须单独记录 `DirectGm`。 |
| `void` 返回的含义 | 本次 signal 更新完成后才返回，远端 WQE 路径需要内部等待 CQ。 |
| 与异步 PUT 的顺序 | 不自动排序；用户先 Wait，或改用 `TPUT_ASYNC_NOTIFY`。 |
| 临时资源 | value/operand/result 必须位于已注册且按并发执行流隔离的稳定 slot。 |
| RDMA AtomicAdd 状态 | masked FAA 设计支持，但当前固件阻塞，验证完成前不作为用户可用能力。 |

## 9. 建议验证项

1. peer 左值/右值、零个或多个前置 Event，以及旧重载兼容性；
2. peer 越界、空位图、未初始化资源和不支持操作的确定性失败；
3. `DirectGm` Set/Add，以及不把 SDMA-only peer 错分派到 Scalar 路径；
4. URMA 4-byte Set、32-bit FAA、CQE、slot 回绕和多执行流隔离；
5. RDMA 4-byte Set 的 MR、QP/CQ 选择、远端结果和运行后 NIC 状态；
6. 固件支持后验证 RDMA masked FAA 的高/低 lane、正负增量、相邻字段和旧值 sink；
7. `TPUT_ASYNC -> Wait -> TNOTIFY -> TWAIT -> 读取 payload` 的完整组合流程。
