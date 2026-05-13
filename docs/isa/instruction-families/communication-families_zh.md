# 通信指令族

通信指令族覆盖跨 NPU 的点对点传输、集合数据搬运、集合归约以及基于 signal 的同步。

## 指令概览

| 操作 | PTO 名称 | 说明 |
|---|---|---|
| 点对点 put | `pto.tput` | 同步远程写入 |
| 点对点 get | `pto.tget` | 同步远程读取 |
| 异步 put | `pto.tput_async` | 返回 `AsyncEvent` 的异步远程写入 |
| 异步 get | `pto.tget_async` | 返回 `AsyncEvent` 的异步远程读取 |
| Broadcast | `pto.tbroadcast` | 从源缓冲区广播到组内各 rank 目标缓冲区 |
| Scatter | `pto.tscatter` | 从源缓冲区向组内各 rank 分发 |
| Gather | `pto.tgather` | 从组内各 rank 收集到目标缓冲区 |
| Reduce | `pto.treduce` | 归约到目标缓冲区 |
| Notify | `pto.tnotify` | signal / notify 更新 |
| Test | `pto.ttest` | 非阻塞 signal 测试 |
| Wait | `pto.twait` | 阻塞 signal 等待 |

## 共享编程模型

`include/pto/comm/pto_comm_inst.hpp` 中已核实的 public wrapper 展示出若干共同模式：

- 同步的数据搬运类通信 wrapper 返回 `RecordEvent`；
- `TNOTIFY` 和 `TWAIT` 返回 `void`，`TTEST` 返回 `bool`；
- wrapper 在分发到底层实现前会先等待所有传入事件 token；
- 多个 collective 和点对点搬运操作提供显式的单暂存 tile 与 ping-pong 暂存 tile overload；
- 异步点对点操作使用 `AsyncSession`，并返回 `AsyncEvent`。

## 共享操作数

通信指令可能使用：

- `ParallelGroup` 句柄，
- 源 / 目标 `GlobalTensor` 视图，
- 显式暂存 tile，
- reduction / compare / notify 枚举，
- `RecordEvent` 与 `AsyncEvent` 同步对象。

## 共享约束

!!! warning "约束"
    - collective 操作依赖参与 rank 之间语义一致的 `ParallelGroup` 契约。
    - 缓冲区角色、元素兼容性、layout 兼容性以及暂存 tile 要求都属于具体操作约束，请以各 per-op 页面为准。
    - 异步操作要求先构建 `AsyncSession`，并通过返回的 `AsyncEvent` 显式检查完成状态。
    - CPU 仿真器可用性与后端特定传输限制依赖具体实现；必要时请查看各操作页面与后端代码。

## 不允许的情形

!!! danger "不允许的情形"
    - 把后端特定的 collective 约定当成 public wrapper 已显式校验的通用规则。
    - 在未通过相应 `AsyncEvent` API 检查完成状态前就复用异步结果。
    - 未核对具体契约就假定一个通信操作的 root / rank 规则可直接套用到另一个操作。

## 相关页面

- [通信指令参考](../comm/README_zh.md)
- [通信与运行时](../comm/communication-runtime_zh.md)
