# 通信与运行时

通信指令覆盖跨 NPU 的点对点传输、集合数据搬运、集合归约以及基于 signal 的同步。

## 操作

| 操作 | 说明 | IR spelling | C++ spelling |
|---|---|---|---|
| [TBROADCAST](./TBROADCAST_zh.md) | 从源缓冲区广播到组内各 rank 目标缓冲区 | `pto.tbroadcast` | `TBROADCAST` |
| [TGET](./TGET_zh.md) | 同步远程读取 | `pto.tget` | `TGET` |
| [TGET_ASYNC](./TGET_ASYNC_zh.md) | 异步远程读取 | `pto.tget_async` | `TGET_ASYNC` |
| [TNOTIFY](./TNOTIFY_zh.md) | signal / notify 更新 | `pto.tnotify` | `TNOTIFY` |
| [TPUT](./TPUT_zh.md) | 同步远程写入 | `pto.tput` | `TPUT` |
| [TPUT_ASYNC](./TPUT_ASYNC_zh.md) | 异步远程写入 | `pto.tput_async` | `TPUT_ASYNC` |
| [TREDUCE](./TREDUCE_zh.md) | 集合归约 | `pto.treduce` | `TREDUCE` |
| [TSCATTER](./TSCATTER_zh.md) | 从源缓冲区向组内各 rank 分发 | `pto.tscatter` | `TSCATTER` |
| [TGATHER](./TGATHER_zh.md) | 从组内各 rank 收集到目标缓冲区 | `pto.tgather` | `TGATHER` |
| [TTEST](./TTEST_zh.md) | 非阻塞 signal 测试 | `pto.ttest` | `TTEST` |
| [TWAIT](./TWAIT_zh.md) | 阻塞 signal 等待 | `pto.twait` | `TWAIT` |

## 编程模型说明

`include/pto/comm/pto_comm_inst.hpp` 中已核实的 public wrapper 展示出若干共同模式：

- 同步 comm wrapper 对点对点与 collective 数据搬运通常返回 `RecordEvent`；
- signal 风格的 `TNOTIFY`、`TWAIT` 返回 `void`，`TTEST` 返回 `bool`；
- wrapper 在分发到底层实现前会先等待所有传入事件 token；
- 多个搬运类 collective 提供显式的单 tile 与 ping-pong tile overload；
- 异步点对点操作基于 session，并返回 `AsyncEvent`。

## 异步运行时说明

异步传输操作使用 `AsyncSession` 与 `AsyncEvent` 作为 public 同步模型。

已核实的 public helper 包括：

- `BuildAsyncSession(...)`
- `AsyncEvent::Wait(const AsyncSession &session)`
- `AsyncEvent::Test(const AsyncSession &session)`

session 保存所选 DMA engine 以及 engine-specific 的执行/事件上下文。

## 本页范围

本页总结的是 public wrapper surface。更细的后端行为、合法性约束、root 角色约定以及传输特定限制，请参见各操作页面与具体后端实现代码。

## 相关页面

- [TPUT](./TPUT_zh.md)
- [TGET](./TGET_zh.md)
- [TPUT_ASYNC](./TPUT_ASYNC_zh.md)
- [TGET_ASYNC](./TGET_ASYNC_zh.md)
- [TNOTIFY](./TNOTIFY_zh.md)
- [TWAIT](./TWAIT_zh.md)
- [TTEST](./TTEST_zh.md)
- [TBROADCAST](./TBROADCAST_zh.md)
- [TGATHER](./TGATHER_zh.md)
- [TSCATTER](./TSCATTER_zh.md)
- [TREDUCE](./TREDUCE_zh.md)
