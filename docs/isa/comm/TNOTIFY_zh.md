# pto.tnotify

## 概要

`pto.tnotify` 使用指定的 notify 操作，把一个整数通知值写入到信号位置。

## 语义

已核实的 public wrapper 接受：

- 目标 signal 对象，
- `int32_t` 通知值，
- `NotifyOp`，
- 以及可选的等待事件 token。

从概念上看：

- `NotifyOp::Set` 写入给定值；
- 其他原子风格的 `NotifyOp` 变体执行该枚举对应的后端更新语义。

## 汇编语法

```text
pto.tnotify %signal_remote, %value {op = #pto.notify_op<Set>} : (!pto.memref<i32>, i32)
```

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`。

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignalData, int32_t value, NotifyOp op, WaitEvents &... events);
```

## 约束

!!! warning "约束"
    - public wrapper 将 value 类型固定为 `int32_t`。
    - 信号存储必须与所选后端实现兼容。
    - wrapper 在发出通知前会先等待所有传入事件 token。

## 与 `TWAIT` / `TTEST` 的关系

`TNOTIFY` 通常与 `TWAIT` 或 `TTEST` 配合，用于基于标志的同步与轮询。

## 示例

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void notify_one(auto &signal) {
    comm::TNOTIFY(signal, 1, comm::NotifyOp::Set);
}
```
