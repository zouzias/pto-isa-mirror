# TNOTIFY

## 简介

向远端 NPU 发送标志通知。用于 NPU 之间的轻量级同步，无需传输大量数据。

## 数学语义

`NotifyOp::Set` 时：

$$\mathrm{signal}^{\mathrm{remote}} = \mathrm{value}$$

`NotifyOp::AtomicAdd` 时：

$$\mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{value} \quad (\text{原子操作})$$

## 汇编语法

```text
tnotify %signal_remote, %value {op = #pto.notify_op<Set>} : (!pto.memref<i32>, i32)
tnotify %signal_remote, %value {op = #pto.notify_op<AtomicAdd>} : (!pto.memref<i32>, i32)
```

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignalData, int32_t value, NotifyOp op, WaitEvents&... events);
```

## 约束

- **类型约束**：
    - `GlobalSignalData::DType` 必须为 `int32_t`（32 位信号）。
- **内存约束**：
    - `dstSignalData` 必须指向远端地址（目标 NPU）。
    - `dstSignalData` 应 4 字节对齐。
- **操作语义**：
    - `NotifyOp::Set`：直接存储到远端内存。
    - `NotifyOp::AtomicAdd`：使用 `st_atomic` 指令执行硬件原子加。

## 示例

### 基础 Set 通知

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void notify_set(__gm__ int32_t* remote_signal) {
    comm::Signal sig(remote_signal);

    // 将远端信号置为 1
    comm::TNOTIFY(sig, 1, comm::NotifyOp::Set);
}
```

### 原子计数器自增

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void atomic_increment(__gm__ int32_t* remote_counter) {
    comm::Signal counter(remote_counter);

    // 对远端计数器原子加 1
    comm::TNOTIFY(counter, 1, comm::NotifyOp::AtomicAdd);
}
```

### 生产者-消费者模式

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// 生产者：数据就绪后发送通知
void producer(__gm__ int32_t* remote_flag) {
    // ... 生产数据 ...

    comm::Signal flag(remote_flag);
    comm::TNOTIFY(flag, 1, comm::NotifyOp::Set);
}

// 消费者：等待数据就绪
void consumer(__gm__ int32_t* local_flag) {
    comm::Signal flag(local_flag);
    comm::TWAIT(flag, 1, comm::WaitCmp::EQ);

    // ... 消费数据 ...
}
```

### 连续多次通知同一地址（需保证时序）

向**同一个** 远端地址连续多次发送标志通知时，必须保证消费者**先读到前一个值**，
生产者再用下一个值覆盖它。`Set` 是直接覆盖写、`TWAIT(EQ)` 是精确匹配，如果生产者
抢先把下一个值写进去，消费者的 `TWAIT` 就永远等不到前一个值而挂死。

做法：消费者收到标志通知后需要回复ack，生产者在第一次发送标志通知之后等待消费者回复的ack，收到之后再发送下一次标志通知：

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// 生产者：向同一地址发两次，用 ack 卡住时序
void producer(__gm__ int32_t* remote_flag, __gm__ int32_t* local_ack) {
    comm::Signal flag(remote_flag);
    comm::Signal ack(local_ack);

    comm::TNOTIFY(flag, 1, comm::NotifyOp::Set);   // 第一轮
    comm::TWAIT(ack, 1, comm::WaitCmp::EQ);        // 等消费者读完第一轮
    comm::TNOTIFY(flag, 2, comm::NotifyOp::Set);   // 第二轮（此时才安全）
}

// 消费者：读完每个值，第一轮后回 ack
void consumer(__gm__ int32_t* local_flag, __gm__ int32_t* remote_ack) {
    comm::Signal flag(local_flag);
    comm::Signal ack(remote_ack);

    comm::TWAIT(flag, 1, comm::WaitCmp::EQ);       // 第一轮
    comm::TNOTIFY(ack, 1, comm::NotifyOp::Set);    // 告诉生产者第一轮已消费
    comm::TWAIT(flag, 2, comm::WaitCmp::EQ);       // 第二轮
}
```
