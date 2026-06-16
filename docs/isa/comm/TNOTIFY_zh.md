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
- **时序约束**：
    - 向同一远端地址连续多次 `Set` 时，必须确保消费者已读取前一个值后，生产者才能写入下一个值。否则 `TWAIT(EQ)` 可能因前值被覆盖而永远阻塞。可通过反向 ack 机制保证时序（参见示例）。

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

### 连续多次通知同一地址（反向 ack 保证时序）

**错误示例**——生产者连续 Set 两次，没有等待消费者确认：

```text
生产者                        消费者
  │  TNOTIFY(flag, 1, Set)      │
  │ ──────────────────────────> │
  │  TNOTIFY(flag, 2, Set)      │   消费者还没执行 TWAIT
  │ ──────────────────────────> │   flag 被覆盖为 2
  │                              │   TWAIT(flag, 1, EQ) → 永远阻塞！
```

**正确做法**——生产者等消费者回复 ack 后，再发送下一次通知：

```text
生产者                        消费者
  │  TNOTIFY(flag, 1, Set)      │
  │ ──────────────────────────> │
  │                              │   TWAIT(flag, 1, EQ) ✓
  │                              │   TNOTIFY(ack, 1, Set)
  │  <────────────────────────  │
  │  TWAIT(ack, 1, EQ) ✓        │
  │  TNOTIFY(flag, 2, Set)      │
  │ ──────────────────────────> │
  │                              │   TWAIT(flag, 2, EQ) ✓
```

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
