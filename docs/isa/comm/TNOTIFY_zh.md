# TNOTIFY

## 简介

向远程 NPU 发送标志通知。用于 NPU 之间的轻量级同步，无需传输大量数据。

## 数学表示

对于 `NotifyOp::Set`：

$$ \mathrm{signal}^{\mathrm{remote}} = \mathrm{value} $$

对于 `NotifyOp::AtomicAdd`：

$$ \mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{value} \quad (\text{原子操作}) $$

## 汇编语法

PTO-AS 形式：见 [PTO-AS 规范](../../assembly/PTO-AS.md)。

```text
tnotify %signal_remote, %value {op = #pto.notify_op<Set>} : (!pto.memref<i32>, i32)
tnotify %signal_remote, %value {op = #pto.notify_op<AtomicAdd>} : (!pto.memref<i32>, i32)
```

## C++ Intrinsic

声明在 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignalData, int32_t value, NotifyOp op, WaitEvents&... events);
```

## 约束

- **类型约束**：
  - `GlobalSignalData::DType` 必须为 `int32_t`（32 位信号）。
- **内存约束**：
  - `dstSignalData` 必须指向远程地址（目标 NPU 上）。
  - `dstSignalData` 应 4 字节对齐。
- **操作语义**：
  - `NotifyOp::Set`：直接存储到远程内存。
  - `NotifyOp::AtomicAdd`：使用 `st_atomic` 指令的硬件原子加。

## 示例

### 基本设置通知

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void notify_set(__gm__ int32_t* remote_signal) {
    comm::Signal sig(remote_signal);
    
    // 将远程信号设置为 1
    comm::TNOTIFY(sig, 1, comm::NotifyOp::Set);
}
```

### 原子计数器自增

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void atomic_increment(__gm__ int32_t* remote_counter) {
    comm::Signal counter(remote_counter);
    
    // 对远程计数器原子加 1
    comm::TNOTIFY(counter, 1, comm::NotifyOp::AtomicAdd);
}
```

### 生产者-消费者模式

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// 生产者：数据就绪时发送通知
void producer(__gm__ int32_t* remote_flag) {
    // ... 生产数据 ...
    
    comm::Signal flag(remote_flag);
    comm::TNOTIFY(flag, 1, comm::NotifyOp::Set);
}

// 消费者：等待数据
void consumer(__gm__ int32_t* local_flag) {
    comm::Signal flag(local_flag);
    comm::TWAIT(flag, 1, comm::WaitCmp::EQ);
    
    // ... 消费数据 ...
}
```
