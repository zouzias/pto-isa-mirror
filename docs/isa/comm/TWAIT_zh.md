# TWAIT

## 简介

阻塞等待直到信号满足比较条件。与 `TNOTIFY` 配合用于基于标志的同步。

支持单信号或多维信号张量（最高 5-D，shape 由 GlobalTensor 派生）。

## 数学表示

等待（自旋）直到以下条件满足：

单信号：

$$ \mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue} $$

信号张量（所有元素必须满足）：

$$ \forall d_0, d_1, d_2, d_3, d_4: \mathrm{signal}_{d_0, d_1, d_2, d_3, d_4} \;\mathtt{cmp}\; \mathrm{cmpValue} $$

其中 `cmp` ∈ {`EQ`, `NE`, `GT`, `GE`, `LT`, `LE`}

## 汇编语法

PTO-AS 形式：见 [PTO-AS 规范](../../assembly/PTO-AS.md)。

```text
twait %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32)
twait %signal_matrix, %cmp_value {cmp = #pto.cmp<GE>} : (!pto.memref<i32, MxN>, i32)
```

## C++ Intrinsic

声明在 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TWAIT(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events);
```

## 约束

- **类型约束**：
  - `GlobalSignalData::DType` 必须为 `int32_t`（32 位信号）。
- **内存约束**：
  - `signalData` 必须指向本地地址（当前 NPU 上）。
- **Shape 语义**：
  - 对于单信号：Shape 为 `<1,1,1,1,1>`。
  - 对于信号张量：Shape 决定要等待的多维区域（最高 5-D）。张量中所有信号必须满足条件。
- **比较运算符**（WaitCmp）：
  | 值 | 条件 |
  |-------|-----------|
  | `EQ` | `signal == cmpValue` |
  | `NE` | `signal != cmpValue` |
  | `GT` | `signal > cmpValue` |
  | `GE` | `signal >= cmpValue` |
  | `LT` | `signal < cmpValue` |
  | `LE` | `signal <= cmpValue` |

## 示例

### 等待单信号

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void wait_for_ready(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);
    
    // 等待直到 signal == 1
    comm::TWAIT(sig, 1, comm::WaitCmp::EQ);
}
```

### 等待信号矩阵

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// 等待 4x8 密集工作网格的信号
void wait_worker_grid(__gm__ int32_t* signal_matrix) {
    comm::Signal2D<4, 8> grid(signal_matrix);
    
    // 等待直到所有 32 个信号 == 1
    comm::TWAIT(grid, 1, comm::WaitCmp::EQ);
}
```

### 等待计数器阈值

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void wait_for_count(__gm__ int32_t* local_counter, int expected_count) {
    comm::Signal counter(local_counter);
    
    // 等待直到 counter >= expected_count
    comm::TWAIT(counter, expected_count, comm::WaitCmp::GE);
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
