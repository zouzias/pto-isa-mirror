# TTEST

## 简介

非阻塞测试信号是否满足比较条件。满足条件返回 `true`，否则返回 `false`。用于基于轮询的同步，可配合超时或交替工作使用。

支持单信号或多维信号张量（最高 5-D，shape 由 GlobalTensor 派生）。对于张量，仅当所有信号都满足条件时才返回 `true`。

## 数学表示

测试并返回结果：

单信号：

$$ \mathrm{result} = (\mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue}) $$

信号张量（所有元素必须满足）：

$$ \mathrm{result} = \bigwedge_{d_0, d_1, d_2, d_3, d_4} (\mathrm{signal}_{d_0, d_1, d_2, d_3, d_4} \;\mathtt{cmp}\; \mathrm{cmpValue}) $$

其中 `cmp` ∈ {`EQ`, `NE`, `GT`, `GE`, `LT`, `LE`}

## 汇编语法

PTO-AS 形式：见 [PTO-AS 规范](../../assembly/PTO-AS.md)。

```text
%result = ttest %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32) -> i1
%result = ttest %signal_matrix, %cmp_value {cmp = #pto.cmp<GE>} : (!pto.memref<i32, MxN>, i32) -> i1
```

## C++ Intrinsic

声明在 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST bool TTEST(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events);
```

## 约束

- **类型约束**：
  - `GlobalSignalData::DType` 必须为 `int32_t`（32 位信号）。
- **内存约束**：
  - `signalData` 必须指向本地地址（当前 NPU 上）。
- **返回值**：
  - 满足条件返回 `true`，否则返回 `false`。
  - 对于信号张量，仅当所有信号都满足条件时才返回 `true`。
- **Shape 语义**：
  - 对于单信号：Shape 为 `<1,1,1,1,1>`。
  - 对于信号张量：Shape 决定要测试的多维区域（最高 5-D）。
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

### 基本测试

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

bool check_ready(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);
    
    // 检查 signal 是否 == 1
    return comm::TTEST(sig, 1, comm::WaitCmp::EQ);
}
```

### 测试信号矩阵

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// 测试 4x8 密集工作网格的所有信号是否就绪
bool check_worker_grid(__gm__ int32_t* signal_matrix) {
    comm::Signal2D<4, 8> grid(signal_matrix);
    
    // 仅当所有 32 个信号 == 1 时返回 true
    return comm::TTEST(grid, 1, comm::WaitCmp::EQ);
}
```

### 带超时的轮询

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

bool poll_with_timeout(__gm__ int32_t* local_signal, int max_iterations) {
    comm::Signal sig(local_signal);
    
    for (int i = 0; i < max_iterations; ++i) {
        if (comm::TTEST(sig, 1, comm::WaitCmp::EQ)) {
            return true;  // 收到信号
        }
        // 可以在轮询间隔做其他工作
    }
    return false;  // 超时
}
```

### 基于进度的轮询

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void process_with_progress(__gm__ int32_t* local_counter, int expected_count) {
    comm::Signal counter(local_counter);
    
    while (!comm::TTEST(counter, expected_count, comm::WaitCmp::GE)) {
        // 等待期间做一些有用的工作
        // ...
    }
    // 已收到所有预期信号
}
```

### TWAIT 与 TTEST 对比

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void compare_wait_test(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);

    // 阻塞：自旋直到 signal == 1
    comm::TWAIT(sig, 1, comm::WaitCmp::EQ);

    // 非阻塞：立即返回结果
    bool ready = comm::TTEST(sig, 1, comm::WaitCmp::EQ);
}
```
