# PTO-ISA TTEST 指令详解

## 概述

`TTEST` 是 PTO-COMM-ISA 中的一个**非阻塞信号测试指令**，用于检查信号值是否满足特定的比较条件。它是 PTO 通信指令集中的重要组成部分，与 `TWAIT`、`TNOTIFY` 配合使用实现高效的同步机制。

## 核心特性

### 1. 非阻塞特性
- **立即返回**：`TTEST` 不会阻塞执行，立即返回测试结果
- **返回值**：满足条件返回 `true`，否则返回 `false`
- **适用场景**：轮询式同步、超时处理、交错工作

### 2. 与 TWAIT 的区别

| 特性 | TTEST | TWAIT |
|------|-------|-------|
| **阻塞性** | 非阻塞，立即返回 | 阻塞，直到条件满足 |
| **返回值** | `bool` (true/false) | `void` |
| **使用场景** | 轮询检查、超时处理 | 确定性等待 |
| **性能** | 适合在等待时做其他工作 | 硬件级优化，等待效率高 |

### 3. 信号矩阵支持
- **单信号**：测试单个 32 位整型信号
- **信号矩阵**：支持 2D 信号矩阵（通过 GlobalTensor 的形状确定）
- **矩阵语义**：对于信号矩阵，只有当**所有信号**都满足条件时才返回 `true`

## 函数签名

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST bool TTEST(
    GlobalSignalData &signalData,  // 要测试的信号（必须是 int32_t 类型）
    int32_t cmpValue,               // 比较值
    WaitCmp cmp,                    // 比较运算符
    WaitEvents&... events           // 可选的等待事件
);
```

## 支持的比较运算符

| `WaitCmp` 枚举值 | 含义 | 条件表达式 |
|-----------------|------|-----------|
| `EQ` | 等于 | `signal == cmpValue` |
| `NE` | 不等于 | `signal != cmpValue` |
| `GT` | 大于 | `signal > cmpValue` |
| `GE` | 大于等于 | `signal >= cmpValue` |
| `LT` | 小于 | `signal < cmpValue` |
| `LE` | 小于等于 | `signal <= cmpValue` |

## 约束条件

1. **类型约束**：
   - `GlobalSignalData::DType` 必须是 `int32_t`（32 位信号）
   
2. **内存约束**：
   - `signalData` 必须指向本地地址（当前 NPU 上的地址）
   - 不能直接测试远程地址，需要通过 `ShmemPtr` 获取本地映射

3. **返回值语义**：
   - 单信号：条件满足返回 `true`，否则返回 `false`
   - 信号矩阵：所有信号都满足条件才返回 `true`

## 实现原理

从 `include/pto/comm/TTest.hpp` 的实现可以看到：

```cpp
template <typename GlobalSignalData>
PTO_INTERNAL bool TTEST_IMPL(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp)
{
    // 1. 获取信号的 5-D 形状和步长
    // 2. 遍历所有信号元素
    // 3. 使用 DCCI（Data Cache Coherency Instruction）确保缓存一致性
    // 4. 对每个信号执行比较操作
    // 5. 如果所有信号都满足条件，返回 true
}
```

**关键实现细节**：
- 使用 `dcci` 指令确保缓存一致性
- 支持 5-D 张量的完整遍历
- 短路求值：一旦发现不满足条件的信号，立即返回 `false`

## 使用示例

### 示例 1：基础信号测试

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void check_ready(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);
    
    // 检查信号是否等于 1
    if (comm::TTEST(sig, 1, comm::WaitCmp::EQ)) {
        // 信号已就绪，可以继续处理
    } else {
        // 信号未就绪，可以做其他工作
    }
}
```

### 示例 2：轮询模式（带超时）

```cpp
bool poll_with_timeout(__gm__ int32_t* local_signal, int max_iterations) {
    comm::Signal sig(local_signal);
    
    for (int i = 0; i < max_iterations; ++i) {
        if (comm::TTEST(sig, 1, comm::WaitCmp::EQ)) {
            return true;  // 信号已接收
        }
        // 在轮询间隔中可以做一些有用的工作
        // ...
    }
    return false;  // 超时
}
```

### 示例 3：进度检查（基于计数器）

```cpp
void process_with_progress(__gm__ int32_t* local_counter, int expected_count) {
    comm::Signal counter(local_counter);
    
    while (!comm::TTEST(counter, expected_count, comm::WaitCmp::GE)) {
        // 在等待时可以做一些有用的工作
        // 例如：处理其他任务、预加载数据等
    }
    // 所有预期的信号都已接收
}
```

### 示例 4：信号矩阵测试

```cpp
// 测试 4x8 的工作网格是否全部就绪
bool check_worker_grid(__gm__ int32_t* signal_matrix) {
    comm::Signal2D<4, 8> grid(signal_matrix);
    
    // 只有当所有 32 个信号都 == 1 时才返回 true
    return comm::TTEST(grid, 1, comm::WaitCmp::EQ);
}
```

### 示例 5：在 GEMM AllReduce 中的应用

在 `gemm_allreduce_opt` 中，`TTEST` 用于轮询计算核的队列：

```cpp
// 轮询队列检查是否有就绪的 tile
volatile __gm__ PerBlockQueue* pq = GetMyBlockQueue(qset, queue_idx);
pto::comm::Signal sig(const_cast<__gm__ int32_t*>(&pq->count));

// 非阻塞检查：队列中是否有新的 tile
if (comm::TTEST(sig, expected_count, comm::WaitCmp::GE)) {
    // 有就绪的 tile，可以处理
    int32_t tile = PerBlockQueueTryDequeue(pq, head);
} else {
    // 没有就绪的 tile，可以轮询其他队列或做其他工作
}
```

## TTEST vs TWAIT 选择指南

### 使用 TTEST 的场景：
1. **需要超时处理**：在等待信号时设置最大等待时间
2. **交错工作**：在等待时可以处理其他任务
3. **多队列轮询**：需要检查多个队列，不想被单个队列阻塞
4. **条件检查**：只需要检查条件是否满足，不需要等待

### 使用 TWAIT 的场景：
1. **确定性等待**：必须等待条件满足才能继续
2. **简单同步**：不需要在等待时做其他工作
3. **性能优化**：硬件级优化，等待效率更高

### 优化建议

在 `gemm_allreduce_opt` 中，有一个重要的优化：

**优化前（TTEST + TWAIT 模式）**：
```cpp
// 先使用 TTEST 检查
if (!comm::TTEST(sig, expected_count, comm::WaitCmp::GE)) {
    // 如果未就绪，再使用 TWAIT 等待
    comm::TWAIT(sig, expected_count, comm::WaitCmp::GE);
}
```

**优化后（直接使用 TWAIT）**：
```cpp
// TWAIT 在硬件层面会先检查条件，如果已满足则立即返回
// 因此不需要先调用 TTEST
comm::TWAIT(sig, expected_count, comm::WaitCmp::GE);
```

**优化原因**：
- `TWAIT` 在硬件层面已经实现了条件检查
- 如果条件已满足，`TWAIT` 会立即返回，不会阻塞
- 减少了一次 `TTEST` 调用的开销

## 数学语义

### 单信号测试
$$\mathrm{result} = (\mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue})$$

### 信号矩阵测试（所有信号必须满足）
$$\mathrm{result} = \bigwedge_{i,j} (\mathrm{signal}_{i,j} \;\mathtt{cmp}\; \mathrm{cmpValue})$$

其中 `cmp` ∈ {`EQ`, `NE`, `GT`, `GE`, `LT`, `LE`}

## 性能考虑

1. **缓存一致性**：`TTEST` 使用 `dcci` 指令确保缓存一致性，但会有一定开销
2. **矩阵测试**：对于大型信号矩阵，需要遍历所有元素，开销较大
3. **轮询频率**：在轮询循环中，可以适当降低 `TTEST` 的调用频率，在两次测试之间做一些有用工作

## 相关指令

- **TWAIT**：阻塞等待信号条件满足
- **TNOTIFY**：发送信号通知到远程 NPU
- **TSYNC**：等待异步事件完成（用于 `TPUT_ASYNC`/`TGET_ASYNC`）

## 参考资料

- 官方文档：`docs/isa/comm/TTEST.md`
- 测试用例：`tests/npu/a2a3/comm/st/testcase/ttest/`
- 实现代码：`include/pto/comm/TTest.hpp`
- 接口定义：`include/pto/comm/pto_comm_inst.hpp`
