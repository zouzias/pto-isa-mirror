# TWAIT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T03:18:09.800Z pushedAt=2026-08-29T09:05:18.410Z -->

## Introduction

Spin-waits until the signal satisfies the comparison condition. It is used together with `TNOTIFY` to implement flag-based synchronization.

A single signal or multi-dimensional signal tensor (up to 5 dimensions, with the shape determined by GlobalSignalData) is supported.

## Mathematical Semantics

Spin-waits until the following condition is satisfied:

Single signal:

$$\mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue}$$

Signal tensor (all elements must satisfy the condition):

$$\forall d_0, d_1, d_2, d_3, d_4: \mathrm{signal}_{d_0, d_1, d_2, d_3, d_4} \;\mathtt{cmp}\; \mathrm{cmpValue}$$

Where `cmp` ∈ {`EQ`, `NE`, `GT`, `GE`, `LT`, `LE`}

## Assembly Syntax

```text
twait %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32)
twait %signal_matrix, %cmp_value {cmp = #pto.cmp<GE>} : (!pto.memref<i32, MxN>, i32)
```

## C++ Built-in APIs

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TWAIT(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
    - `GlobalSignalData::DType` must be `int32_t` (32-bit signal).
- **Memory constraints**:
    - `signalData` must point to a GM/HBM address local to the current NPU (`__gm__`), which a remote NPU writes to through `TNOTIFY`. "Local" refers to NPU ownership (the current NPU's GM vs. the remote NPU's GM), not the CCE address space modifier.
- **Shape semantics**:
    - Single signal: the shape is `<1,1,1,1,1>`.
    - Signal tensor: the shape determines the multi-dimensional region to wait on (up to 5 dimensions). All signals in the tensor must satisfy the condition.
- **Comparison operators** (WaitCmp):

  | Value | Condition |
  |-------|--------|
  | `EQ` | `signal == cmpValue` |
  | `NE` | `signal != cmpValue` |
  | `GT` | `signal > cmpValue` |
  | `GE` | `signal >= cmpValue` |
  | `LT` | `signal < cmpValue` |
  | `LE` | `signal <= cmpValue` |

## Examples

### Waiting for a Single Signal

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void wait_for_ready(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);

    // Wait for signal == 1.
    comm::TWAIT(sig, 1, comm::WaitCmp::EQ);
}
```

### Waiting for a Signal Matrix

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Wait for the signals of all workers in the 4x8 grid to be ready.
void wait_worker_grid(__gm__ int32_t* signal_matrix) {
    comm::Signal2D<4, 8> grid(signal_matrix);

    // Wait for all 32 signals to be 1.
    comm::TWAIT(grid, 1, comm::WaitCmp::EQ);
}
```

### Waiting for the Counter Threshold

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void wait_for_count(__gm__ int32_t* local_counter, int expected_count) {
    comm::Signal counter(local_counter);

    // Wait for counter >= expected_count.
    comm::TWAIT(counter, expected_count, comm::WaitCmp::GE);
}
```

### Producer-Consumer Pattern

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Producer: sends a notification after data is ready.
void producer(__gm__ int32_t* remote_flag) {
    // ... produce data ...

    comm::Signal flag(remote_flag);
    comm::TNOTIFY(flag, 1, comm::NotifyOp::Set);
}

// Consumer: waits for data to be ready.
void consumer(__gm__ int32_t* local_flag) {
    comm::Signal flag(local_flag);
    comm::TWAIT(flag, 1, comm::WaitCmp::EQ);

    // ... consume data ...
}
```
