# TTEST

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T03:16:49.986Z pushedAt=2026-08-29T09:05:18.409Z -->

## Introduction

Detects whether a signal satisfies a comparison condition in non-blocking mode. It returns `true` if satisfied and `false` otherwise, and is suitable for polling-based synchronization (with timeout) or scenarios interleaved with other work.

A single signal or multi-dimensional signal tensor (up to 5 dimensions, with the shape determined by GlobalSignalData) is supported. For a tensor, `true` is returned only when **all** signals satisfy the condition.

## Mathematical Semantics

Detects and returns the result:

Single signal:

$$\mathrm{result} = (\mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue})$$

Signal tensor (all elements must be satisfied):

$$\mathrm{result} = \bigwedge_{d_0, d_1, d_2, d_3, d_4} (\mathrm{signal}_{d_0, d_1, d_2, d_3, d_4} \;\mathtt{cmp}\; \mathrm{cmpValue})$$

Where `cmp` ∈ {`EQ`, `NE`, `GT`, `GE`, `LT`, `LE`}

## Assembly Syntax

```text
%result = ttest %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32) -> i1
%result = ttest %signal_matrix, %cmp_value {cmp = #pto.cmp<GE>} : (!pto.memref<i32, MxN>, i32) -> i1
```

## C++ Built-in APIs

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST bool TTEST(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp);
```

## Constraints

- **Type constraints**:
    - `GlobalSignalData::DType` must be `int32_t` (32-bit signal).
- **Memory constraints**:
    - `signalData` must point to a local address (on the current NPU).
- **Return value**:
    - Returns `true` when the condition is satisfied, and `false` otherwise.
    - For a signal tensor, returns `true` only when all signals satisfy the condition.
- **Shape semantics**:
    - Single signal: the shape is `<1,1,1,1,1>`.
    - Signal tensor: the shape determines the multi-dimensional region to be detected (up to 5 dimensions).
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

### Basic Detection

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

bool check_ready(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);

    // Detect signal == 1.
    return comm::TTEST(sig, 1, comm::WaitCmp::EQ);
}
```

### Signal Matrix Detection

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Check whether the signals of all workers in the 4x8 grid are ready.
bool check_worker_grid(__gm__ int32_t* signal_matrix) {
    comm::Signal2D<4, 8> grid(signal_matrix);

    // Return true only when all 32 signals are 1.
    return comm::TTEST(grid, 1, comm::WaitCmp::EQ);
}
```

### Polling With Timeout

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

bool poll_with_timeout(__gm__ int32_t* local_signal, int max_iterations) {
    comm::Signal sig(local_signal);

    for (int i = 0; i < max_iterations; ++i) {
        if (comm::TTEST(sig, 1, comm::WaitCmp::EQ)) {
            return true;  // Signal received.
        }
        // Other work can be performed between two polls.
    }
    return false;  // Timeout.
}
```

### Progress-Based Polling

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void process_with_progress(__gm__ int32_t* local_counter, int expected_count) {
    comm::Signal counter(local_counter);

    while (!comm::TTEST(counter, expected_count, comm::WaitCmp::GE)) {
        // Perform other useful work while waiting.
        // ...
    }
    // All expected signals have been received.
}
```

### TWAIT and TTEST Comparison

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void compare_wait_test(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);

    // Blocking: spins until signal == 1.
    comm::TWAIT(sig, 1, comm::WaitCmp::EQ);

    // Non-blocking: returns the result immediately.
    bool ready = comm::TTEST(sig, 1, comm::WaitCmp::EQ);
}
```
