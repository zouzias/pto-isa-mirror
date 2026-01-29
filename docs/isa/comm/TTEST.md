# TTEST

## Introduction

Non-blocking test if signal(s) meet comparison condition. Returns `true` if condition is satisfied, `false` otherwise. Used for polling-based synchronization with timeout or interleaved work.

Supports single signal or 2D signal matrix (shape derived from GlobalTensor). For matrix, returns `true` only if ALL signals meet the condition.

## Math Interpretation

Test and return result:

Single signal:

$$ \mathrm{result} = (\mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue}) $$

Signal matrix (all must satisfy):

$$ \mathrm{result} = \bigwedge_{i,j} (\mathrm{signal}_{i,j} \;\mathtt{cmp}\; \mathrm{cmpValue}) $$

where `cmp` ∈ {`EQ`, `NE`, `GT`, `GE`, `LT`, `LE`}

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

```text
%result = ttest %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32) -> i1
%result = ttest %signal_matrix, %cmp_value {cmp = #pto.cmp<GE>} : (!pto.memref<i32, MxN>, i32) -> i1
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST bool TTEST(GlobalSignalData &signal, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `GlobalSignalData::DType` must be `int32_t` (32-bit signal).
- **Memory constraints**:
  - `signal` must point to local address (on current NPU).
- **Return value**:
  - Returns `true` if condition is satisfied, `false` otherwise.
  - For signal matrix, returns `true` only if ALL signals satisfy the condition.
- **Shape semantics**:
  - For single signal: Shape is `<1,1,1,1,1>`.
  - For signal matrix: Shape determines the 2D region to test.
- **Comparison operators** (WaitCmp):
  | Value | Condition |
  |-------|-----------|
  | `EQ` | `signal == cmpValue` |
  | `NE` | `signal != cmpValue` |
  | `GT` | `signal > cmpValue` |
  | `GE` | `signal >= cmpValue` |
  | `LT` | `signal < cmpValue` |
  | `LE` | `signal <= cmpValue` |

## Examples

### Basic Test

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

bool check_ready(__gm__ int32_t* local_signal) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

    GSignal sigG(local_signal);
    
    // Check if signal == 1
    return comm::TTEST(sigG, 1, comm::WaitCmp::EQ);
}
```

### Test Signal Matrix

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Test if all signals from a 4x8 grid of workers are ready
bool check_worker_grid(__gm__ int32_t* signal_matrix) {
    constexpr int ROWS = 4;
    constexpr int COLS = 8;
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,ROWS,COLS>, 
                                 Stride<ROWS*COLS,ROWS*COLS,ROWS*COLS,COLS,1>, Layout::ND>;

    GSignal sigMatrix(signal_matrix);
    
    // Returns true only if all 32 signals == 1
    return comm::TTEST(sigMatrix, 1, comm::WaitCmp::EQ);
}
```

### Polling with Timeout

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

bool poll_with_timeout(__gm__ int32_t* local_signal, int max_iterations) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

    GSignal sigG(local_signal);
    
    for (int i = 0; i < max_iterations; ++i) {
        if (comm::TTEST(sigG, 1, comm::WaitCmp::EQ)) {
            return true;  // Signal received
        }
        // Could do other work here between polls
    }
    return false;  // Timeout
}
```

### Progress-Based Polling

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void process_with_progress(__gm__ int32_t* local_counter, int expected_count) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

    GSignal counterG(local_counter);
    
    while (!comm::TTEST(counterG, expected_count, comm::WaitCmp::GE)) {
        // Do some useful work while waiting
        // ...
    }
    // All expected signals received
}
```

### Compare TWAIT vs TTEST

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void compare_wait_test(__gm__ int32_t* local_signal) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;
    GSignal sigG(local_signal);

    // Blocking: spins until signal == 1
    comm::TWAIT(sigG, 1, comm::WaitCmp::EQ);

    // Non-blocking: returns immediately with result
    bool ready = comm::TTEST(sigG, 1, comm::WaitCmp::EQ);
}
```
