# TTEST

## Introduction

Non-blocking test if signal(s) meet comparison condition. Returns `true` if condition is satisfied, `false` otherwise. Used for polling-based synchronization with timeout or interleaved work.

## Math Interpretation

Test and return result:

$$ \mathrm{result} = (\mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue}) $$

where `cmp` ∈ {`==`, `!=`, `>`, `>=`, `<`, `<=`}

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
%result = ttest<EQ> %signal, %cmp_value : (!pto.memref<i32>, i32) -> i1
%result = ttest<NE> %signal, %cmp_value : (!pto.memref<i32>, i32) -> i1
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Compile-time specified comparison (recommended, zero overhead)
template <WaitCmp cmp = WaitCmp::EQ, typename GlobalSignalData>
PTO_INST bool TTEST(GlobalSignalData &signal, int32_t cmpValue);

// Runtime specified comparison
template <typename GlobalSignalData>
PTO_INST bool TTEST(GlobalSignalData &signal, WaitCmp cmp, int32_t cmpValue);

// Test all signals in array (returns true only if ALL meet condition)
template <WaitCmp cmp = WaitCmp::EQ, typename GlobalSignalData>
PTO_INST bool TTEST_ALL(GlobalSignalData *signals, int count, int32_t cmpValue);

// Runtime specified comparison
template <typename GlobalSignalData>
PTO_INST bool TTEST_ALL(GlobalSignalData *signals, int count, WaitCmp cmp, int32_t cmpValue);
```

## Constraints

- **Type constraints**:
  - `GlobalSignalData::DType` must be `int32_t` (32-bit signal).
- **Memory constraints**:
  - `signal` must point to local address (on current NPU).
- **Return value**:
  - Returns `true` if condition is satisfied, `false` otherwise.
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
    return comm::TTEST<comm::WaitCmp::EQ>(sigG, 1);
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
        if (comm::TTEST<comm::WaitCmp::EQ>(sigG, 1)) {
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
    
    while (!comm::TTEST<comm::WaitCmp::GE>(counterG, expected_count)) {
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
    comm::TWAIT<comm::WaitCmp::EQ>(sigG, 1);

    // Non-blocking: returns immediately with result
    bool ready = comm::TTEST<comm::WaitCmp::EQ>(sigG, 1);
}
```
