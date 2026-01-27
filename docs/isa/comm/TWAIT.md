# TWAIT

## Introduction

Blocking wait until signal(s) meet comparison condition. Used in conjunction with `TNOTIFY` for synchronization.

## Math Interpretation

Wait (spin) until the following condition is satisfied:

$$ \mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue} $$

where `cmp` ∈ {`==`, `!=`, `>`, `>=`, `<`, `<=`}

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
twait<EQ> %signal, %cmp_value : (!pto.memref<i32>, i32)
twait<GE> %signal, %cmp_value : (!pto.memref<i32>, i32)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Compile-time specified comparison (recommended, zero overhead)
template <WaitCmp cmp = WaitCmp::EQ, typename GlobalSignalData>
PTO_INST void TWAIT(GlobalSignalData &signal, int32_t cmpValue);

// Runtime specified comparison
template <typename GlobalSignalData>
PTO_INST void TWAIT(GlobalSignalData &signal, WaitCmp cmp, int32_t cmpValue);

// Wait for all signals in array to meet condition
template <WaitCmp cmp = WaitCmp::EQ, typename GlobalSignalData>
PTO_INST void TWAIT_ALL(GlobalSignalData *signals, int count, int32_t cmpValue);

// Runtime specified comparison
template <typename GlobalSignalData>
PTO_INST void TWAIT_ALL(GlobalSignalData *signals, int count, WaitCmp cmp, int32_t cmpValue);
```

## Constraints

- **Type constraints**:
  - `GlobalSignalData::DType` must be `int32_t` (32-bit signal).
- **Memory constraints**:
  - `signal` must point to local address (on current NPU).
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

### Wait for Signal Equals Value

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void wait_for_ready(__gm__ int32_t* local_signal) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

    GSignal sigG(local_signal);
    
    // Wait until signal == 1
    comm::TWAIT<comm::WaitCmp::EQ>(sigG, 1);
}
```

### Wait for Counter Reaches Threshold

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void wait_for_count(__gm__ int32_t* local_counter, int expected_count) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

    GSignal counterG(local_counter);
    
    // Wait until counter >= expected_count
    comm::TWAIT<comm::WaitCmp::GE>(counterG, expected_count);
}
```

### Producer-Consumer Pattern

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Producer: notify when data is ready
void producer(__gm__ int32_t* remote_flag, int consumer_npu) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

    // ... produce data ...
    
    GSignal flagG(remote_flag);
    comm::TNOTIFY<comm::NotifyOp::Set>(flagG, 1);
}

// Consumer: wait for data
void consumer(__gm__ int32_t* local_flag) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

    GSignal flagG(local_flag);
    comm::TWAIT<comm::WaitCmp::EQ>(flagG, 1);
    
    // ... consume data ...
}
```
