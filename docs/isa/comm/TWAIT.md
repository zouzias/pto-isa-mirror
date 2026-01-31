# TWAIT

## Introduction

Blocking wait until signal(s) meet comparison condition. Used in conjunction with `TNOTIFY` for flag-based synchronization.

Supports single signal or 2D signal matrix (shape derived from GlobalTensor).

> **Note**: For waiting on `AsyncEvent` (from `TPUT_ASYNC`/`TGET_ASYNC`), use `TSYNC` instead.

## Math Interpretation

Wait (spin) until the following condition is satisfied:

Single signal:

$$ \mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue} $$

Signal matrix (all must satisfy):

$$ \forall i,j: \mathrm{signal}_{i,j} \;\mathtt{cmp}\; \mathrm{cmpValue} $$

where `cmp` ∈ {`EQ`, `NE`, `GT`, `GE`, `LT`, `LE`}

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

```text
twait %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32)
twait %signal_matrix, %cmp_value {cmp = #pto.cmp<GE>} : (!pto.memref<i32, MxN>, i32)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TWAIT(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `GlobalSignalData::DType` must be `int32_t` (32-bit signal).
- **Memory constraints**:
  - `signalData` must point to local address (on current NPU).
- **Shape semantics**:
  - For single signal: Shape is `<1,1,1,1,1>`.
  - For signal matrix: Shape determines the 2D region to wait on. All signals must satisfy the condition.
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

### Wait for Single Signal

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void wait_for_ready(__gm__ int32_t* local_signal) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

    GSignal sigG(local_signal);
    
    // Wait until signal == 1
    comm::TWAIT(sigG, 1, comm::WaitCmp::EQ);
}
```

### Wait for Signal Matrix

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Wait for signals from a 4x8 grid of workers
void wait_worker_grid(__gm__ int32_t* signal_matrix) {
    constexpr int ROWS = 4;
    constexpr int COLS = 8;
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,ROWS,COLS>, 
                                 Stride<ROWS*COLS,ROWS*COLS,ROWS*COLS,COLS,1>, Layout::ND>;

    GSignal sigMatrix(signal_matrix);
    
    // Wait until all 32 signals == 1
    comm::TWAIT(sigMatrix, 1, comm::WaitCmp::EQ);
}
```

### Wait for Counter Threshold

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void wait_for_count(__gm__ int32_t* local_counter, int expected_count) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

    GSignal counterG(local_counter);
    
    // Wait until counter >= expected_count
    comm::TWAIT(counterG, expected_count, comm::WaitCmp::GE);
}
```

### Producer-Consumer Pattern

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Producer: notify when data is ready
void producer(__gm__ int32_t* remote_flag) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

    // ... produce data ...
    
    GSignal flagG(remote_flag);
    comm::TNOTIFY(flagG, 1, comm::NotifyOp::Set);
}

// Consumer: wait for data
void consumer(__gm__ int32_t* local_flag) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

    GSignal flagG(local_flag);
    comm::TWAIT(flagG, 1, comm::WaitCmp::EQ);
    
    // ... consume data ...
}
```