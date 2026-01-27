# TNOTIFY

## Introduction

Send flag notification to remote NPU. Used for lightweight synchronization between NPUs without transferring bulk data.

## Math Interpretation

For `NotifyOp::Set`:

$$ \mathrm{signal}^{\mathrm{remote}} = \mathrm{value} $$

For `NotifyOp::AtomicAdd`:

$$ \mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{value} \quad (\text{atomic}) $$

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
tnotify<Set> %signal_remote, %value : (!pto.memref<i32>, i32)
tnotify<AtomicAdd> %signal_remote, %value : (!pto.memref<i32>, i32)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Compile-time specified NotifyOp (recommended, zero overhead)
template <NotifyOp op = NotifyOp::Set, typename GlobalSignalData>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignal, int32_t value = 1);

// Runtime specified NotifyOp
template <typename GlobalSignalData>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignal, int32_t value, NotifyOp op);
```

## Constraints

- **Type constraints**:
  - `GlobalSignalData::DType` must be `int32_t` (32-bit signal).
- **Memory constraints**:
  - `dstSignal` must point to remote address (on target NPU).
- **Operation semantics**:
  - `NotifyOp::Set`: Direct store to remote memory.
  - `NotifyOp::AtomicAdd`: Hardware atomic add using `st_atomic` instruction.

## Examples

### Basic Set Notification

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void notify_set(__gm__ int32_t* remote_signal, int target_npu) {
    using GShape = Shape<1, 1, 1, 1, 1>;
    using GStride = Stride<1, 1, 1, 1, 1>;
    using GSignal = GlobalTensor<int32_t, GShape, GStride, Layout::ND>;

    GSignal sigG(remote_signal);
    
    // Set remote signal to 42
    comm::TNOTIFY<comm::NotifyOp::Set>(sigG, 42);
}
```

### Atomic Counter Increment

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void atomic_increment(__gm__ int32_t* remote_counter, int target_npu) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

    GSignal counterG(remote_counter);
    
    // Atomically add 1 to remote counter
    comm::TNOTIFY<comm::NotifyOp::AtomicAdd>(counterG, 1);
}
```

### Runtime Operation Selection

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void notify_runtime(__gm__ int32_t* remote_signal, int target_npu, bool use_atomic) {
    using GSignal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

    GSignal sigG(remote_signal);
    
    comm::NotifyOp op = use_atomic ? comm::NotifyOp::AtomicAdd : comm::NotifyOp::Set;
    comm::TNOTIFY(sigG, 1, op);
}
```
