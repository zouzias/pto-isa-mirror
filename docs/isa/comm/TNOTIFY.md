# TNOTIFY

## Introduction

Send flag notification to remote NPU. Used for lightweight synchronization between NPUs without transferring bulk data.

## Math Interpretation

For `NotifyOp::Set`:

$$ \mathrm{signal}^{\mathrm{remote}} = \mathrm{value} $$

For `NotifyOp::AtomicAdd`:

$$ \mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{value} \quad (\text{atomic}) $$

## Assembly Syntax

```text
tnotify %signal_remote, %value {op = #pto.notify_op<Set>} : (!pto.memref<i32>, i32)
tnotify %signal_remote, %value {op = #pto.notify_op<AtomicAdd>} : (!pto.memref<i32>, i32)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignalData, int32_t value, NotifyOp op, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
    - `GlobalSignalData::DType` must be `int32_t` (32-bit signal).
- **Memory constraints**:
    - `dstSignalData` must point to remote address (on target NPU).
    - `dstSignalData` should be 4-byte aligned.
- **Operation semantics**:
    - `NotifyOp::Set`: Direct store to remote memory.
    - `NotifyOp::AtomicAdd`: Hardware atomic add using `st_atomic` instruction.
- **Ordering constraints**:
    - When sending multiple `Set` notifications to the same remote address, the consumer must have read the previous value before the producer writes the next one. Otherwise `TWAIT(EQ)` may block forever because the previous value was overwritten. Use a reverse ack to enforce ordering (see examples).

## Examples

### Basic Set Notification

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void notify_set(__gm__ int32_t* remote_signal) {
    comm::Signal sig(remote_signal);
    
    // Set remote signal to 1
    comm::TNOTIFY(sig, 1, comm::NotifyOp::Set);
}
```

### Atomic Counter Increment

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void atomic_increment(__gm__ int32_t* remote_counter) {
    comm::Signal counter(remote_counter);
    
    // Atomically add 1 to remote counter
    comm::TNOTIFY(counter, 1, comm::NotifyOp::AtomicAdd);
}
```

### Producer-Consumer Pattern

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Producer: notify when data is ready
void producer(__gm__ int32_t* remote_flag) {
    // ... produce data ...
    
    comm::Signal flag(remote_flag);
    comm::TNOTIFY(flag, 1, comm::NotifyOp::Set);
}

// Consumer: wait for data
void consumer(__gm__ int32_t* local_flag) {
    comm::Signal flag(local_flag);
    comm::TWAIT(flag, 1, comm::WaitCmp::EQ);
    
    // ... consume data ...
}
```

### Multiple Notifications to the Same Address (Reverse Ack for Ordering)

**Wrong** — producer sends two Set notifications back-to-back without waiting:

```text
Producer                      Consumer
  │  TNOTIFY(flag, 1, Set)      │
  │ ──────────────────────────> │
  │  TNOTIFY(flag, 2, Set)      │   consumer hasn't called TWAIT yet
  │ ──────────────────────────> │   flag overwritten to 2
  │                              │   TWAIT(flag, 1, EQ) → hangs forever!
```

**Correct** — producer waits for a reverse ack before sending the next notification:

```text
Producer                      Consumer
  │  TNOTIFY(flag, 1, Set)      │
  │ ──────────────────────────> │
  │                              │   TWAIT(flag, 1, EQ) ✓
  │                              │   TNOTIFY(ack, 1, Set)
  │  <────────────────────────  │
  │  TWAIT(ack, 1, EQ) ✓        │
  │  TNOTIFY(flag, 2, Set)      │
  │ ──────────────────────────> │
  │                              │   TWAIT(flag, 2, EQ) ✓
```

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Producer: send two values to the same address, gated by an ack
void producer(__gm__ int32_t* remote_flag, __gm__ int32_t* local_ack) {
    comm::Signal flag(remote_flag);
    comm::Signal ack(local_ack);

    comm::TNOTIFY(flag, 1, comm::NotifyOp::Set);   // round 1
    comm::TWAIT(ack, 1, comm::WaitCmp::EQ);        // wait until consumer read round 1
    comm::TNOTIFY(flag, 2, comm::NotifyOp::Set);   // round 2 (now safe)
}

// Consumer: read each value, ack after round 1
void consumer(__gm__ int32_t* local_flag, __gm__ int32_t* remote_ack) {
    comm::Signal flag(local_flag);
    comm::Signal ack(remote_ack);

    comm::TWAIT(flag, 1, comm::WaitCmp::EQ);       // round 1
    comm::TNOTIFY(ack, 1, comm::NotifyOp::Set);    // tell producer round 1 is consumed
    comm::TWAIT(flag, 2, comm::WaitCmp::EQ);       // round 2
}
```
