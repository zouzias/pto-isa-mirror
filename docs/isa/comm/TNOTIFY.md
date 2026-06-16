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

### Multiple Notifications to the Same Address (Ordering Required)

When sending multiple flag notifications to the **same** remote address in
sequence, you must ensure the consumer has read the **previous value** before
the producer overwrites it with the next one. `Set` is a direct overwrite store
and `TWAIT(EQ)` is an exact-value match, so if the producer writes the next
value first, the consumer's `TWAIT` will never observe the previous value and
will hang.

Approach: after receiving a flag notification, the consumer replies with an ack;
the producer waits for the consumer's ack after sending the first flag
notification, and only sends the next flag notification after receiving it:

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
