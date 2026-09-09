# TNOTIFY

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T03:14:19.229Z pushedAt=2026-08-29T09:05:18.406Z -->

## Introduction

Sends a flag notification to the remote NPU. It is used for lightweight synchronization between NPUs without transferring large amounts of data.

## Mathematical Semantics

For `NotifyOp::Set`:

$$\mathrm{signal}^{\mathrm{remote}} = \mathrm{value}$$

For `NotifyOp::AtomicAdd`:

$$\mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{value} \quad (\text{atomic operation})$$

## Assembly Syntax

```text
tnotify %signal_remote, %value {op = #pto.notify_op<Set>} : (!pto.memref<i32>, i32)
tnotify %signal_remote, %value {op = #pto.notify_op<AtomicAdd>} : (!pto.memref<i32>, i32)
```

## C++ Built-in APIs

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignalData, int32_t value, NotifyOp op, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
    - `GlobalSignalData::DType` must be `int32_t` (32-bit signal).
- **Memory constraints**:
    - `dstSignalData` must point to a remote address (target NPU).
    - `dstSignalData` must be 4-byte aligned.
- **Operation semantics**:
    - `NotifyOp::Set`: directly stores to the remote memory.
    - `NotifyOp::AtomicAdd`: performs a hardware atomic addition using the `st_atomic` instruction.

## Examples

### Basic Set Notification

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void notify_set(__gm__ int32_t* remote_signal) {
    comm::Signal sig(remote_signal);

    // Set the remote signal to 1.
    comm::TNOTIFY(sig, 1, comm::NotifyOp::Set);
}
```

### Atomic Counter Increment

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void atomic_increment(__gm__ int32_t* remote_counter) {
    comm::Signal counter(remote_counter);

    // Atomically increment the remote counter by 1.
    comm::TNOTIFY(counter, 1, comm::NotifyOp::AtomicAdd);
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
