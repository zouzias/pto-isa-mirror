# pto.tnotify

## Summary

`pto.tnotify` sends an integer notification value to a signal location using a selected notify operation.

## Semantics

The verified public wrapper accepts:

- destination signal object,
- `int32_t` notification value,
- `NotifyOp`, and
- optional event tokens to wait on before issue.

Conceptually:

- `NotifyOp::Set` writes the given value.
- `NotifyOp` atomic-style variants perform the backend-selected update semantics associated with that enum value.

## Assembly Syntax

```text
pto.tnotify %signal_remote, %value {op = #pto.notify_op<Set>} : (!pto.memref<i32>, i32)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignalData, int32_t value, NotifyOp op, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - The public wrapper hard-codes the value type to `int32_t`.
    - Signal storage must be compatible with the selected backend implementation.
    - The wrapper waits on all incoming event tokens before issuing the notification.

## Relationship with `TWAIT` / `TTEST`

`TNOTIFY` is typically paired with `TWAIT` or `TTEST` for flag-style synchronization and polling.

## Examples

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void notify_one(auto &signal) {
    comm::TNOTIFY(signal, 1, comm::NotifyOp::Set);
}
```
