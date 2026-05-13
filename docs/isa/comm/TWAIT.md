# pto.twait

## Summary

`pto.twait` blocks until a signal object satisfies a comparison against an `int32_t` reference value.

## Semantics

`TWAIT` is the blocking counterpart to `TTEST`.

The verified public wrapper accepts:

- signal object,
- `int32_t` comparison value,
- `WaitCmp` comparison mode, and
- optional event tokens to wait on before the wait begins.

Conceptually, execution waits until the implementation-defined signal state satisfies the chosen comparison relation.

## Assembly Syntax

```text
pto.twait %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TWAIT(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - The public wrapper hard-codes the compare value type to `int32_t`.
    - Signal storage must be compatible with the selected backend implementation.
    - The wrapper waits on all incoming event tokens before entering the wait.

## Relationship with `TNOTIFY` / `TTEST`

- `TNOTIFY` produces signal updates.
- `TWAIT` blocks until the update becomes visible and satisfies the comparison.
- `TTEST` provides the non-blocking test form.

## Examples

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void wait_ready(auto &signal) {
    comm::TWAIT(signal, 1, comm::WaitCmp::EQ);
}
```
