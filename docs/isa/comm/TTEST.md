# pto.ttest

## Summary

`pto.ttest` performs a non-blocking comparison test on a signal object and returns a boolean result.

## Semantics

The verified public wrapper accepts:

- signal object,
- `int32_t` comparison value,
- `WaitCmp` comparison mode, and
- optional event tokens to wait on before issuing the test.

It returns the boolean result produced by the backend implementation.

## Assembly Syntax

```text
%result = pto.ttest %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32) -> i1
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST bool TTEST(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - The public wrapper hard-codes the compare value type to `int32_t`.
    - Signal storage must be compatible with the selected backend implementation.
    - The wrapper waits on all incoming event tokens before issuing the test.

## Relationship with `TWAIT` / `TNOTIFY`

- `TTEST` is the non-blocking companion to `TWAIT`.
- `TNOTIFY` typically provides the producer-side signal update.

## Examples

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

bool is_ready(auto &signal) {
    return comm::TTEST(signal, 1, comm::WaitCmp::EQ);
}
```
