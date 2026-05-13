# pto.tpush

`pto.tpush` is part of the [System Scheduling](../../scalar/ops/micro-instruction/README.md) instruction set.

## Summary

`TPUSH` is the producer-side publication step for PTO FIFO-style pipe communication.

The public API surface provides at least a `GlobalData`-based form, and backend implementations also support tile-oriented producer paths for NPU FIFO workflows.

## Semantics

Conceptually, `TPUSH` publishes producer-side data availability for a FIFO slot associated with a pipe object.

Two common usage styles appear in the codebase:

- tile-oriented producer publication for NPU pipe workflows,
- `GlobalData` slot publication after slot acquisition / preparation.

## C++ Intrinsic

The verified public wrapper surface exposes `TPUSH` overloads in `include/pto/common/pto_instr.hpp`.

At minimum, the current inspected wrapper evidence directly confirms a `GlobalData` form:

```cpp
template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

Backend implementation code also indicates richer tile-oriented producer paths on NPU targets.

## Constraints

!!! warning "Constraints"
    - `TPUSH` is pipe-protocol specific; it is not a generic scalar stack push.
    - Correctness depends on matching producer/consumer use of the same pipe protocol.
    - Pipe direction, slot sizing, and split mode must be compatible with the selected backend implementation.
    - For slot-view / `GlobalData` workflows, the slot must be prepared according to the surrounding FIFO protocol before publication.

## Target-Profile Notes

- A5 backend code shows detailed producer-side support for multiple direction modes, split modes, and GM/UB FIFO paths.
- Tile-oriented FIFO protocol behavior is backend-specific and primarily NPU-focused.
- The exact wrapper overload set should be treated according to the current `pto_instr.hpp` surface rather than older broad prose assumptions.

## Relationship with `TPOP`

`TPUSH` is the producer-side counterpart of `TPOP`. Producer publication and consumer acquisition must follow the same pipe contract.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_push(auto &pipe, auto &slotView) {
    TPUSH(pipe, slotView);
}
```
