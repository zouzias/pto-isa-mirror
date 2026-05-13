# pto.tpop

`pto.tpop` is part of the [System Scheduling](../../scalar/ops/micro-instruction/README.md) instruction set.

## Summary

`TPOP` is the consumer-side acquisition step for PTO FIFO-style pipe communication.

The public API surface provides pipe-based pop workflows, including slot-view style consumption, while backend implementations also support richer tile-oriented consumer paths for NPU FIFO workflows.

## Semantics

Conceptually, `TPOP` acquires consumer-side access to the current FIFO payload or slot associated with a pipe object.

Two common usage styles appear in the codebase:

- tile-oriented consumer acquisition for NPU pipe workflows,
- `GlobalData` slot-view consumption followed by explicit data access / release logic.

## C++ Intrinsic

`TPOP` overloads are declared in `include/pto/common/pto_instr.hpp`.

The current documentation here keeps the surface conservative and aligned with the verified pipe-consumer role rather than restating unverified older overload details.

## Constraints

!!! warning "Constraints"
    - `TPOP` is pipe-protocol specific; it is not a generic scalar stack pop.
    - Correctness depends on matching producer/consumer use of the same pipe protocol.
    - Pipe direction, slot sizing, and split mode must be compatible with the selected backend implementation.
    - Slot-view workflows must follow the surrounding FIFO protocol, including any explicit release step required by that path.

## Target-Profile Notes

- Tile-oriented FIFO protocol behavior is backend-specific and primarily NPU-focused.
- Older prose describing exact three-phase backend behavior may still reflect implementation intent, but users should treat the currently exported API surface as authoritative.

## Relationship with `TPUSH`

`TPOP` is the consumer-side counterpart of `TPUSH`. Consumer acquisition and producer publication must follow the same pipe contract.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_pop(auto &pipe, auto &slotView) {
    TPOP(pipe, slotView);
}
```
