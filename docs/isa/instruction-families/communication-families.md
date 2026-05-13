# Communication Instruction Set

The Communication ISA covers inter-NPU point-to-point transfer, collective data movement, collective reduction, and signal-style synchronization.

## Instruction Overview

| Operation | PTO Name | Description |
|---|---|---|
| Point-to-point put | `pto.tput` | Synchronous remote write |
| Point-to-point get | `pto.tget` | Synchronous remote read |
| Async put | `pto.tput_async` | Asynchronous remote write returning `AsyncEvent` |
| Async get | `pto.tget_async` | Asynchronous remote read returning `AsyncEvent` |
| Broadcast | `pto.tbroadcast` | Broadcast from a source buffer to group destinations |
| Scatter | `pto.tscatter` | Scatter from a source buffer to group destinations |
| Gather | `pto.tgather` | Gather from group sources into a destination buffer |
| Reduce | `pto.treduce` | Collective reduction into a destination buffer |
| Notify | `pto.tnotify` | Signal / notify update |
| Test | `pto.ttest` | Non-blocking signal test |
| Wait | `pto.twait` | Blocking signal wait |

## Shared Programming Model

The verified public wrappers in `include/pto/comm/pto_comm_inst.hpp` show several common patterns:

- synchronous data-moving communication wrappers return `RecordEvent`,
- `TNOTIFY` and `TWAIT` return `void`, while `TTEST` returns `bool`,
- wrappers wait on incoming event tokens before dispatching the underlying implementation,
- several collectives and point-to-point movement operations expose explicit single-staging-tile and ping-pong staging-tile overloads,
- async point-to-point operations use `AsyncSession` and return `AsyncEvent`.

## Shared Operands

Communication instructions may use:

- `ParallelGroup` handles,
- source / destination `GlobalTensor` views,
- explicit staging tiles,
- reduction / compare / notify enums,
- `RecordEvent` and `AsyncEvent` synchronization objects.

## Shared Constraints

!!! warning "Constraints"
    - Collective operations depend on a semantically consistent `ParallelGroup` contract across participating ranks.
    - Buffer roles, element compatibility, layout compatibility, and staging-tile requirements are operation-specific and defined on the per-op pages.
    - Async operations require a previously built `AsyncSession` and explicit completion checks through the returned `AsyncEvent`.
    - CPU simulator availability and backend-specific transport restrictions are implementation-specific; see per-op pages and backend code when needed.

## Not Allowed

!!! danger "Cases That Are Not Allowed"
    - Treating backend-specific collective conventions as if they were all explicitly validated by the public wrapper.
    - Reusing async results without checking completion through the appropriate `AsyncEvent` API.
    - Assuming one communication op's root / rank convention automatically applies to a different op without checking its contract.

## Navigation

- [Communication ISA reference](../comm/README.md)
- [Communication and Runtime](../comm/communication-runtime.md)
