# Communication And Runtime

Communication instructions expose point-to-point transfer, collective movement, collective reduction, and signal-style synchronization across NPUs.

## Operations

| Operation | Description | IR spelling | C++ spelling |
|---|---|---|---|
| [TBROADCAST](./TBROADCAST.md) | Broadcast from a source buffer to group destinations | `pto.tbroadcast` | `TBROADCAST` |
| [TGET](./TGET.md) | Synchronous remote read | `pto.tget` | `TGET` |
| [TGET_ASYNC](./TGET_ASYNC.md) | Asynchronous remote read | `pto.tget_async` | `TGET_ASYNC` |
| [TNOTIFY](./TNOTIFY.md) | Signal/notify update | `pto.tnotify` | `TNOTIFY` |
| [TPUT](./TPUT.md) | Synchronous remote write | `pto.tput` | `TPUT` |
| [TPUT_ASYNC](./TPUT_ASYNC.md) | Asynchronous remote write | `pto.tput_async` | `TPUT_ASYNC` |
| [TREDUCE](./TREDUCE.md) | Collective reduction | `pto.treduce` | `TREDUCE` |
| [TSCATTER](./TSCATTER.md) | Scatter from source buffer to group destinations | `pto.tscatter` | `TSCATTER` |
| [TGATHER](./TGATHER.md) | Gather from group sources into destination buffer | `pto.tgather` | `TGATHER` |
| [TTEST](./TTEST.md) | Non-blocking signal test | `pto.ttest` | `TTEST` |
| [TWAIT](./TWAIT.md) | Blocking signal wait | `pto.twait` | `TWAIT` |

## Programming Model Notes

The verified public wrappers in `include/pto/comm/pto_comm_inst.hpp` show several common patterns:

- synchronous comm wrappers return `RecordEvent` for data-moving collectives and point-to-point operations,
- signal-style wrappers `TNOTIFY` and `TWAIT` return `void`, while `TTEST` returns `bool`,
- wrappers wait on incoming event tokens before dispatching the underlying implementation,
- several movement collectives expose explicit single-tile and ping-pong staging-tile overloads,
- async point-to-point operations are session-based and return `AsyncEvent`.

## Async Runtime Notes

Async transfer operations use `AsyncSession` and `AsyncEvent` as the public synchronization model.

Verified public helpers include:

- `BuildAsyncSession(...)`
- `AsyncEvent::Wait(const AsyncSession &session)`
- `AsyncEvent::Test(const AsyncSession &session)`

The session stores the selected DMA engine and engine-specific execution/event context.

## Scope of This Overview

This page summarizes the public wrapper surface. More detailed backend behavior, legality constraints, root-role conventions, and transport-specific restrictions are documented on the per-operation pages and in backend implementation code.

## See Also

- [TPUT](./TPUT.md)
- [TGET](./TGET.md)
- [TPUT_ASYNC](./TPUT_ASYNC.md)
- [TGET_ASYNC](./TGET_ASYNC.md)
- [TNOTIFY](./TNOTIFY.md)
- [TWAIT](./TWAIT.md)
- [TTEST](./TTEST.md)
- [TBROADCAST](./TBROADCAST.md)
- [TGATHER](./TGATHER.md)
- [TSCATTER](./TSCATTER.md)
- [TREDUCE](./TREDUCE.md)
