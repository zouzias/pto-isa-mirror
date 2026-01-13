# Pipe, TPUSH, and TPOP

This document describes the cross-core GM FIFO helpers in `include/pto/common/pto_pipe.hpp`:

- `pto::Pipe<...>`: ring-buffer synchronization for a GM FIFO shared between CUBE and VECTOR cores.
- `pto::TPUSH(...)`: store a tile into a GM FIFO slot and signal the consumer.
- `pto::TPOP(...)`: wait for a GM FIFO slot and load a tile from it.

These helpers are designed to eliminate manual `set_flag(...)` / `wait_flag(...)` and `wait_flag_dev(...)` / `ffts_cross_core_sync(...)` sequences in kernels, replacing them with higher-level, readable primitives.

## Building blocks

### `pto::PipeEvent<SrcPipe, DstPipe>`

`PipeEvent` is a lightweight event token that directly represents a pipeline-pair flag dependency (the legacy `set_flag` / `wait_flag` pattern), but exposes the common event API:

- `Wait()`
- `Record()`
- assignment from `RecordEvent` (`evt = OP(...)`)

This is useful when your dependency is naturally described in terms of *pipelines* rather than *Ops*.

### `pto::CvFlagEvent<SignalPipe, FlagId>`

`CvFlagEvent` is a lightweight event token for CV cross-core flags (FFTS-style):

- `Wait()` waits on `wait_flag_dev(FlagId)`.
- `Record()` signals via `ffts_cross_core_sync(SignalPipe, ...)`.

It is independent of `pto::Op` mapping.

## `pto::Pipe<ReadyFlag, ConsumedFlag, Depth, Period, ProdRole, ConsRole>`

`Pipe` models a circular FIFO protocol between two cores:

- `ReadyFlag`: producer → consumer (“data ready”)
- `ConsumedFlag`: consumer → producer (“space released”)
- `Depth`: FIFO depth (e.g. `2` for double-buffering)
- `Period`: reduce cross-core traffic by waiting/signaling every `Period` items (must be `<= Depth`)
- `ProdRole` / `ConsRole`: logical roles (`CoreType::CUBE` / `CoreType::VECTOR`), used to choose the signal pipeline

`Pipe` provides two stateful endpoints:

- `Pipe::Producer` with `alloc()` and `record()` and a private `iter`
- `Pipe::Consumer` with `wait()` and `free()` and a private `iter`

The default pipeline mapping in `include/pto/common/pto_pipe.hpp` is:

- Producer signal pipe: CUBE → `PIPE_FIX`, VECTOR → `PIPE_MTE3`
- Consumer signal pipe: CUBE → `PIPE_MTE2`, VECTOR → `PIPE_MTE2`

## `TPUSH` and `TPOP`

### `TPUSH`

`TPUSH(prod, tile, fifo_base, events...)`:

1. `TSYNC(events...)` (optional intra-core dependency)
2. `prod.alloc()` (wait for space when needed)
3. `TSTORE(...)` to the ring slot address
4. `prod.record()` (signal “ready”)

Returns a `RecordEvent` that can be assigned into an event token to express subsequent ordering.

### `TPOP`

`TPOP(cons, tile, fifo_base, events...)`:

1. `TSYNC(events...)` (optional intra-core dependency)
2. `cons.wait()` (wait for “ready”)
3. `TLOAD(...)` from the ring slot address
4. `cons.free()` (signal “consumed” / space released)

Returns a `RecordEvent` that can be assigned into an event token to express subsequent ordering.

## Notes and constraints

- `Pipe` is intended for device builds; CPU simulator builds treat cross-core operations as no-ops.
- `ReadyFlag` and `ConsumedFlag` are encoded as 4-bit CV flag IDs (0–15).
- `TPUSH` / `TPOP` currently assume each FIFO element is a single contiguous tile stored in ND layout.

