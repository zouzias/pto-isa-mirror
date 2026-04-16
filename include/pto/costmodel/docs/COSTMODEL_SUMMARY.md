# PTO Common-Layer Costmodel Summary

This note summarizes the current `__COSTMODEL` design in `pto-isa-costmodel` after the `pto_instr.hpp` cleanup.

The key change is that `include/pto/common/pto_instr.hpp` is now kept identical to upstream `pto-isa`. Costmodel-only PTO instruction tracing no longer lives in `include/pto/common`; it is routed through `include/pto/pto-inst.hpp` into `include/pto/costmodel/pto_instr.hpp`.

## Main Design Change

The costmodel path is now split into two layers:

- `include/pto/common` keeps only the common host-runtime compatibility changes needed by `__COSTMODEL`.
- `include/pto/costmodel` owns PTO API tracing, fake runtime support, and evaluator-side cycle modeling.

That means the common layer is no longer responsible for PTO instruction name tracing. The trace hook exists only in the costmodel wrapper header selected by `pto-inst.hpp`.

## Current Include Flow

When `__COSTMODEL` is defined:

1. `include/pto/pto-inst.hpp` includes `pto/costmodel/runtime_stub.hpp`.
2. `include/pto/pto-inst.hpp` includes the normal common headers such as `arch_macro.hpp` and `pto_tile.hpp`.
3. `include/pto/pto-inst.hpp` includes `pto/costmodel/pto_instr.hpp` instead of `pto/common/pto_instr.hpp`.
4. `include/pto/costmodel/pto_instr.hpp` wraps top-level PTO APIs with `PtoInstrScope` and records the PTO instruction name trace.

For non-costmodel builds, `include/pto/pto-inst.hpp` still includes `pto/common/pto_instr.hpp`.

## File-by-File Summary

### `type.hpp`

This is still the main host-runtime unification point.

- Uses the explicit host-runtime condition `defined(__CPU_SIM) || defined(__COSTMODEL)`.
- Makes `AICORE` empty in host runtime.
- Enables host-side assertion helpers through `PTO_CPU_ASSERT`.
- Adds host definitions for `half`, `aclFloat16`, `float16_t`, `float32_t`, and fallback `bfloat16_t`.
- Keeps the host-runtime condition explicit where the common headers need it.

Costmodel impact:

- `__COSTMODEL` can compile the common PTO headers as host C++.
- Host floating-point compatibility is sufficient for trace generation and argument capture.

### `pto_instr.hpp`

This file is now intentionally clean.

- `include/pto/common/pto_instr.hpp` matches upstream `pto-isa`.
- It no longer includes `pto/costmodel/trace.hpp`.
- It no longer defines `PTO_TRACE_CALL`.
- It no longer contains `__COSTMODEL`-specific PTO wrapper behavior.

Costmodel impact:

- The common PTO instruction declarations remain reusable and upstream-aligned.
- Costmodel-specific tracing is isolated outside `include/pto/common`.

### `pto_instr_impl.hpp`

This file still changes how implementations are selected.

- Removes the old bottom-of-file include block that directly pulled in `pto/costmodel/a2a3/*.hpp`.
- For A5 under `__COSTMODEL`, uses a dedicated include subset that pulls only the PTO/NPU instruction headers supported by the current costmodel path.
- Reuses normal PTO/NPU instruction templates instead of a separate common-layer implementation.

Costmodel impact:

- The backend runs the normal PTO instruction code on top of the costmodel fake runtime and fake CCE intrinsics.
- The supported API surface for costmodel remains intentionally narrower than full upstream PTO.

### `pto_tile.hpp`

This file keeps the host-runtime tile behavior used by costmodel.

- Keeps host logic spelled out directly as `__CPU_SIM || __COSTMODEL`.
- Removes the old `__COSTMODEL`-only `cycle`, `SetCycle`, and `GetCycle` members.
- Uses pointer-backed tile data in host runtime.
- Adds fixed internal storage arrays for host-runtime `Tile` objects and initializes `data_` to that storage in constructors.
- Makes host-runtime `data()` return the pointer value or pointer reference directly, so `TASSIGN` can redirect storage.

Costmodel impact:

- Tiles are only host-side storage and metadata carriers for trace generation.
- Cost is no longer stored on each tile.

### `constants.hpp`

This file adjusts custom pad-value handling for host runtime.

- Includes `type.hpp` first and uses the explicit host-runtime preprocessor condition.
- Fixes custom `PadValue` encoding and decoding to store custom bits in the high 32 bits.
- Uses `std::bit_cast` in host runtime and a union fallback elsewhere.
- Simplifies some host and bfloat handling through `PTO_HOST_BFLOAT_IS_HALF`.

Costmodel impact:

- Host-runtime and costmodel builds preserve custom pad constants correctly.

### `tassign_check.hpp`

- Uses `__CPU_SIM || __COSTMODEL` directly in host-only branches.
- Keeps host-runtime behavior as "skip static on-chip buffer capacity checks".

Costmodel impact:

- The common layer stays compile-friendly for tracing and host execution.

### `debug.h`

- Includes `type.hpp` first.
- Keeps host-only debug utilities guarded directly by `__CPU_SIM || __COSTMODEL`.

Costmodel impact:

- Host-side debug helpers remain usable in costmodel mode through the shared host-runtime path.

### `event.hpp`

- Removes `TRANDOM` from the `Op` enum and from `opPipeList`.

Costmodel impact:

- Reflects the reduced PTO API surface currently modeled by the costmodel branch.

### `memory.hpp`

- Removes `GetTileTypeName()`.

Costmodel impact:

- No direct tracing or evaluator behavior lives here. This is cleanup of common host-side support.

### `cpu_stub.hpp`

- Removes some CPU-stub helpers and macros such as `set_mask_norm`, `set_vector_mask`, `get_block_idx`, `get_subblockid`, and `get_subblockdim`.

Costmodel impact:

- No direct cycle-model behavior lives here. This is a reduction of unused host stub surface.

## Costmodel Files Outside `include/pto/common`

These files now carry the tracing logic that used to be mixed into the common PTO instruction header:

### `include/pto/pto-inst.hpp`

- Selects `pto/costmodel/runtime_stub.hpp` for `__COSTMODEL`.
- Selects `pto/costmodel/pto_instr.hpp` instead of `pto/common/pto_instr.hpp` for `__COSTMODEL`.

### `include/pto/costmodel/pto_instr.hpp`

- Acts as the costmodel-only wrapper around the clean common PTO instruction declarations.
- Reuses the `PTO_INSTR_HPP` include guard so it can be a drop-in replacement through `pto-inst.hpp`.
- Includes `pto/costmodel/trace.hpp`.
- Defines `PTO_TRACE_CALL(API, ...)`.
- Wraps top-level PTO APIs in `PtoInstrScope` so the evaluator can recover PTO instruction names such as `TASSIGN`, `TLOAD`, `TADD`, and `TSTORE`.

## What Was Removed From the Old Common-Layer Approach

Compared with earlier costmodel wiring, the following behavior is intentionally removed from `include/pto/common`:

- PTO trace hooks inside `common/pto_instr.hpp`
- costmodel-specific PTO wrapper selection inside the common header
- per-tile `cycle` state in `pto_tile.hpp`
- direct inclusion of `pto/costmodel/a2a3/*.hpp` from `pto_instr_impl.hpp`

The current split is:

- `include/pto/common` provides host-runtime compatibility and shared PTO declarations
- `include/pto/pto-inst.hpp` routes the build to the right front door
- `include/pto/costmodel` provides tracing, fake runtime support, and evaluator logic

## Practical Reading Order

If you want to understand the current costmodel flow starting from the public PTO include path, read files in this order:

1. `include/pto/pto-inst.hpp`
2. `include/pto/common/type.hpp`
3. `include/pto/common/pto_instr.hpp`
4. `include/pto/costmodel/pto_instr.hpp`
5. `include/pto/common/pto_instr_impl.hpp`
6. `include/pto/common/pto_tile.hpp`
7. `include/pto/costmodel/trace.hpp`
8. `include/pto/costmodel/runtime_stub.hpp`
9. `include/pto/costmodel/common/arch_select.hpp`
10. `include/pto/costmodel/a2a3/cce_stub.hpp` or `include/pto/costmodel/a5/cce_stub.hpp`
11. `include/pto/costmodel/evaluator/trace_evaluator.hpp`
12. `include/pto/costmodel/evaluator/cce_evaluator.hpp`

## Bottom Line

The important cleanup is:

- keep `include/pto/common/pto_instr.hpp` identical to upstream
- keep common-layer changes focused on host-runtime compatibility
- move PTO instruction name tracing into `include/pto/costmodel/pto_instr.hpp`
- let `include/pto/pto-inst.hpp` select the costmodel wrapper only when `__COSTMODEL` is defined

The actual cost estimation logic remains outside `include/pto/common`, under `include/pto/costmodel/`.
