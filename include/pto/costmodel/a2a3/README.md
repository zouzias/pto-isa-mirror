# A2A3 Mocker Notes

This directory contains the A2/A3-specific fake intrinsic layer used by `__COSTMODEL`.

Unlike A5, A2A3 does not need a special reduced include list in
[`pto_instr_impl.hpp`](../../common/pto_instr_impl.hpp). The A2A3 mocker works by making the original A2A3 PTO
headers believe they are compiling in a valid device-like environment, while replacing the device/compiler/runtime
pieces with host-side fake headers and trace stubs.

In practice, A2A3 mocker mode can include the full A2A3 PTO instruction surface from one branch:

- `92` A2A3 instruction headers are included from the single A2A3 branch in
  [`include/pto/common/pto_instr_impl.hpp`](../../common/pto_instr_impl.hpp).
- There is no separate `#ifdef __COSTMODEL` A2A3 include subset.

## Illustration

```text
                          user code includes pto/pto-inst.hpp
                                       |
                          __COSTMODEL selects runtime_stub.hpp
                                       |
                                       v
                        include/pto/costmodel/runtime_stub.hpp
                                       |
             +-------------------------+--------------------------+
             |                         |                          |
             v                         v                          v
   common/qualifiers.hpp      common/aclrt_stub.hpp    common/runtime_util.hpp
   - blank device qualifiers  - fake aclrt APIs        - fake flags / trap / helpers
   - identity tile ptr        - malloc/memcpy/free     - host utility shims
                                       |
                                       v
                           common/arch_select.hpp
                                       |
                                       v
                             a2a3/cce_stub.hpp
                 - fake A2A3 intrinsic names with host definitions
                 - some no-op compile shims
                 - many trace-recording stubs
                                       |
                                       v
                          original PTO A2A3 headers compile
                    (TLoad/TStore/TMatmul/TAdd/... unchanged)
                                       |
                                       v
                           pto/common/pto_instr.hpp
                wraps PTO API calls in PtoInstrScope under __COSTMODEL
                                       |
                                       v
                             trace.hpp collects PTO + CCE trace
```

## What “Fake Headers” Means Here

The mocker does not replace the original PTO instruction headers with separate mocker copies. Instead, it injects a
small host-side compatibility layer before those original headers are parsed.

The important fake-header pieces are:

- [`../runtime_stub.hpp`](../runtime_stub.hpp)
  The umbrella entry point included from [`pto-inst.hpp`](../../../pto-inst.hpp) when `__COSTMODEL` is enabled.
- [`../common/qualifiers.hpp`](../common/qualifiers.hpp)
  Turns device-only qualifiers such as `__gm__`, `__ubuf__`, `__cbuf__`, `__cc__`, `AICORE`, and `__tf__` into empty
  host-safe macros, and defines `__cce_get_tile_ptr(x)` as identity.
- [`../common/aclrt_stub.hpp`](../common/aclrt_stub.hpp)
  Fakes the minimal ACL runtime API on host using plain `malloc`, `free`, `memcpy`, and `memset`.
- [`../common/runtime_util.hpp`](../common/runtime_util.hpp)
  Fakes low-level helpers like `__pto_set_flag`, `__pto_wait_flag`, `trap`, and some small utility functions.
- [`../common/arch_select.hpp`](../common/arch_select.hpp)
  Selects [`cce_stub.hpp`](./cce_stub.hpp) when `__NPU_ARCH__ == 2201`.
- [`cce_stub.hpp`](./cce_stub.hpp)
  Defines the intrinsic/function names that the original A2A3 PTO headers expect to call.

So the idea is:

1. Keep the real PTO A2A3 headers.
2. Fake the environment they expect.
3. Replace the low-level calls with host stubs.
4. Record the calls into trace data instead of executing real NPU work.

## Why A2A3 Is Cleaner Than A5

A2A3 PTO headers are mostly built around older free-function CCE intrinsics and simpler control helpers. The mocker
can satisfy enough of that environment with:

- blank qualifier macros
- a handful of fake enums/constants
- helper functions like `get_ctrl(...)` and `get_imm(...)`
- a large intrinsic-name shim table in [`cce_stub.hpp`](./cce_stub.hpp)

That is why the single A2A3 include list in
[`pto_instr_impl.hpp`](../../common/pto_instr_impl.hpp#L18) works for mocker mode without a special branch.

By contrast, A5 pulls in more register-model, SIMT, and launch/runtime concepts, so its host compatibility layer is
not yet broad enough to admit the full A5 header set directly.

## What `cce_stub.hpp` Actually Does

[`cce_stub.hpp`](./cce_stub.hpp) mixes three kinds of fake definitions:

### 1. Compile-only placeholders

These exist only so the original PTO headers can parse and instantiate:

- constants and enums such as `QuantMode_t`, `VA0..VA7`, `DSB_UB`, `PIPE_FIX`
- helper functions like `sbitset0`, `sbitset1`, `get_ctrl(...)`, `get_imm(...)`

### 2. Traceable fake intrinsics

These replace real device intrinsics with host functions that only record metadata:

- memory movement:
  `copy_gm_to_cbuf`, `copy_cbuf_to_gm`, `copy_gm_to_ubuf_align_*`, `copy_ubuf_to_gm_align_*`
- matmul / control:
  `mad`, `set_flag`, `wait_flag`, `wait_flag_dev`, `set_cmpmask`, `set_vector_mask`
- vector arithmetic and math:
  `vadd`, `vsub`, `vaxpy`, `vln`, many `vcmp*`, selected `vconv*`, `vsel`

These emit `RecordCceCall(...)` into the shared trace state in [`../trace.hpp`](../trace.hpp).

### 3. No-op compatibility shims

Some names are currently only empty functions. They make host compilation succeed, but they do not record useful
trace data. Examples include:

- memory / matrix helpers:
  `copy_cbuf_to_bt`, `copy_cbuf_to_fbuf`, `copy_matrix_cc_to_cbuf`
- setup/config helpers:
  `set_ctrl`, `set_deqscale`, `set_fmatrix`, `set_nd_para`, `set_quant_pre`
- some vector families:
  `vbitsort`, `vbrcb`, `vcadd`, `vcgadd`, `vcgmax`, `vcgmin`, `vcmax`, `vcmin`
- many conversion variants:
  a large part of the `vconv_*` surface is still empty
- miscellaneous:
  `vgather`, `vgatherb`, `vmrgsort4`, `vreducev2`

So A2A3 mocker coverage is broad enough to compile the full include surface, but not every low-level intrinsic path is
equally rich in trace fidelity.

## Trace Flow

When a PTO API is called in mocker mode:

1. [`pto/common/pto_instr.hpp`](../../common/pto_instr.hpp) wraps it in `PtoInstrScope`.
2. The original PTO implementation runs unchanged.
3. Whenever that implementation calls a fake intrinsic from [`cce_stub.hpp`](./cce_stub.hpp), the stub records a
   `CceCallRecord` into [`../trace.hpp`](../trace.hpp).
4. The trace ends up grouped by top-level PTO API call, with direct raw calls stored separately.

This is why the mocker can answer questions like:

- which PTO API ran
- which low-level A2A3 intrinsics it emitted
- what key parameters were passed

without simulating the actual numeric result.

## Worked Example: `tests/a2a3/testcase/tmatmul/tmatmul_kernel.cpp`

[`tmatmul_kernel.cpp`](../tests/a2a3/testcase/tmatmul/tmatmul_kernel.cpp) is a good example of how mocker handles
tile types and addresses:

```cpp
using LeftTile = TileLeft<float, 16, 16, 16, 16>;
using RightTile = TileRight<float, 16, 16, 16, 16>;
using AccTile = TileAcc<float, 16, 16, 16, 16>;

LeftTile aTile;
RightTile bTile;
AccTile cTile;

TASSIGN(aTile, 0x0);
TASSIGN(bTile, 0x20000);
TASSIGN(cTile, 0x40000);

TMATMUL(cTile, aTile, bTile);
```

### What the actual runtime types are in mocker

Under `__COSTMODEL`, [`type.hpp`](../../common/type.hpp) defines `PTO_HOST_RUNTIME`. In host runtime,
[`pto_tile.hpp`](../../common/pto_tile.hpp) switches tile storage to plain host pointers:

- `Tile::TileDType = DType *`
- the tile object stores a field `data_` of that pointer type

So in this test:

- `LeftTile::TileDType` is `float *`
- `RightTile::TileDType` is `float *`
- `AccTile::TileDType` is `float *`

The tile categories still matter for compile-time checks and PTO dispatch:

- `LeftTile` has `Loc == TileType::Left`
- `RightTile` has `Loc == TileType::Right`
- `AccTile` has `Loc == TileType::Acc`

But in mocker they do not become different machine-level pointer types. They are all plain `float *` at runtime.

### What `TASSIGN` stores

[`TASSIGN_IMPL`](../../npu/a2a3/TAssign.hpp) takes the integer address, casts it through `uintptr_t`, then stores it
as `typename T::TileDType`.

That means the three calls above produce:

- `aTile.data_ == reinterpret_cast<float *>(0x0)`
- `bTile.data_ == reinterpret_cast<float *>(0x20000)`
- `cTile.data_ == reinterpret_cast<float *>(0x40000)`

These values are not real allocated CPU buffers in this test. They are just pointer-shaped values stored into the tile
objects.

If a tile is never `TASSIGN`'d, host runtime falls back to its internal storage buffer inside the tile object. But this
test explicitly overwrites all three tile pointers with the fake addresses above.

### What `__cce_get_tile_ptr` does here

In mocker, [`../common/qualifiers.hpp`](../common/qualifiers.hpp) defines:

```cpp
#define __cce_get_tile_ptr(x) x
```

So it performs no translation. It simply returns the value already stored in `data_`.

The device-space qualifiers are also blank in mocker:

- `__ca__`
- `__cb__`
- `__cc__`
- `__ubuf__`
- `__cbuf__`
- `__biasbuf__`

So a line like this in [`TMatmul.hpp`](../../npu/a2a3/TMatmul.hpp):

```cpp
__ca__ typename TileLeft::DType *a = (__ca__ typename TileLeft::DType *)__cce_get_tile_ptr(aMatrix);
```

effectively becomes:

```cpp
float *a = (float *)aMatrix;
```

The same applies to the right-hand and accumulator tiles:

- `a == reinterpret_cast<float *>(0x0)`
- `b == reinterpret_cast<float *>(0x20000)`
- `c == reinterpret_cast<float *>(0x40000)`

So yes: `__cce_get_tile_ptr` returns exactly the address that `TASSIGN` stored.

### Why the fake addresses do not crash this test

[`TMATMUL_IMPL`](../../npu/a2a3/TMatmul.hpp) eventually calls:

```cpp
mad(c, a, b, m, k, n, phase, kDirectionAlign, cmatrixSource, cmatrixInitVal);
```

In mocker, [`cce_stub.hpp`](./cce_stub.hpp) replaces `mad(...)` with a trace stub that only records arguments through
`RecordCceCall(...)`. It does not dereference `a`, `b`, or `c`, and it does not perform any numeric computation.

That is why the fake addresses `0x0`, `0x20000`, and `0x40000` are safe in this test: they are used as symbolic
address labels, not as real readable/writable buffers.

This is different from data-touching PTO instructions such as elementwise kernels that directly index the returned
pointer. Those paths need a real host buffer or the tile's default internal storage.

### What survives into the trace

Running the mocker test shows:

```text
executed_pto:
  [0] TASSIGN
  [1] TASSIGN
  [2] TASSIGN
  [3] TMATMUL
    cce_calls:
      [0] mad (c=0x40000, a=0x0, b=0x20000, m=0x10, k=0x10, n=0x10, ...)
```

So the trace keeps the exact pointer values written by `TASSIGN`, plus the matrix shape metadata derived from the tile
shapes:

- `m = 16`
- `k = 16`
- `n = 16`

### Bottom line for this example

- The tile runtime pointer type in mocker is just `float *`.
- `TASSIGN` writes the integer literal into that stored pointer.
- `__cce_get_tile_ptr` returns the same value unchanged.
- `TMATMUL` re-tags the pointer as left/right/acc via empty qualifiers only.
- `mad` records the pointer values but does not dereference them.

So in this particular mocker test, the addresses are intentionally fake but still "correct" relative to what PTO and
the trace system are supposed to observe.

## Bottom Line

The A2A3 mocker is “clean” because it relies on a successful fake-environment strategy:

- fake the compiler qualifiers
- fake the runtime
- fake the intrinsic symbols
- keep the original A2A3 PTO headers unchanged

That is enough for host compilation of the whole A2A3 PTO include surface, even though some individual low-level
intrinsics are still only no-op compatibility shims rather than full trace stubs.
