# A3 vs A5 Differences (auto-mode kernels)

Source-grounded enumeration of differences between A3 (`PTO_NPU_ARCH_A2A3`,
`__NPU_ARCH__ == 2201`) and A5 (`PTO_NPU_ARCH_A5`, `__NPU_ARCH__ == 3101 || 3510`)
that matter when generating, reviewing, or debugging **auto-mode** kernels.

Companion to [repo_kernel_map.md](repo_kernel_map.md),
[known_good_kernel_examples.md](known_good_kernel_examples.md),
[auto_mode_bad_patterns.md](auto_mode_bad_patterns.md),
[qualifier_reference.md](qualifier_reference.md),
[tile_type_reference.md](tile_type_reference.md), and
[external_context/pr_852_notes.md](external_context/pr_852_notes.md).

CPU-sim, cost-model, and Kirin paths are referenced **only** when they explain
arch_macro logic or per-arch `#ifdef`s. Their behavior is **never** evidence
that an A3/A5 auto-mode pattern is valid.

Confidence labels: **Known** (cited path), **Inferred** (derived from cited
paths), **Assumption** (not in source — confirm), **Unknown** (referenced but
not yet verified).

---

## 1. Arch macros and target compile flags

Defined: [include/pto/common/arch_macro.hpp:14-27](../include/pto/common/arch_macro.hpp#L14-L27). (Known)

| Aspect | A3 | A5 |
|---|---|---|
| `__NPU_ARCH__` value | `2201` | `3101` or `3510` |
| Repo arch macro | `PTO_NPU_ARCH_A2A3` | `PTO_NPU_ARCH_A5` |
| Comm support | Yes | Yes; `__NPU_ARCH__ == 3510` adds `PTO_URMA_SUPPORTED` (Known: [arch_macro.hpp:18-20](../include/pto/common/arch_macro.hpp#L18-L20); used by [include/pto/comm/a5/async/](../include/pto/comm/a5/async/)) |
| Vec compile flag | `--cce-aicore-arch=dav-c220-vec` (Known: [tests/npu/a2a3/src/st/testcase/CMakeLists.txt:14](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L14)) | `--cce-aicore-arch=dav-c310-vec` (Inferred from [docs/auto_mode/Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md): example `--cce-aicore-arch=dav-c310-vec`) |
| Cube compile flag | `--cce-aicore-arch=dav-c220-cube` (Known: [tests/npu/a2a3/src/st/testcase/CMakeLists.txt:43](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L43)) | **Unknown** — likely `dav-c310-cube`, but not visible in [tests/npu/a5/src/st/testcase/CMakeLists.txt](../tests/npu/a5/src/st/testcase/CMakeLists.txt). Confirm with the user. |
| Mixed (vec+cube) flag | `--cce-aicore-arch=dav-c220` (Known: [tests/npu/a2a3/src/st/testcase/CMakeLists.txt:73](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L73)) | **Unknown** |
| Auto-mode flag (both) | `--cce-enable-pto-passes` plus `__PTO_AUTO__` defined; auto mode requires `-O2` (Known: [docs/auto_mode/Auto_Mode_Overview.md](../docs/auto_mode/Auto_Mode_Overview.md), [demos/auto_mode/baseline/add/CMakeLists.txt:65-67](../demos/auto_mode/baseline/add/CMakeLists.txt#L65-L67)) | Same |
| Common toolchain shape | `bisheng -c -x cce -O2 --cce-aicore-only --cce-enable-pto-passes ... -std=c++17` per the auto-mode overview (Known) | Same |

The cube macro `__DAV_C220_CUBE__` ([include/pto/common/memory.hpp:74](../include/pto/common/memory.hpp#L74)) is named for A3; whether A5 has an equivalent (e.g., `__DAV_C310_CUBE__`) gating `Bias` storage is **Unknown**. See §6.

---

## 2. Include-tree differences

`include/pto/npu/a2a3/` and `include/pto/npu/a5/` are independent sub-trees.
A5 is **not** an include-of-A3 wrapper.

A3-only files (Known — `comm a3.txt a5.txt` of header lists):

```
TBitwiseSOp.hpp        TCI.hpp            TDequant.hpp
TFmod.hpp              TFmodS.hpp         TMaxS.hpp
TPartArgOp.hpp         TPartOp.hpp        TRowMax.hpp
TRowMin.hpp            TRowReduceIdxOps.hpp  TRowReduceOps.hpp
TRowSum.hpp
```

A5-only files (Known):

```
datatype.hpp           MGather.hpp        MScatter.hpp
TAndS.hpp              Tci.hpp (lowercase)
TDeQuant.hpp (camel)   TFMod.hpp / TFModS.hpp (different case)
TGetScaleAddr.hpp      THistogram.hpp     TMaxs.hpp (different case)
TOrS.hpp               TPartArgBinOps.hpp / TPartArgMax.hpp / TPartArgMin.hpp
TPartBinOps.hpp        TRandom.hpp
TRowReduce.hpp         TRowReduceIdx.hpp  TRsqrt.hpp
TShlS.hpp / TShrS.hpp  TXorS.hpp
utils.hpp
```

Filename casing differences worth noting (Known):
- A3 uses `TCI.hpp`, A5 uses `Tci.hpp` ([pto_instr_impl.hpp:131 vs 215](../include/pto/common/pto_instr_impl.hpp#L131)).
- A3 uses `TDequant.hpp`, A5 uses `TDeQuant.hpp`.
- A3 uses `TFmod.hpp` / `TFmodS.hpp`, A5 uses `TFMod.hpp` / `TFModS.hpp`.
- A3 uses `TMaxS.hpp`, A5 uses `TMaxs.hpp`. Same for `TMins.hpp`.
- A3 uses `TRowReduceIdxOps.hpp` and `TRowSum.hpp`/`TRowMax.hpp`/`TRowMin.hpp`; A5 splits into `TRowReduce.hpp` and `TRowReduceIdx.hpp`.

> **Implication**: Cross-platform code that includes per-arch headers must use `pto/pto-inst.hpp` and let `pto_instr_impl.hpp` pick the right tree. Direct `#include "pto/npu/a2a3/TCI.hpp"` will not compile on A5.

A5-only top-level structure: A5 has its own `datatype.hpp` (not present in A3 — A3 uses common types) and `utils.hpp` (A3 has it under different name conventions). (Known)

`custom/` subdirectory (per-arch helper sets, Known):

| File | A3 (`a2a3/custom/`) | A5 (`a5/custom/`) |
|---|---|---|
| `TSync_Custom.hpp` | yes | yes |
| `TSyncCVID.hpp` | yes | yes |
| `common.hpp` | no | yes |
| `Div754.hpp` | no | yes (high-precision divide) |
| `TExp_Custom.hpp` | no | yes |
| `TFmodRemHp.hpp` | no | yes (high-precision fmod/rem) |
| `TLog_Custom.hpp` | no | yes |
| `TSqrtHp.hpp` | no | yes |

> A5 has additional high-precision math customizations (`Hp` = high precision). A3 does not. **Inferred**: any A3 kernel that wants the same high-precision behavior must rewrite the math directly in PTO instructions; the `*_Custom`/`*Hp` helpers are A5-only.

---

## 3. Instruction availability and naming differences

### 3.1 Cube / GEMM family (Known: comparing [a2a3/TMatmul.hpp](../include/pto/npu/a2a3/TMatmul.hpp) and [a5/TMatmul.hpp](../include/pto/npu/a5/TMatmul.hpp))

| Instruction | A3 (`a2a3/TMatmul.hpp`) | A5 (`a5/TMatmul.hpp`) |
|---|---|---|
| `TMATMUL_IMPL` | line 153 | line 160 |
| `TMATMUL_ACC_IMPL` | line 166, 180 | line 175, 191 |
| `TMATMUL_BIAS_IMPL` | line 187 | line 198 |
| `TMATMUL_MX_IMPL` | **NOT PRESENT** | line 259, 275, 291 |

`TMATMUL_MX` is **A5-only**. (Known.) Used by the A5 MX FP4/FP8 GEMM kernel at [tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp](../tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp).

### 3.2 Gather / scatter family

| Instruction | A3 | A5 |
|---|---|---|
| `TGATHER` / `TGATHERB` / `TSCATTER` | yes ([a2a3/TGather.hpp](../include/pto/npu/a2a3/TGather.hpp), [a2a3/TGatherB.hpp](../include/pto/npu/a2a3/TGatherB.hpp), [a2a3/TScatter.hpp](../include/pto/npu/a2a3/TScatter.hpp)) | yes ([a5/TGather.hpp](../include/pto/npu/a5/TGather.hpp), [a5/TGatherB.hpp](../include/pto/npu/a5/TGatherB.hpp), [a5/TScatter.hpp](../include/pto/npu/a5/TScatter.hpp)) |
| `MGATHER` (multi-table) | **NOT PRESENT** | yes — [a5/MGather.hpp](../include/pto/npu/a5/MGather.hpp), e.g. `MGatherRowImpl` at line 164 |
| `MSCATTER` (multi-table) | **NOT PRESENT** | yes — [a5/MScatter.hpp](../include/pto/npu/a5/MScatter.hpp), e.g. `MScatterRowImpl` at line 274 |

`TSCATTER` IMPL bodies differ (Known: A3 line 56, A5 line 136 — different file sizes; semantics may diverge — Inferred Unknown).

### 3.3 Reduction family

A3 has dedicated header per reduction (`TRowSum.hpp`, `TRowMax.hpp`, `TRowMin.hpp`, `TColMax.hpp`, etc.). A5 collapses them into `TRowReduce.hpp` and `TRowReduceIdx.hpp` (Known: see §2 list). The user-facing PTO instructions (`TROWSUM`, `TROWMAX`, …) are routed through [pto_instr.hpp](../include/pto/common/pto_instr.hpp) and dispatched per-arch by [pto_instr_impl.hpp](../include/pto/common/pto_instr_impl.hpp) — kernel-side calls do not need to know.

### 3.4 Quant family

`QuantType` enum differs (Known):

| arch | enum members | source |
|---|---|---|
| A3 | `INT8_SYM`, `INT8_ASYM` | [a2a3/TQuant.hpp:21-25](../include/pto/npu/a2a3/TQuant.hpp#L21-L25) |
| A5 | `MXFP8`, `INT8_SYM`, `INT8_ASYM` | [a5/TQuant.hpp:23-28](../include/pto/npu/a5/TQuant.hpp#L23-L28) |

A5 has an additional `MXFP8` quant mode tied to MX FP8 microscaling. A3 cannot do MXFP8 quant. (Known.)

A5's `TQuant.hpp` is much larger (~1000+ lines vs ~125 on A3, Known: [a5/TQuant.hpp:992, 1013](../include/pto/npu/a5/TQuant.hpp#L992) shows TQUANT_IMPL appearing far down) and includes additional helpers like `AbsReduceMax_Naive` ([a5/TQuant.hpp:35](../include/pto/npu/a5/TQuant.hpp#L35)). **Inferred**: A5 quant has different semantics for the MX path; the INT8 path may also differ. Treat them as distinct implementations.

### 3.5 Misc A5 extras

A5-only headers without A3 equivalents (Known: §2 list):
- `TGetScaleAddr.hpp` — get the GM address of a scale buffer (used by quant/MX paths).
- `THistogram.hpp` — histogram primitive.
- `TRandom.hpp` — random-number primitive.
- `TRsqrt.hpp` — reciprocal square root (A3 likely composes via `TRECIP` + `TSQRT`; **Inferred**, not verified).
- `TPartArgMax.hpp` / `TPartArgMin.hpp` — split into separate files on A5 (A3 has them inside `TPartArgOp.hpp`).
- `TAndS.hpp`, `TOrS.hpp`, `TXorS.hpp`, `TShlS.hpp`, `TShrS.hpp` — separate scalar variants on A5 (A3 collapses these into `TBitwiseSOp.hpp`).

The user-facing API names (`TANDS`, `TORS`, `TXORS`, `TSHLS`, `TSHRS`, `TPARTARGMAX`, `TPARTARGMIN`, `TRSQRT`, etc.) are dispatched per-arch through [pto_instr_impl.hpp](../include/pto/common/pto_instr_impl.hpp).

---

## 4. Tile / layout / alignment differences

### 4.1 `TileLeft` base layout — A3 RowMajor, A5 ColMajor (significant)

Defined: [include/pto/common/pto_tile.hpp:1681-1699](../include/pto/common/pto_tile.hpp#L1681-L1699). (Known)

```cpp
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
using TileLeft = Tile<TileType::Left, T, R, C, BLayout::RowMajor, ...>;       // <-- A3
using TileLeftCompact = Tile<TileType::Left, T, R, C, BLayout::RowMajor, ..., CompactMode::Normal>;
#endif

#if (!defined(PTO_NPU_ARCH_A2A3) && !defined(PTO_NPU_ARCH_KIRINX90)) || defined(__CPU_SIM)
using TileLeft = Tile<TileType::Left, T, R, C, BLayout::ColMajor, ...>;       // <-- A5
using TileLeftCompact = Tile<TileType::Left, T, R, C, BLayout::ColMajor, ..., CompactMode::Normal>;
#endif
```

This **resolves the open item** flagged in [tile_type_reference.md §3.1, §12](tile_type_reference.md):
- **A3**: `TileLeft<...>` has `BLayout::RowMajor`.
- **A5**: `TileLeft<...>` has `BLayout::ColMajor`.

`TileLeftCompact` follows the same split. `TileRight`, `TileAcc`, `TileLeftScale`, `TileRightScale` are defined unconditionally — same on both archs. (Known: [pto_tile.hpp:1701-1732](../include/pto/common/pto_tile.hpp#L1701-L1732).)

> **Practical impact**: A GEMM whose A operand uses `TileLeft<U, M, K, ...>` will get different tile storage layout on A3 vs A5. Down-stream `TLOAD`/`TMOV` choices and the matching `TileMatA` (the L1-resident `Tile<TileType::Mat, ...>` source) need different `BLayout` values to feed `TMOV` correctly. Both [tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp:51](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L51) and [tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp:53](../tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L53) declare their A-side `TileMat` as `BLayout::ColMajor` despite the `TileLeft` divergence — Inferred that the per-arch `TMOV` and TileLeft alias take care of the swap internally.

### 4.2 GEMM block alignment formula

Compare:

```cpp
// A3 — tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp:128
constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);            // 32 bytes / sizeof(U)

// A5 — tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp:32
constexpr int blockAlign = (sizeof(AType) == 1) ? 32 : 16;
```

(Known.) A5 special-cases 1-byte types (INT8 / FP8 / FP4-pair packed forms) to `blockAlign = 32`; everything else (FP16/BF16/FP32) uses `blockAlign = 16`. A3 uniformly divides `C0_SIZE_BYTE = 32` by `sizeof(U)` ([include/pto/common/constants.hpp:34](../include/pto/common/constants.hpp#L34)).

For FP16 (sizeof == 2), both formulas give `16` — agree.
For INT8 (sizeof == 1), both give `32` — agree.
For FP32 (sizeof == 4), A3 gives `8` while A5 gives `16` — **disagree**.

> **Implication**: copying the A3 `blockAlign = C0_SIZE_BYTE / sizeof(U)` formula into an A5 GEMM produces wrong alignment for FP32 and any other 4-byte type. Use the A5 conditional. See [auto_mode_bad_patterns.md §4.3](auto_mode_bad_patterns.md).

### 4.3 GEMM `BiasTile` divergence

| arch | declaration ([source](#)) |
|---|---|
| A3 | `using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;` ([tmatmul_kernel.cpp:59](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L59)) — element type = `B` (bias's own type), valid col = `validN`. |
| A5 | `using BiasTile = Tile<TileType::Bias, OutType, 1, N, BLayout::RowMajor, 1, N>;` ([tmatmul_kernel.cpp:60](../tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L60)) — element type = **`OutType`** (output type, not bias's), valid col = padded `N` (not `validN`). |

(Known.) **Inferred**: A5's bias path expects bias to be in the output tile's element type (e.g., the matmul promotes the bias into FP32 if `OutType` is FP32), and uses the padded valid column. A3 keeps bias in its native type.

> **Implication**: when porting an A3 GEMM kernel to A5, do not blindly copy the `BiasTile` declaration. A5 needs `OutType` and padded `N`.

### 4.4 `Tile::TileDType` definition diverges between A3 and A5 in auto mode

[include/pto/common/pto_tile.hpp:1530-1540](../include/pto/common/pto_tile.hpp#L1530-L1540) (Known):

```cpp
#ifdef __PTO_AUTO__
#if defined(PTO_NPU_ARCH_A2A3)
    using TileDType = typename MemoryQualifier<Loc, DType>::type tile_size(Rows*Cols);
#else  // <-- A5 (and other non-A3)
    using TileDType = std::conditional_t<Loc == TileType::Bias,
                                         typename MemoryQualifier<Loc, DType>::type,
                                         typename MemoryQualifier<Loc, DType>::type tile_size(Rows*Cols)>;
#endif
#else
    using TileDType = typename MemoryQualifier<Loc, DType>::type;   // pointer (manual)
#endif
```

> **A5 special-cases `TileType::Bias`**: skips the `tile_size(Rows*Cols)` modifier and uses the bare memory-qualified type. **Inferred**: bisheng-CCE's A5 backend treats bias storage differently (e.g., pre-allocated fixed register file) and the `tile_size` allocator marker would be wrong.
> **A3 does not have this exception**: `Tile<TileType::Bias, ...>::TileDType` carries `tile_size(Rows*Cols)` like every other tile type.

### 4.5 `ConvTile<Loc, T, BufferSize_, Layout, Shape>` — same on both

[pto_tile.hpp:1096](../include/pto/common/pto_tile.hpp#L1096) is unconditional. (Known.) `BufferSize_` is element count on both A3 and A5 (see [tile_type_reference.md §10.2](tile_type_reference.md) and [auto_mode_bad_patterns.md §5.6](auto_mode_bad_patterns.md)).

### 4.6 `ConvTile` extra fields — A5 only

[pto_tile.hpp:1333-1350, 1373-1376](../include/pto/common/pto_tile.hpp#L1333-L1350) (Known):

```cpp
#ifndef PTO_NPU_ARCH_A2A3
    PTO_INTERNAL void SetDstStride(uint16_t dstStride) { ... }
    PTO_INTERNAL uint16_t GetDstStride() const { ... }
    PTO_INTERNAL void SetDstMposition(uint16_t dstMposition) { ... }
    PTO_INTERNAL uint16_t GetDstMposition() const { ... }
#endif
...
#ifndef PTO_NPU_ARCH_A2A3
    uint16_t dstStride_ = 0;
    uint16_t dstMposition_ = 0;
#endif
```

So `ConvTile` on A5 (and any non-A3) carries an additional `dstStride_` / `dstMposition_` pair plus accessors. A3 does not. **Inferred**: A5's `TTrans*` paths use these fields for non-trivial destination strides; A3 does not need them.

### 4.7 `TileConfig` constants and `Tile` compile-time asserts — same

[pto_tile.hpp:947-956, 1505-1524](../include/pto/common/pto_tile.hpp#L947-L956). (Known.) Both archs share `alignedSize=32`, `fractalABSize=512`, `fractalCSize=1024`, `fractalMxSize=32`, and the same Tile-shape asserts.

`SLayout::RowMajor + 512` for A-side and `SLayout::ColMajor + 512` for B-side are used unconditionally in both A3 and A5 `tmatmul_kernel.cpp`s (Known: [a3 tmatmul:51](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L51), [a5 tmatmul:53-54](../tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L53-L54)). Whether `512` remains correct for FP4/FP8 on A5 — see [auto_mode_bad_patterns.md §4.4](auto_mode_bad_patterns.md). Inferred Unknown.

---

## 5. Datatype differences

### 5.1 A5-only element types

[include/pto/npu/a5/datatype.hpp:14-90](../include/pto/npu/a5/datatype.hpp#L14-L90) (Known) defines `TypeGet<...>` specializations under `#if defined(__DAV_VEC__)`:

```
bfloat16_t      -> vector_bf16
float8_e5m2_t   -> vector_f8e5m2
float8_e4m3_t   -> vector_f8e4m3
hifloat8_t      -> vector_hif8
float8_e8m0_t   -> vector_f8e8m0
float4_e1m2x2_t -> vector_f4e1m2x2
float4_e2m1x2_t -> vector_f4e2m1x2 (Inferred — pattern continues)
plus standard u64/s64/u32/s32/f32/u16/f16/s16/...
```

> A5 supports **MX FP8** (`float8_e4m3_t`, `float8_e5m2_t`, `float8_e8m0_t`, `hifloat8_t`) and **MX FP4** (`float4_e1m2x2_t`, `float4_e2m1x2_t`). **A3 does not** (`include/pto/npu/a2a3/` has no `datatype.hpp`; verified by `ls`).

### 5.2 A5-only register-aliases

[include/pto/npu/a5/common.hpp](../include/pto/npu/a5/common.hpp) (Known: file exists, A3 equivalent does not — verified by directory listing):

```cpp
using MaskReg    = vector_bool;
using UnalignReg = vector_align;
using AddrReg    = vector_address;
```

Used by A5 helpers (e.g., [a5/common.hpp `CreatePredicate`/`CreatePredicateImpl`](../include/pto/npu/a5/common.hpp)). A3 uses different machinery for predicates and unaligned addresses (Inferred — not yet inspected end-to-end).

### 5.3 `int4b_t` packed scalar — both, in `common/type.hpp`

[include/pto/common/type.hpp:109-118](../include/pto/common/type.hpp#L109-L118). (Known.) Shared by both archs.

---

## 6. Memory qualifier / `TileDType` summary

[include/pto/common/memory.hpp:23-115](../include/pto/common/memory.hpp#L23-L115) (Known):

| `TileType` | A3 (`__PTO_AUTO__`) | A5 (`__PTO_AUTO__`) |
|---|---|---|
| `Vec` | `__ubuf__ T` | `__ubuf__ T` |
| `Mat` | `__cbuf__ T` | `__cbuf__ T` |
| `Left` | `__ca__ T` | `__ca__ T` |
| `Right` | `__cb__ T` | `__cb__ T` |
| `Acc` | `__cc__ T` | `__cc__ T` |
| `Bias` | `__biasbuf__ T` (only when `__DAV_C220_CUBE__`); else `uint64_t` | Same — but `__DAV_C220_CUBE__` is A3-named. **Inferred**: A5 has its own cube-define gate; if missing, also falls back to `uint64_t`. **Unknown** what the A5 macro is. |
| `Scaling` | `__fbuf__ T` | `__fbuf__ T` |
| `ScaleLeft` | `__ca__ T` | `__ca__ T` |
| `ScaleRight` | `__cb__ T` | `__cb__ T` |
| `Ctrl` | `uint64_t` | `uint64_t` |

Manual mode adds a trailing `*` to all qualified types. (Known: each `MemoryQualifier<TileType, DType>` specialization in `memory.hpp`.)

The **only divergence in the qualifier mapping** is via the Bias `__DAV_C220_CUBE__` gate ([memory.hpp:73-83](../include/pto/common/memory.hpp#L73-L83)) — A3-named, A5 status Unknown.

The **`TileDType`-level divergence** is in §4.4: A5 adds the `Bias` exception that skips `tile_size(...)`. A3 does not.

---

## 7. GEMM / TMATMUL summary

Combined from §3.1, §4.1, §4.2, §4.3, §4.4.

| Aspect | A3 | A5 |
|---|---|---|
| `TMATMUL_IMPL` / `TMATMUL_ACC_IMPL` / `TMATMUL_BIAS_IMPL` | yes | yes |
| `TMATMUL_MX_IMPL` (microscaled) | no | yes |
| `TileLeft<...>::BLayout` | RowMajor | ColMajor |
| Block alignment | `C0_SIZE_BYTE / sizeof(U)` | `(sizeof(T) == 1) ? 32 : 16` |
| `BiasTile` element type | `B` (bias's own) | `OutType` (matmul's output) |
| `BiasTile` valid col | `validN` | padded `N` |
| `TileDType` for `Bias` | `... tile_size(Rows*Cols)` | bare qualified type (no `tile_size`) |
| Scale tiles (`TileLeftScale`, `TileRightScale`) | available (alias defined unconditionally, [pto_tile.hpp:1709-1724](../include/pto/common/pto_tile.hpp#L1709-L1724)); used only by MX paths | **used in MX FP4/FP8 GEMM** |
| `MX_A_*` / `MX_B_*` `Layout` | enum values exist ([type.hpp:175-180](../include/pto/common/type.hpp#L175-L180)) but no MX kernel in tree | used in [a5 tmatmul_mx_kernel.cpp:75-80](../tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp#L75-L80) |

In-tree references (Known):
- A3: [tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp), [kernels/manual/a2a3/gemm_performance/](../kernels/manual/a2a3/gemm_performance/), [kernels/manual/a2a3/gemm_ar/](../kernels/manual/a2a3/gemm_ar/). gemm_performance is **manual-mode only** (ping-pong + ungauarded set_flag/wait_flag — see [known_good_kernel_examples.md §C1](known_good_kernel_examples.md)).
- A5: [tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp](../tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp), [tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp](../tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp), [kernels/manual/a5/matmul_mxfp4_performance/](../kernels/manual/a5/matmul_mxfp4_performance/), [kernels/manual/a5/matmul_mxfp8_performance/](../kernels/manual/a5/matmul_mxfp8_performance/), [kernels/manual/a5/gemm_ar/](../kernels/manual/a5/gemm_ar/). The mx* directories are **A5-only** by definition.

---

## 8. Gather / scatter summary

Combined from §3.2.

| Op | A3 | A5 |
|---|---|---|
| `TGATHER` (single-table, indexed) | yes ([a2a3/TGather.hpp](../include/pto/npu/a2a3/TGather.hpp)) | yes ([a5/TGather.hpp](../include/pto/npu/a5/TGather.hpp)) |
| `TGATHERB` | yes ([a2a3/TGatherB.hpp](../include/pto/npu/a2a3/TGatherB.hpp)) | yes ([a5/TGatherB.hpp](../include/pto/npu/a5/TGatherB.hpp)) |
| `TSCATTER` | yes ([a2a3/TScatter.hpp](../include/pto/npu/a2a3/TScatter.hpp)) | yes ([a5/TScatter.hpp](../include/pto/npu/a5/TScatter.hpp)) — IMPL body is at line 136 vs A3's line 56; semantics may differ — **Inferred** divergence. |
| `MGATHER` (multi-table) | **NOT PRESENT** | yes ([a5/MGather.hpp](../include/pto/npu/a5/MGather.hpp)) |
| `MSCATTER` (multi-table) | **NOT PRESENT** | yes ([a5/MScatter.hpp](../include/pto/npu/a5/MScatter.hpp)) |

In-tree tests: [tests/npu/a5/src/st/testcase/mgather/](../tests/npu/a5/src/st/testcase/mgather/) and [tests/npu/a5/src/st/testcase/mscatter/](../tests/npu/a5/src/st/testcase/mscatter/) exist; A3 does not have these directories. (Known.)

> **Implication**: An attention or sparse-attention kernel that needs multi-table gather/scatter must target A5 only. A3 has to compose via repeated `TGATHER`/`TSCATTER` (Inferred — performance trade-off Unknown).

---

## 9. Reduction / softmax / attention-relevant differences

### 9.1 Reductions

A5 has unified `TRowReduce.hpp` / `TRowReduceIdx.hpp` headers; A3 has six separate ones (`TRowSum.hpp`, `TRowMax.hpp`, `TRowMin.hpp`, `TRowProd.hpp`, `TRowReduceOps.hpp`, `TRowReduceIdxOps.hpp`). (Known.) The user-facing `TROWSUM`/`TROWMAX`/etc. instructions exist on both; the per-arch dispatch is via [pto_instr_impl.hpp](../include/pto/common/pto_instr_impl.hpp).

`TRsqrt` is a separate header on A5 ([a5/TRsqrt.hpp](../include/pto/npu/a5/TRsqrt.hpp)); on A3 there is no dedicated `TRsqrt.hpp`. **Inferred**: A3 composes via `TRECIP` + `TSQRT`, but Unknown until verified.

### 9.2 Softmax math primitives

The math sequence (`TROWMAX → TROWEXPAND/TSUB → TMULS(scale) → TEXP → TROWSUM → TROWEXPAND/TDIV`) is identical on both archs at the user-API level (Known: [docs/coding/tutorials/row-softmax.md](../docs/coding/tutorials/row-softmax.md)). The library headers under [a2a3/](../include/pto/npu/a2a3/) and [a5/](../include/pto/npu/a5/) provide independent implementations; the kernel side is the same shape.

### 9.3 Attention-shaped helpers

A5 has additional **high-precision math** customizations under [a5/custom/](../include/pto/npu/a5/custom/): `Div754.hpp`, `TExp_Custom.hpp`, `TFmodRemHp.hpp`, `TLog_Custom.hpp`, `TSqrtHp.hpp`. A3 does not have analogues. (Known.)

> **Implication**: A5 attention kernels that need IEEE-754-precise division or a high-precision `exp`/`log`/`sqrt` can pull from these helpers. A3 cannot use them. A3 attention kernels must accept the lower-precision baseline (or implement their own software-precision math, which Inferred would conflict with auto mode if it requires raw CCE intrinsics outside `__tf__`).

A5 also has `TPow_Custom.hpp` references (Inferred — not directly verified; the directory listing shows `TExp_Custom.hpp` and `TLog_Custom.hpp`. There may be more under the same naming convention). **Status: Unknown**.

---

## 10. Manual-kernel directory differences

[kernels/manual/](../kernels/manual/) (Known):

| Kernel | A3 (`a2a3/`) | A5 (`a5/`) |
|---|---|---|
| `flash_atten` | yes | yes (also `fa_performance_dn_kernel.cpp` for the dn variant) |
| `gemm_ar` | yes | yes |
| `gemm_performance` | yes | **NOT PRESENT** |
| `allgather_gemm` | yes | yes |
| `conv2d_forward` | yes | **NOT PRESENT** |
| `topk` | yes | **NOT PRESENT** |
| `tget_bandwidth` | yes | **NOT PRESENT** |
| `engram_simt` | **NOT PRESENT** | yes |
| `matmul_mxfp4_performance` | **NOT PRESENT** | yes |
| `matmul_mxfp8_performance` | **NOT PRESENT** | yes |
| `flash_atten` (cross-platform) | shared via [kernels/manual/common/flash_atten/](../kernels/manual/common/flash_atten/) (per [kernels/manual/README.md](../kernels/manual/README.md): A2/A3/A5) | same (cross-platform) |

(Known.)

> **Implication**: When using a manual reference for an auto-mode A5 kernel design, the A5-side directories are the right starting point for matmul-mx and engram_simt. For conv2d, topk, gemm_performance, only A3 manual references exist; **Inferred** that the A5 path differs enough that semantic transfer is non-trivial. All manual kernels are heavy on `TASSIGN`, `set_flag`/`wait_flag`, `TPUSH`/`TPOP`, and double buffering — none of those work in auto mode (see [known_good_kernel_examples.md Group C](known_good_kernel_examples.md), [auto_mode_bad_patterns.md §2.4, §2.5](auto_mode_bad_patterns.md)).

---

## 11. PR-852 context (A3-only, merged)

PR-852 is documented in [external_context/pr_852_notes.md](external_context/pr_852_notes.md). Re-stating the points relevant to A3 vs A5:

- **Status / scope**: PR-852 is merged into the current branch and is A3-only. The PR description says: *"Some of the ST test cases (for A3) failed for the PTO Auto Mode."* All file paths in the diff are under `include/pto/npu/a2a3/` and `tests/npu/a2a3/`.
- **A5 is independent**, not a wrapper. A5 has its own [a5/Tci.hpp](../include/pto/npu/a5/Tci.hpp), [a5/TConcat.hpp](../include/pto/npu/a5/TConcat.hpp), [a5/TFillPad.hpp](../include/pto/npu/a5/TFillPad.hpp), [a5/TQuant.hpp](../include/pto/npu/a5/TQuant.hpp), [a5/TRowReduce.hpp](../include/pto/npu/a5/TRowReduce.hpp), [a5/TRowReduceIdx.hpp](../include/pto/npu/a5/TRowReduceIdx.hpp), [a5/TTrans.hpp](../include/pto/npu/a5/TTrans.hpp). PR-852 does not touch A5.
- **Do not assume A5 needs the same fixes** unless A5 source actually shows the same pattern.

A5 mirror status (cross-cutting, audit pending) — Known where verified, Unknown where not:

| PR-852 fix area | A3 status (current) | A5 mirror status |
|---|---|---|
| `__cce_get_tile_ptr(x + N)` pointer-arithmetic | Fixed by merged PR-852 in A3; keep `__cce_get_tile_ptr(x) + N` as the review rule. | **A5 spot-check negative**: [a5/Tci.hpp](../include/pto/npu/a5/Tci.hpp) does NOT contain the buggy pattern (Known: grep shows only `__cce_get_tile_ptr(dst)` style, no `+ N` inside the macro). |
| `PtoSetWaitFlag` inside `__tf__` body | Fixed by merged PR-852 in the affected A3 headers by guarding `PtoSetWaitFlag` and emitting raw `set_flag`/`wait_flag` under `__PTO_AUTO__`. | **A5 spot-check negative**: A5 [TConcat.hpp](../include/pto/npu/a5/TConcat.hpp) / [TFillPad.hpp](../include/pto/npu/a5/TFillPad.hpp) / [TRowReduceIdx.hpp](../include/pto/npu/a5/TRowReduceIdx.hpp) / [TTrans.hpp](../include/pto/npu/a5/TTrans.hpp) do NOT contain `PtoSetWaitFlag` calls at all (Known: grep returned empty). They may use a different sync style. **Unknown** whether the A5 sync style has its own auto-mode hazards. |
| `TQuantBuffersOverlap` `reinterpret_cast<uintptr_t>` | Fixed by merged PR-852 in [a2a3/TQuant.hpp](../include/pto/npu/a2a3/TQuant.hpp) with a `#ifndef __PTO_AUTO__` raw-pointer branch and `return true` in auto mode. | **Unknown** for A5 — [a5/TQuant.hpp](../include/pto/npu/a5/TQuant.hpp) is a much larger and structurally different file; whether it has an analogous overlap check has not been audited. |
| `TQuantCvtS32ToFp16` not `__tf__` | Fixed by merged PR-852 in [a2a3/TQuant.hpp](../include/pto/npu/a2a3/TQuant.hpp): the helper is now `__tf__` and takes `TileDType __in__` / `__out__`. | **Unknown** — A5 quant path has different functions (e.g., `AbsReduceMax_Naive`); this specific function may not exist on A5. |
| `TRESHAPE` rejects `ConvTile` | Fixed by merged PR-852 in [a2a3/TReshape.hpp](../include/pto/npu/a2a3/TReshape.hpp): strict asserts are manual-mode-only and auto mode uses `__cce_alias`. | **Likely same risk on A5 if its source has the old asserts**: [a5/TReshape.hpp](../include/pto/npu/a5/TReshape.hpp) exists and is included by [a5/TQuant.hpp:18](../include/pto/npu/a5/TQuant.hpp#L18). Whether it has the same `is_tile_data_v` asserts outside the `__PTO_AUTO__` guard — **Unknown** (file not yet inspected end-to-end). |
| Single template across mismatched TileTypes (`TTransConv*`) | Fixed by merged PR-852 in [a2a3/TTrans.hpp](../include/pto/npu/a2a3/TTrans.hpp) via separate `TileDataDst`, `TileDataSrc`, and `TileDataTmp` template parameters. | **Unknown** — A5 [TTrans.hpp](../include/pto/npu/a5/TTrans.hpp) is independent. Audit pending. |
| `ConvTile<..., bytes, ...>` misuse | Fixed by merged PR-852 in [tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp](../tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp), which now passes `elementSize`. | A5 has no `texpands_mat` testcase ([tests/npu/a5/src/st/testcase/](../tests/npu/a5/src/st/testcase/) does not list it; verified by `comm` of testcase dirs). Whether other A5 kernels make the same mistake — **Unknown**. |

> **Default stance** for cross-arch design: PR-852 fixes are current A3 behavior but remain A3-only until you have verified the A5 source line-by-line. Specifically, do NOT pre-emptively apply the dual-mode aliasing recipe (PR-852 [§T3](external_context/pr_852_notes.md)) to A5 kernels unless A5 testcases show the same `TASSIGN(a, X); TASSIGN(b, X);` overlap pattern.

---

## 12. Open assumptions and items to verify

Tracked with full Confidence/Status labels in [assumptions_to_verify.md](assumptions_to_verify.md):

- **§2** — A5 cube compile flag, A5 cube macro name, A5 `Tile<Bias>::TileDType` `tile_size(...)` skip, `TileLeft` BLayout split + `TileMatA` declarations, A5 ST sweep for "auto-mode failures."
- **§5.4** — A5 quant `MXFP8` bit-level equivalence with A3 `TQUANT<INT8_SYM, ...>`.
- **§5.5** — A5 `PTO_URMA` path (`__NPU_ARCH__ == 3510` only) effect on auto-mode kernel generation.
- **§6** — A3 vs A5 GEMM-MX `SFractalSize = 512` for non-FP16, `BiasTile` element type divergence, alignment formula divergence.
- **§7** — A3 `TRSQRT` user-API surface; A5 `Custom`/`Hp` helper kernel-callability and the A3-vs-A5 precision floor.
- **§8.3** — A5 `TSCATTER` semantic divergence from A3.
- **§9.1** — A5 mirror full audit of PR-852 fix targets ([a5/TQuant.hpp](../include/pto/npu/a5/TQuant.hpp), [a5/TReshape.hpp](../include/pto/npu/a5/TReshape.hpp), [a5/TTrans.hpp](../include/pto/npu/a5/TTrans.hpp)).

---

## 13. Quick reference card

What is **definitely different** between A3 and A5 (auto-mode-relevant, Known):

- Arch macros, SoC compile flags.
- Per-arch include subtree under `include/pto/npu/`.
- Header file naming case (`TCI.hpp` vs `Tci.hpp`, etc.).
- Reduction header organization (separate vs unified).
- `TMATMUL_MX` exists only on A5.
- `MGATHER` / `MSCATTER` exist only on A5.
- `THistogram`, `TRandom`, `TRsqrt`, `TGetScaleAddr` headers exist only on A5.
- `*_Custom` / `Hp` helpers exist only on A5.
- A5 has independent `datatype.hpp` defining MX FP4/FP8 element types.
- A5 has `MaskReg`/`UnalignReg`/`AddrReg` aliases.
- `TileLeft<...>` base layout: RowMajor on A3, ColMajor on A5.
- GEMM block alignment formula differs (A5 conditional vs A3 division).
- `BiasTile` declaration differs in element type and valid col.
- `Tile<TileType::Bias>::TileDType` skips `tile_size(...)` on A5; not on A3.
- `ConvTile` carries extra `dstStride_`/`dstMposition_` fields on A5 (and accessors).
- A5 has manual matmul-mxfp4/8 and engram_simt kernels; A3 has manual conv2d_forward, gemm_performance, topk, tget_bandwidth.

What is **the same** between A3 and A5 (auto-mode-relevant, Known):

- The auto-mode contract overall (`__PTO_AUTO__`, `--cce-enable-pto-passes`, `-O2`).
- `TileType` enum members and their qualifier mapping (modulo the Bias gate).
- `TileConfig` constants (`alignedSize=32`, `fractalABSize=512`, `fractalCSize=1024`, `fractalMxSize=32`).
- `Tile`/`ConvTile`/`GlobalTensor` template signatures.
- `Shape`/`Stride`/`TileShape2D`/`BaseShape2D` helpers.
- The `TileDType` vs `DType` distinction.
- The "kernel rules" and "library rules" auto-mode contract ([docs/auto_mode/](../docs/auto_mode/)).
- The `__tf__`/`__in__`/`__out__`/`__cce_get_tile_ptr` keyword set ([qualifier_reference.md](qualifier_reference.md)).

What is **Unknown** (audit pending) — see §12.

---

## 14. Per-AI-core buffer capacities and cube/vec data flow

Source: user-provided architecture briefing (2026-05). The full diagram and tile-budget arithmetic live in [pto_auto_mode_hw_optimization_guide.md §1.3](pto_auto_mode_hw_optimization_guide.md). Summarized here for cross-arch comparison.

| Resource | A3 (`dav-c220` / Ascend 910B1) | A5 (`dav-c310`) |
|---|---|---|
| AI cores per chip | 25 | **Unknown** |
| Vec cores per chip | 50 | **Unknown** |
| AICPUs per chip | 4 | **Unknown** |
| L1 (cube staging) | 512 KB | **Unknown** |
| L0A (`TileLeft`, `__ca__`) | 64 KB | **Unknown** |
| L0B (`TileRight`, `__cb__`) | 64 KB | **Unknown** |
| L0C (`TileAcc`, `__cc__`, fp32) | 128 KB | **Unknown** |
| UB (vec staging, `__ubuf__`) | 192 KB | **Unknown** |
| GM total | **Unknown** | **Unknown** |
| Cube ↔ vec handoff | round-trips through GM | round-trips through GM (Inferred — same arch family) |

A5 capacities are **Unknown** pending user input. See [assumptions_to_verify.md §6.1](assumptions_to_verify.md). Cube unit only does matmul; non-matmul work (TMAXS / ReLU, casts, element ops, reductions, gather/scatter) runs on vector on both arches — same kernel-rules contract from [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md).
