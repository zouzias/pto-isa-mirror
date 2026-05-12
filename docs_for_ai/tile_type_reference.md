# Tile-Type Reference (auto-mode A3/A5)

Source-grounded reference for `pto::Tile<...>`, `pto::ConvTile<...>`,
`pto::GlobalTensor<...>`, related shape/layout/valid-region helpers, and the
auto-mode-specific risks around them. Companion to
[repo_kernel_map.md](repo_kernel_map.md),
[known_good_kernel_examples.md](known_good_kernel_examples.md),
[auto_mode_bad_patterns.md](auto_mode_bad_patterns.md),
[qualifier_reference.md](qualifier_reference.md), and
[external_context/pr_852_notes.md](external_context/pr_852_notes.md).

Scope: A3 (`PTO_NPU_ARCH_A2A3`, `__NPU_ARCH__ == 2201`) and A5
(`PTO_NPU_ARCH_A5`, `__NPU_ARCH__ == 3101 || 3510`) per
[include/pto/common/arch_macro.hpp:14-20](../include/pto/common/arch_macro.hpp#L14-L20).
CPU-sim and cost-model paths are referenced only as compile-fallbacks; do not
take their behavior as evidence of A3/A5 auto-mode validity.

Confidence labels:
- **Known** — directly readable from a quoted file path.
- **Inferred** — derived by combining file evidence; reasoning given.
- **Assumption** — not present in source; flagged for confirmation.
- **Unknown** — referenced but not yet confirmable.

---

## 1. The five core types

### 1.1 `pto::Tile<...>` — primary template

Definition: [include/pto/common/pto_tile.hpp:1381-1385](../include/pto/common/pto_tile.hpp#L1381-L1385). (Known)

```cpp
template <
  pto::TileType  Loc_,
  typename       Element_,
  const int      Rows_,
  const int      Cols_,
  const BLayout  BFractal_     = BLayout::RowMajor,
  const int      RowValid_     = Rows_,
  const int      ColValid_     = Cols_,
  const SLayout  SFractal_     = SLayout::NoneBox,
  const int      SFractalSize_ = TileConfig::fractalABSize,  // = 512
  auto           PadVal_       = PadValue::Null,
  const CompactMode Compact_   = CompactMode::Null
>
struct Tile { ... };
```

Static members exposed (Known: [pto_tile.hpp:1387, 1415-1432](../include/pto/common/pto_tile.hpp#L1387-L1432)):
- `using DType = Element_;` — the **scalar element type** (e.g., `float`, `half`).
- `Loc`, `Rows`, `Cols`, `RowStride`, `ColStride`, `ValidRow`, `ValidCol`, `BFractal`, `SFractal`, `Numel = Rows*Cols`, `SFractalSize`, `PadVal`, `Compact`.
- `static constexpr bool isRowMajor = (BFractal_ == BLayout::RowMajor);`
- `static constexpr bool isBoxedLayout = (SFractal != SLayout::NoneBox);`

`TileDType` definition (Known: [pto_tile.hpp:1526-1540](../include/pto/common/pto_tile.hpp#L1526-L1540)):

```cpp
#if defined(__CPU_SIM) || defined(__COSTMODEL)
    using TileDType = Tile::DType *;          // pointer (CPU-sim, cost-model)
#else
#ifdef __PTO_AUTO__
#if defined(PTO_NPU_ARCH_A2A3)
    using TileDType = typename MemoryQualifier<Loc, DType>::type tile_size(Rows*Cols);
#else  // A5
    using TileDType = std::conditional_t<Loc == TileType::Bias,
        typename MemoryQualifier<Loc, DType>::type,                    // Bias special-cased
        typename MemoryQualifier<Loc, DType>::type tile_size(Rows*Cols)>;
#endif
#else  // manual mode (A3 or A5)
    using TileDType = typename MemoryQualifier<Loc, DType>::type;       // pointer
#endif
#endif
```

Concrete consequence (Known + Inferred):
- **Auto mode (A3/A5 device)**: `TileDType` is a sized **vector value** declared with the bisheng-CCE `tile_size(N)` modifier on top of a memory-qualified element type (e.g., `__ubuf__ float tile_size(R*C)`). The token `tile_size` is not `#define`d anywhere in the repo (Known) — Inferred to be a CCE compiler builtin/keyword, in the same family as `__tf__`/`__cce_get_tile_ptr` (see [qualifier_reference.md §3](qualifier_reference.md)).
- **Manual mode**: `TileDType` is a memory-qualified raw pointer (`__ubuf__ float *`, `__ca__ half *`, etc.).
- **A5 special case**: `TileType::Bias` skips the `tile_size(...)` wrap even in auto mode — it is just `typename MemoryQualifier<Bias, DType>::type`. (Known: [pto_tile.hpp:1534-1536](../include/pto/common/pto_tile.hpp#L1534-L1536).)

### 1.2 `pto::ConvTile<...>` — convolution-shaped tile (separate template)

Definition: [include/pto/common/pto_tile.hpp:1096-1097](../include/pto/common/pto_tile.hpp#L1096-L1097). (Known)

```cpp
template <
  pto::TileType  Loc_,
  typename       Element_,
  const int      BufferSize_,   // **element count**, NOT bytes (see §10.2)
  pto::Layout    Layout_,
  typename       Shape_         // e.g., pto::ConvTileShape<N, C1, H, W, C0>
>
struct ConvTile { ... };
```

Static members (Known: [pto_tile.hpp:1099-1116, 1166-1170](../include/pto/common/pto_tile.hpp#L1099-L1116)):
- `using DType = Element_;`
- `using ShapeType = Shape_;`
- `Loc`, `bufferSize = BufferSize_`, `layout = Layout_`, `totalDimCount`, `staticShape[6]`, `dynamicDimCount`, `isDynamicDim[6]`.
- `int64_t shape[6]` — runtime-mutable shape array; in **auto mode** `GetShape(dim)` returns `staticShape[dim]` only — **dynamic shapes are not supported in auto mode** (Known: [pto_tile.hpp:1124-1129](../include/pto/common/pto_tile.hpp#L1124-L1129)).

`TileDType` definition (Known: [pto_tile.hpp:1166-1170](../include/pto/common/pto_tile.hpp#L1166-L1170)):

```cpp
#if defined(__PTO_AUTO__) && !defined(__CPU_SIM)
    using TileDType = typename MemoryQualifier<Loc_, DType>::type tile_size(bufferSize);
#else
    using TileDType = typename MemoryQualifier<Loc_, DType>::type;     // pointer
#endif
```

This is the signature underlying [auto_mode_bad_patterns.md §5.6](auto_mode_bad_patterns.md): `bufferSize` is fed to `tile_size(...)`, which expects an element count, not a byte count. See §10.2 below.

Distinguished from `Tile` by `is_conv_tile_v<T>` ([pto_tile.hpp:1758-1762](../include/pto/common/pto_tile.hpp#L1758-L1762)).

### 1.3 `pto::GlobalTensor<Element_, Shape_, Stride_, Layout_>`

Definition: [include/pto/common/pto_tile.hpp:248-253](../include/pto/common/pto_tile.hpp#L248-L253). (Known)

```cpp
template <typename Element_, typename Shape_, typename Stride_, Layout Layout_ = Layout::ND>
struct GlobalTensor {
    using Shape = Shape_;
    using Stride = Stride_;
    using RawDType = Element_;
    using DType = __gm__ Element_;             // <-- DType is __gm__-qualified
    ...
};
```

Key contrast with `Tile` (Known + Inferred):
- `Tile::DType` = scalar element type (`float`, `half`, …).
- `GlobalTensor::DType` = `__gm__ Element_` — already memory-qualified (`__gm__` is the GM address space).
- `GlobalTensor::data()` returns a raw `__gm__ Element_ *` pointer in **all** modes (Known: [pto_tile.hpp:549](../include/pto/common/pto_tile.hpp#L549) is the relevant non-vector `data()` member; verified by ubiquitous `out = dstGlobal.data();` usage in test kernels). This is **always safe** to call from kernel code, unlike `Tile::data()`.

### 1.4 `Shape<N1,N2,N3,N4,N5>` and `Stride<S1,S2,S3,S4,S5>`

Defined at [include/pto/common/pto_tile.hpp:27-134](../include/pto/common/pto_tile.hpp#L27-L134) and [pto_tile.hpp:137-245](../include/pto/common/pto_tile.hpp#L137-L245). (Known)

Both are **5-dimensional** templates. Constants (Known: [type.hpp:398-405](../include/pto/common/type.hpp#L398-L405)):

```cpp
namespace GlobalTensorDim {
constexpr int DIM_0 = 0; constexpr int DIM_1 = 1; constexpr int DIM_2 = 2;
constexpr int DIM_3 = 3; constexpr int DIM_4 = 4;
constexpr int TOTAL_DIM = 5;
}
```

`DYNAMIC = -1` ([pto_tile.hpp:25](../include/pto/common/pto_tile.hpp#L25)) marks any dim/stride that should be supplied at runtime via the constructor.

### 1.5 `TileShape2D<T, rows, cols, Layout>` and `BaseShape2D<T, rows, cols, Layout>`

Defined at [pto_tile.hpp:594-945](../include/pto/common/pto_tile.hpp#L594-L945) (`TileShape2D`) and [pto_tile.hpp:679-942](../include/pto/common/pto_tile.hpp#L679-L942) (`BaseShape2D`). (Known)

Both inherit from `Shape`/`Stride` and are 2-D convenience wrappers that lower a `(rows, cols)` shape into the 5-D representation according to a `Layout`. Specializations exist for `Layout::ND`, `Layout::DN`, `Layout::NZ`, `Layout::MX_A_ZZ`, `Layout::MX_A_ND`, `Layout::MX_A_DN`, `Layout::MX_B_NN`, `Layout::MX_B_ND`, `Layout::MX_B_DN` (Known — line numbers in the grep above).

Used inside library headers and a handful of test kernels (e.g., [tests/npu/a2a3/src/st/testcase/trowsum/trowsum_kernel.cpp:55-57](../tests/npu/a2a3/src/st/testcase/trowsum/trowsum_kernel.cpp#L55-L57): `TileShape2D<T, validRow, srcValidCol> + BaseShape2D<T, row, srcCol> + GlobalTensor<...>`).

---

## 2. `TileType` enum — what the eleven values mean

Definition: [include/pto/common/type.hpp:122-134](../include/pto/common/type.hpp#L122-L134). (Known)

| `TileType` | Storage class (`MemoryQualifier`) | Typical role |
|---|---|---|
| `Vec` | `__ubuf__` | Vector / UB-resident tile |
| `Mat` | `__cbuf__` | L1-resident matrix tile (movement to L0 via `TMOV`) |
| `Left` | `__ca__` | L0A operand of `TMATMUL` |
| `Right` | `__cb__` | L0B operand of `TMATMUL` |
| `Acc` | `__cc__` | L0C accumulator of `TMATMUL` |
| `Bias` | `__biasbuf__` (only when `__DAV_C220_CUBE__`) else `uint64_t` | Cube bias tile |
| `Scaling` | `__fbuf__` | Quant/dequant scale buffer |
| `ScaleLeft` | `__ca__` | MX scale tile, A-side (A5) |
| `ScaleRight` | `__cb__` | MX scale tile, B-side (A5) |
| `Ctrl` | `uint64_t` | Control state (e.g., predicate registers) |

Mapping source: [include/pto/common/memory.hpp:27-115](../include/pto/common/memory.hpp#L27-L115). (Known)

> **A3 vs A5** (Known): `TileType::Vec/Mat/Left/Right/Acc/Bias/Scaling/Ctrl` are shared across A3 and A5. `TileType::ScaleLeft`/`ScaleRight` are used in both, but the tile aliases `TileLeftScale`/`TileRightScale` (see §3) target MX (microscaling) — primarily A5 ([include/pto/npu/a5/datatype.hpp:23-42](../include/pto/npu/a5/datatype.hpp#L23-L42) defines the MX FP4/FP8 element types).

> **`Bias` gotcha** (Known): on builds without `__DAV_C220_CUBE__`, `MemoryQualifier<Bias, T>::type` falls back to `uint64_t` ([memory.hpp:73-83](../include/pto/common/memory.hpp#L73-L83)). A vec-only build that declares a `Tile<TileType::Bias, ...>` gets a degenerate type. See [auto_mode_bad_patterns.md §5.3](auto_mode_bad_patterns.md).

---

## 3. Common typed-alias shapes (TileLeft / TileRight / TileAcc / TileBias / scale variants)

Defined at [include/pto/common/pto_tile.hpp:1682-1732](../include/pto/common/pto_tile.hpp#L1682-L1732). (Known)

### 3.1 `TileLeft<T, Rows, Cols, RowValid=Rows, ColValid=Cols>`

The A2A3 form ([pto_tile.hpp:1682-1684](../include/pto/common/pto_tile.hpp#L1682-L1684)):
```cpp
using TileLeft = Tile<TileType::Left, T, Rows, Cols,
                      BLayout::RowMajor,         // <-- RowMajor on A3
                      RowValid, ColValid,
                      SLayout::RowMajor,         // boxed (Zz)
                      TileConfig::fractalABSize  // = 512
                     >;
```
A different alias exists at [pto_tile.hpp:1692-1694](../include/pto/common/pto_tile.hpp#L1692-L1694) using `BLayout::ColMajor`. The two definitions are guarded by something not visible in this grep — Inferred to be an A3 vs A5 split. **Status: Unknown which arch picks which.** Confirm by reading the surrounding `#ifdef`.

`TileLeftCompact` adds `PadValue::Null + CompactMode::Normal`. ([pto_tile.hpp:1687-1688](../include/pto/common/pto_tile.hpp#L1687-L1688))

### 3.2 `TileRight<T, Rows, Cols, RowValid=Rows, ColValid=Cols>`

[pto_tile.hpp:1701-1703](../include/pto/common/pto_tile.hpp#L1701-L1703):
```cpp
using TileRight = Tile<TileType::Right, T, Rows, Cols,
                       BLayout::RowMajor, RowValid, ColValid,
                       SLayout::ColMajor,        // <-- ColMajor inner box (Nz)
                       TileConfig::fractalABSize>;
```

`TileRightCompact` at [pto_tile.hpp:1705-1707](../include/pto/common/pto_tile.hpp#L1705-L1707).

### 3.3 `TileAcc<T, Rows, Cols, RowValid=Rows, ColValid=Cols>`

[pto_tile.hpp:1726-1728](../include/pto/common/pto_tile.hpp#L1726-L1728):
```cpp
using TileAcc = Tile<TileType::Acc, T, Rows, Cols,
                     BLayout::ColMajor,          // <-- ColMajor base
                     RowValid, ColValid,
                     SLayout::RowMajor,          // boxed
                     TileConfig::fractalCSize    // = 1024
                    >;
```

`TileAccCompact` at [pto_tile.hpp:1730-1732](../include/pto/common/pto_tile.hpp#L1730-L1732).

### 3.4 MX-scale aliases (A5-relevant)

[pto_tile.hpp:1709-1724](../include/pto/common/pto_tile.hpp#L1709-L1724):
- `TileLeftScale<T, R, C, RV=R, CV=C>` — `TileType::ScaleLeft`, `BLayout::RowMajor`, `SLayout::RowMajor`, `SFractalSize = TileConfig::fractalMxSize` (= 32).
- `TileRightScale<T, R, C, RV=R, CV=C>` — `TileType::ScaleRight`, `BLayout::ColMajor`, `SLayout::ColMajor`, `SFractalSize = 32`.
- `TileLeftScaleCompact`, `TileRightScaleCompact` — same with `CompactMode::Normal`.

### 3.5 Bias (no canonical alias — declare the `Tile` directly)

Per the in-tree GEMM examples ([tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp:59](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L59), [a5/.../tmatmul_mx_kernel.cpp:111](../tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp#L111)):
```cpp
using BiasTile = Tile<TileType::Bias, BiasType, 1, N, BLayout::RowMajor, 1, validN>;
```

There is no `TileBias<...>` alias in [pto_tile.hpp](../include/pto/common/pto_tile.hpp) (verified by grep). Declare the `Tile<TileType::Bias, ...>` form explicitly.

---

## 4. `TileConfig` constants

Defined: [include/pto/common/pto_tile.hpp:947-956](../include/pto/common/pto_tile.hpp#L947-L956). (Known)

```cpp
namespace TileConfig {
static constexpr int alignedSize     = 32;     // bytes, "block" alignment
static constexpr int fixedRowSize    = 16;
static constexpr int fixedColSize    = 16;
static constexpr int fixedMxRowSize  = 16;
static constexpr int fixedMxColSize  = 2;
static constexpr int fractalABSize   = 512;    // L0A/L0B fractal block (bytes)
static constexpr int fractalCSize    = 1024;   // L0C fractal block (bytes)
static constexpr int fractalMxSize   = 32;     // MX scale fractal block (bytes)
}
```

Inner box (fractal) shape is computed by `getInnerRow()`/`getInnerCol()` at [pto_tile.hpp:1389-1413](../include/pto/common/pto_tile.hpp#L1389-L1413). For `SFractalSize = fractalCSize` it is fixed `16x16`; for `fractalMxSize` it is fixed `16x2`; otherwise it depends on `BLayout`/`SLayout`/`alignedSize/sizeof(DType)`.

The `Tile` template asserts at compile time (Known: [pto_tile.hpp:1505-1524](../include/pto/common/pto_tile.hpp#L1505-L1524)):
1. `Cols % InnerCols == 0`.
2. For non-Vec, non-MX, non-`Rows==1` tiles: `Rows % InnerRows == 0`.
3. Either: row-major + non-boxed → `Cols * sizeof(DType) % alignedSize == 0`; OR col-major + non-boxed → `Rows * sizeof(DType) % alignedSize == 0`; OR boxed.
4. `SFractalSize` ∈ `{fractalABSize, fractalCSize, fractalMxSize}`.

Useful when picking `Rows`/`Cols` for a new kernel.

---

## 5. `BLayout`, `SLayout`, and `Layout`

`BLayout` — outer (base) layout of the tile ([type.hpp:136-140](../include/pto/common/type.hpp#L136-L140)):
- `RowMajor = 0`, `ColMajor = 1`.

`SLayout` — inner (boxed/fractal) layout ([type.hpp:142-147](../include/pto/common/type.hpp#L142-L147)):
- `NoneBox = 0` (flat), `RowMajor = 1` (Zz), `ColMajor = 2` (Nz).

Combined naming (Known: [memory.hpp:117-129](../include/pto/common/memory.hpp#L117-L129) `GetLayoutName`):
- (Row, NoneBox) → `"ND"`, (Col, NoneBox) → `"DN"`.
- (Row, RowMajor) → `"Zz"`, (Col, RowMajor) → `"Nz"`.
- (Row, ColMajor) → `"Zn"`, (Col, ColMajor) → `"Nn"`.

`Layout` — tensor logical layout ([type.hpp:169-192](../include/pto/common/type.hpp#L169-L192)). 21 values. The relevant set for A3/A5 auto-mode kernels:
- **2-D / general**: `ND`, `DN`, `NZ`, `SCALE`, `MAX` (sentinel).
- **MX (microscale)**: `MX_A_ND`, `MX_A_DN`, `MX_A_ZZ`, `MX_B_ND`, `MX_B_DN`, `MX_B_NN`. Used with `TileShape2D` / `BaseShape2D` to declare `GlobalTensor<ScaleType, ...>` for MX FP4/FP8 GEMM (Known: [a5/.../tmatmul_mx_kernel.cpp:75-80](../tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp#L75-L80)).
- **Conv**: `NC1HWC0`, `GNC1HWC0`, `NCHW`, `GNCHW`, `NHWC`, `NDC1HWC0`, `NCDHW`, `FRACTAL_Z`, `FRACTAL_Z_S16S8`, `FRACTAL_Z_3D`. Drive `ConvTile<...>` and the `TTrans*` library helpers ([include/pto/npu/a2a3/TTrans.hpp](../include/pto/npu/a2a3/TTrans.hpp)).

> **Don't conflate**: `BLayout`/`SLayout` describe **tile** structure; `Layout` describes **tensor** structure (mostly used by `GlobalTensor` and `ConvTile`). They are distinct enums in distinct namespaces.

---

## 6. Valid region — static, dynamic, and the runtime API

Static valid region (the common case, Known):
```cpp
using TileData = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, 64, 64>;
TileData t;             // ValidRow=64, ValidCol=64 are constexpr
```

Dynamic valid region with `DYNAMIC = -1` (Known: [pto_tile.hpp:25](../include/pto/common/pto_tile.hpp#L25)):
```cpp
using TileData = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1>;
TileData t(validRows, validCols);   // both dynamic
TileData t(validRows);              // only row dynamic
TileData t(validCols);              // only col dynamic   (constructors at [pto_tile.hpp:1459-1488])
```

API (Known: [pto_tile.hpp:1584-1631](../include/pto/common/pto_tile.hpp#L1584-L1631)):
- `t.GetValidRow()` / `t.GetValidCol()` — `static constexpr` overload when the dim is static; instance method when dynamic.
- `t.SetValidRow(rowMask)`, `t.SetValidCol(colMask)`, `t.SetValidShape(rowMask, colMask)` — only valid when the corresponding dim is `DYNAMIC`. Header comment (Known: [pto_tile.hpp:1608, 1616, 1624](../include/pto/common/pto_tile.hpp#L1608)): *"Call this function need PIPE_S wait."*
- Asserts that `rowMask <= Rows` / `colMask <= Cols`.

**Auto-mode caveat** (Inferred): `Set*` functions touch state that the auto-sync compiler analyzes. Per the header comment, callers need PIPE_S sync; whether `SetValidShape` interacts safely with auto-sync inside a `__tf__` body is **Unknown**. Existing test kernels prefer constructor-time supply of the dynamic dims (e.g., [trowsum_kernel.cpp:31](../tests/npu/a2a3/src/st/testcase/trowsum/trowsum_kernel.cpp#L31): `srcTileData srcTile(validRow, srcValidCol);`).

`GetKAligned()` — extra accessor on `Tile` ([pto_tile.hpp:1636-1639](../include/pto/common/pto_tile.hpp#L1636-L1639)) used by some library helpers; Inferred to be a K-axis alignment hint set by `TLOAD`/`TMOV`.

---

## 7. `Tile::data()` semantics — the most-cited gotcha

Source (Known: [pto_tile.hpp:1543-1561](../include/pto/common/pto_tile.hpp#L1543-L1561)):

```cpp
#if (defined(__CPU_SIM) && defined(__PTO_AUTO__)) || defined(__COSTMODEL)
TileDType &data() {
    if (!data_) { internalBuffer.resize(Rows*Cols); data_ = internalBuffer.data(); }
    return data_;
}
#else
AICORE TileDType &data() { return data_; }
AICORE const TileDType &data() const { return data_; }
#endif
```

What `data()` returns, by mode (Known + Inferred):

| Mode | Return type | Usable as a pointer? |
|---|---|---|
| Manual (A3/A5 device) | `TileDType &` = `__ubuf__ T * &` (or `__ca__/__cb__/__cc__/__cbuf__/__fbuf__/__biasbuf__`) | Yes — it IS a pointer. |
| Auto (A3/A5 device) | `TileDType &` = `__ubuf__ T tile_size(R*C) &` (i.e., a sized vector reference) | **No.** Apply `__cce_get_tile_ptr(...)` first. |
| CPU-sim, cost-model | `TileDType &` = `T * &` | Yes — it's a host pointer. |

Implications (drawn together in [auto_mode_bad_patterns.md §3.3, §1.3, §3.2.1, §3.2](auto_mode_bad_patterns.md) and [qualifier_reference.md §3.3](qualifier_reference.md)):
- `reinterpret_cast<uintptr_t>(tile.data())` is meaningless in auto mode.
- `__cce_get_tile_ptr(tile.data())` from kernel code is wrong: kernel-level is forbidden by Kernel rules §3.2.
- Inside a `__tf__` body, the canonical shape is to receive `TileDType` **by value** (not `Tile&`) and call `__cce_get_tile_ptr(param)` directly — see PR-852's `TQuantCvtS32ToFp16` migration ([external_context/pr_852_notes.md §L4b](external_context/pr_852_notes.md), not yet merged).
- `__cce_get_tile_ptr(tmp + N)` is wrong because the `+ N` applies arithmetic to a vector value before extraction; use `__cce_get_tile_ptr(tmp) + N` (PR-852 §L1).

`GlobalTensor::data()` is **not** the same — it returns a raw `__gm__ T *` pointer in all modes ([pto_tile.hpp:549](../include/pto/common/pto_tile.hpp#L549)) and is safe to call from kernel code (e.g., `out = dstGlobal.data();`).

---

## 8. Common patterns from A3 and A5 kernels

### 8.1 Elementwise vec template (A3) — auto-mode-confirmed

[demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp](../demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp) and [tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp). (Known)

```cpp
using ShapeDim5 = pto::Shape<1, 1, 1, R, C>;
using StridDim5 = pto::Stride<1, 1, 1, C, 1>;
using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;
using TileData = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, -1, -1>;
TileData t(validR, validC);            // dynamic valid region
GlobalData g(gmPtr);
TLOAD(t, g); ... TSTORE(g, t);
```

### 8.2 Cube GEMM template (A3) — Mat → L0 via TMOV → TMATMUL → TSTORE

[tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp:51-76](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L51-L76). (Known)

```cpp
// L1-resident matrix tiles, ColMajor base, RowMajor 512-block fractal
using TileMatAData = Tile<TileType::Mat, U, M, K, BLayout::ColMajor, validM, validK,
                          SLayout::RowMajor, 512>;
using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, validK, validN,
                          SLayout::RowMajor, 512>;
using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, validN>;

// L0 operand tiles via aliases
using LeftTile  = TileLeft <U, M, K, validM, validK>;
using RightTile = TileRight<S, K, N, validK, validN>;
using AccTile   = TileAcc  <T, M, N, validM, validN>;
using BiasTile  = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;
```

### 8.3 A5 MX FP4/FP8 GEMM template — adds scale tiles + MX layouts

[tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp:75-111](../tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp#L75-L111). (Known)

```cpp
using MxShapeA  = TileShape2D<ScaleType, M, kMX, Layout::MX_A_ZZ>;
using MxStrideA = BaseShape2D<ScaleType, M, kMX, Layout::MX_A_ZZ>;
using GlobalDataSrc2 = GlobalTensor<ScaleType, MxShapeA, MxStrideA, Layout::MX_A_ZZ>;

using TileScaleAData =
    Tile<TileType::Mat, ScaleType, M, kMX, BLayout::RowMajor, validM, kMX, SLayout::RowMajor, 32>;
using TileScaleBData =
    Tile<TileType::Mat, ScaleType, kMX, N, BLayout::ColMajor, kMX, validN, SLayout::ColMajor, 32>;

using LeftScaleTile  = TileLeftScale <ScaleType, M, kMX, validM, kMX>;
using RightScaleTile = TileRightScale<ScaleType, kMX, N, kMX, validN>;
```

Notes (Known + Inferred):
- The `512` fractal-size literal is the FP16-shaped value; A5 reuses `512` for the AB Mat tiles even when the element type is FP4/FP8 (Inferred — file uses `512` unconditionally). [auto_mode_bad_patterns.md §4.4](auto_mode_bad_patterns.md) flags this as a possible hazard for non-FP16 types.
- A5 alignment uses `(sizeof(AType) == 1) ? 32 : 16` ([a5/tmatmul:32](../tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L32)) — different from A3's `C0_SIZE_BYTE / sizeof(U)` ([a3/tmatmul:128](../tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp#L128)).

### 8.4 ND/DN row reduction (A3)

[tests/npu/a2a3/src/st/testcase/trowsum/trowsum_kernel.cpp:55-88](../tests/npu/a2a3/src/st/testcase/trowsum/trowsum_kernel.cpp#L55-L88). (Known)

```cpp
using ValidSrcShape = TileShape2D<T, validRow, srcValidCol>;          // ND default
using NDSrcShape    = BaseShape2D <T, row,      srcCol>;
using GlobalDataSrc = GlobalTensor<T, ValidSrcShape, NDSrcShape>;

using ValidDstShape = TileShape2D<T, dstCol, validRow>;
using NDDstShape    = BaseShape2D <T, row,    dstCol>;
using GlobalDataDst = GlobalTensor<T, ValidDstShape, NDDstShape>;

using srcTileData    = Tile<TileType::Vec, T, row, srcCol, BLayout::RowMajor, row, srcCol>;
using dstTileDataDN  = Tile<TileType::Vec, T, row, 1,      BLayout::ColMajor, row, 1>;
TROWSUM(dstTile, srcTile, tmpTile);
TRESHAPE(dstTileND, dstTile);    // alias to row-major view before TSTORE
TSTORE(dstGlobal, dstTileND);
```

This is one of the few in-tree TRESHAPE-as-aliasing-hint examples — see [known_good_kernel_examples.md §A4](known_good_kernel_examples.md).

### 8.5 ConvTile (A3) — `tload_gm2mat` / `texpands_mat`

[tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp:57-63](../tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp#L57-L63). (Known — current source state, pre-PR-852)

```cpp
constexpr int elementSize = N * C1 * H * W * C0;
constexpr int bufferSizeA = elementSize * sizeof(T);   // BUG (PR-852)
using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, 1, elementSize>,
                                pto::Stride<elementSize, elementSize, elementSize, elementSize, 1>>;
using TileData   = ConvTile<TileType::Mat, T, bufferSizeA,    // BAD — bytes (PR-852 fixes to elementSize)
                            Layout::NC1HWC0,
                            pto::ConvTileShape<N, C1, H, W, C0>>;
```

[tests/npu/a2a3/src/st/testcase/timg2col/timg2col_kernel.cpp:47-55](../tests/npu/a2a3/src/st/testcase/timg2col/timg2col_kernel.cpp#L47-L55) (current source, mixed — set_flag/wait_flag patterns):

```cpp
using TileMatAData = ConvTile<TileType::Mat, U, bufferSizeA, Layout::NC1HWC0, ...>;
using TileMatBData = ConvTile<TileType::Mat, U, bufferSizeB, Layout::FRACTAL_Z, ...>;
```

### 8.6 PadValue patterns (A3)

[tests/npu/a2a3/src/st/testcase/tfillpad/tfillpad_kernel.cpp:21, 113, 163-170](../tests/npu/a2a3/src/st/testcase/tfillpad/tfillpad_kernel.cpp#L21). (Known)

`PadValue` enum ([type.hpp:271-279](../include/pto/common/type.hpp#L271-L279)) has standard values `Null=0, Zero=1, Max=2, Min=3` plus a `CustomBase = 0x100000000ULL` sentinel for custom values encoded in bits [32:63]. The `PadValueCustom<v>` helper (`include/pto/common/constants.hpp`) packs a float/half/bf16 bit pattern into the enum's high bits.

The `Tile` template's `PadVal_` parameter ([pto_tile.hpp:1384](../include/pto/common/pto_tile.hpp#L1384)) accepts any `PadValue` (or a custom-encoded one). Used by `TFILLPAD`, `TLOAD`, `TSTORE` to control padding.

---

## 9. Auto-mode-specific tile risks (cross-reference)

Quick index back to [auto_mode_bad_patterns.md](auto_mode_bad_patterns.md):

| Risk | Where it appears in tile usage |
|---|---|
| §1.1 Manual `TASSIGN(a, X); TASSIGN(b, X);` aliasing | Replace with `TRESHAPE(b, a)` (auto-only) or keep + add `TRESHAPE` (dual-mode, PR-852 recipe in [known_good_kernel_examples.md §A10](known_good_kernel_examples.md)). |
| §1.2 Dynamic `TASSIGN(tile, addr_chosen_at_runtime)` | Tile addresses are constant in auto mode ([pto_tile.hpp:1124-1129](../include/pto/common/pto_tile.hpp#L1124-L1129) for ConvTile; same model for Tile per [docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §2.3](../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md)). |
| §1.3 `reinterpret_cast<uintptr_t>(tile.data())` | `tile.data()` returns a vector value in auto mode (§7); cast is meaningless. |
| §1.5 `TLOAD(dstTile, dstGlobal)` on a write-only dst | Tile liveness analysis may coalesce dst with another tile. Exception: read-modify-write ops like `TAXPY`. |
| §3.2 / §3.2.1 / §3.3 `Tile::data()` and `__cce_get_tile_ptr` misuse | `Tile::data()` is library-only; `__cce_get_tile_ptr` only inside `__tf__`; pointer arithmetic must follow extraction. |
| §5.3 `Tile<TileType::Bias, ...>` without `__DAV_C220_CUBE__` | Falls back to `uint64_t` ([memory.hpp:73-83](../include/pto/common/memory.hpp#L73-L83)). |
| §5.5 Reusing one `TileData` template across dst/src/tmp | Must be split per-role when `TileType` differs (PR-852 [§L7a](external_context/pr_852_notes.md)). |
| §5.6 `ConvTile<..., bufferSizeA, ...>` with bytes | `BufferSize_` is element count (§10.2). |

---

## 10. PR-852 tile-related supporting context (NOT yet merged into this branch)

PR 852 is described in [external_context/pr_852_notes.md](external_context/pr_852_notes.md). Scope per the PR is **A3 only**; A5 has independent files and is not modified. The current source still has the bugs.

### 10.1 `TRESHAPE` is the auto-mode aliasing hint; it must accept `ConvTile` in auto mode

Current source: [include/pto/npu/a2a3/TReshape.hpp:23-28](../include/pto/npu/a2a3/TReshape.hpp#L23-L28) has `static_assert(is_tile_data_v<TileDataIn>, ...)` and `static_assert(Loc == NewLoc, ...)` **outside** the `#ifndef __PTO_AUTO__` guard. That rejects `ConvTile` (which `is_tile_data_v` returns false for; instead `is_conv_tile_v` is the trait). PR-852 moves these asserts inside the `__PTO_AUTO__` guard so `ConvTile` aliasing works in auto mode.

Effect on tile usage (Inferred):
- **Pre-PR (current)**: `TRESHAPE(convTileView, convTileBase)` does not compile.
- **Post-PR**: It will compile under auto mode (and only auto mode; manual mode keeps the asserts).

PR-852 also drops the `Loc == NewLoc` assert in auto mode. **Soft gotcha**: post-merge, mismatched `TileType` between dst and src in `TRESHAPE` is silently accepted in auto mode — verify intent at call sites. (See [auto_mode_bad_patterns.md §1.1](../docs_for_ai/auto_mode_bad_patterns.md) note.)

### 10.2 `ConvTile<..., BufferSize_, ...>` accepts element count, not bytes

Source (Known: [pto_tile.hpp:1102, 1167](../include/pto/common/pto_tile.hpp#L1102)):
```cpp
static constexpr int bufferSize = BufferSize_;
...
using TileDType = typename MemoryQualifier<Loc_, DType>::type tile_size(bufferSize);
```

`tile_size(N)` is the bisheng-CCE allocator marker (§1.1) and consumes an element count. Passing bytes (e.g., `elementSize * sizeof(T)`) inflates the UB allocation by `sizeof(T)`. The misleading name is a documented long-term TODO in PR-852 (rename to `NumElems_`).

[texpands_mat_kernel.cpp:58, 63](../tests/npu/a2a3/src/st/testcase/texpands_mat/texpands_mat_kernel.cpp#L58) is the visible bug site; PR-852 fixes the call site, not the template name.

### 10.3 Decouple template parameters across dst/src/tmp tile-function arguments

Current `__tf__ PTO_INTERNAL TTransConv*<TileData, blockSizeElem>(...)` family in [include/pto/npu/a2a3/TTrans.hpp](../include/pto/npu/a2a3/TTrans.hpp) takes ONE `TileData` template parameter for tiles that may have different `TileType` (e.g., dst/src as `ConvTile<Mat>`, tmp as `Tile<Vec>`). PR-852 splits to `<TileDataDst, TileDataSrc, TileDataTmp, blockSizeElem>` and derives `Tdst`/`Tsrc`/`Ttmp` from `::DType` per role. See [auto_mode_bad_patterns.md §5.5](auto_mode_bad_patterns.md).

### 10.4 PR scope: A3 only

[external_context/pr_852_notes.md](external_context/pr_852_notes.md) Branch state and Scope sections. A5 has independent versions of `TConcat.hpp`, `TFillPad.hpp`, `TQuant.hpp`, `TRowReduce.hpp`, `TRowReduceIdx.hpp`, `TTrans.hpp`, `Tci.hpp` ([include/pto/npu/a5/](../include/pto/npu/a5/)). Spot-checks suggest A5 does NOT have the same `__cce_get_tile_ptr(x + N)` or `PtoSetWaitFlag`-inside-`__tf__` patterns, but a full audit is **Unknown**.

---

## 11. Tile-usage patterns to avoid (auto-mode, A3/A5)

Compact list. Detailed entries in [auto_mode_bad_patterns.md](auto_mode_bad_patterns.md).

1. **`Tile<...> t; ... TASSIGN(t, addr_runtime);`** — tile addresses are pinned in auto mode. (§1.2)
2. **`TASSIGN(a, X); TASSIGN(b, X);`** to alias — auto mode no-ops `TASSIGN`. Use `TRESHAPE`/`TSUBVIEW`. (§1.1)
3. **`reinterpret_cast<uintptr_t>(tile.data())`** — vector type, not pointer. (§1.3)
4. **`tile.data()` from kernel code** — library-only API. (§3.3)
5. **`__cce_get_tile_ptr(tile.data())` from kernel code** — same. (§3.2)
6. **`__cce_get_tile_ptr(tmpTile + N)`** — pointer arithmetic before extraction crashes libexpand. (§3.2.1)
7. **`Tile<TileType::Bias, ...>` on a non-cube target** — `MemoryQualifier::type` becomes `uint64_t`. (§5.3)
8. **`ConvTile<..., elementSize * sizeof(T), ...>`** — `BufferSize_` is element count. (§5.6)
9. **`__tf__ helper<TileData>(TileData::TileDType dst, TileData::TileDType src, TileData::TileDType tmp)`** — single template across mismatched roles. (§5.5)
10. **`TLOAD(dstTile, dstGlobal)` on a write-only dst** — auto-allocator may coalesce. (§1.5)
11. **`Tile&` parameters into `__tf__` helpers** — pass `TileDType` by value with `__in__`/`__out__` instead. (§3.3 fix; PR-852 [§T4a](external_context/pr_852_notes.md))
12. **`ConvTile` with dynamic dims in auto mode** — `GetShape` returns `staticShape` only; runtime values are ignored ([pto_tile.hpp:1124-1129](../include/pto/common/pto_tile.hpp#L1124-L1129)). Use static dims.
13. **`Bias` falling back to `uint64_t`** when the build does not define `__DAV_C220_CUBE__` ([memory.hpp:81-82](../include/pto/common/memory.hpp#L81-L82)). Only use Bias on a cube target.
14. **`TRESHAPE(a, b)` used as memory reuse rather than semantic aliasing.** `TRESHAPE` is an aliasing/view hint (e.g., a different-layout or reinterpreted view of the same data). Aliasing two tiles that both hold independent live values is a correctness bug, not a memory-saving optimization — the auto allocator already coalesces non-overlapping liveness. Add a short comment at every view call naming (a) which tile owns the data, (b) prefix / offset / reinterpret intent, (c) why lifetimes do not conflict.

   **Operator selection convention** (used by [topk_kernel.cpp](../kernels/automode/a2a3/topk/topk_kernel.cpp)):

   | Intent | Form |
   |---|---|
   | Same-type prefix slice | `TSUBVIEW(view, tile, 0, 0)` |
   | Same-type non-zero slice | `TSUBVIEW(view, tile, rowOffset, colOffset)` |
   | Reshape / reinterpret / type-pun (element type, layout, or dimensionality changes) | `TRESHAPE(view, tile)` |

   `TSUBVIEW(x, y, 0, 0)` may be effectively similar to `TRESHAPE(x, y)` for same-type prefix views in the current toolchain, but `TSUBVIEW` is semantically clearer because it explicitly means "take a slice/prefix" and is more robust against future optimization passes that may treat the two ops differently. Reserve `TRESHAPE` for true reshape / reinterpret cases — e.g., [topk_kernel.cpp Phase 5](../kernels/automode/a2a3/topk/topk_kernel.cpp) reinterprets the packed `(val, idx)` float buffer as `uint32` so `TGATHER P1010` can extract index slots.
15. **Mixing source-element and packed-element widths after `TSORT32`.** `TSORT32`'s destination is `srcWidth * 2 * TYPE_COEF` wide (packed (val, idx) pairs). Subsequent `TMRGSORT` loop bounds, `FillMrgArray<>` schedules, tail-block clip caps, scratch-tile widths/types, and final `TGATHER` prefix views must use the **packed** widths, not the source widths. Define `kPackedCols = kCols * 2 * TYPE_COEF` and `kPackedTopK = kTopK * 2 * TYPE_COEF` and thread them through. Using raw source widths silently drops the second half of the TSORT32 output before it reaches the merge. (See [auto_mode_bad_patterns.md §5.7](auto_mode_bad_patterns.md), [known_good_kernel_examples.md §A12](known_good_kernel_examples.md).)

---

## 12. Open assumptions and things to verify

- **`tile_size(...)` keyword** — not defined in any repo header (verified by grep). Inferred to be a bisheng-CCE compiler keyword/builtin. Reach out to compiler docs to confirm it consumes element counts, not bytes.
- **`__cce_tinit(...)`** — used at [pto_tile.hpp:1135, 1454, 1464, 1475, 1485](../include/pto/common/pto_tile.hpp#L1135) to "dummy-initialize" a `TileDType` in auto mode (against the SROA pass producing undef). Not `#define`d in any repo header; Inferred to be a CCE builtin.
- **`TileLeft` definition split** at [pto_tile.hpp:1682-1684 vs 1692-1694](../include/pto/common/pto_tile.hpp#L1682-L1694) (RowMajor vs ColMajor base) — the surrounding `#ifdef` was not visible in the grep. **Unknown** whether A3 picks RowMajor and A5 picks ColMajor (or vice versa). Read the conditional before committing to either form for a new kernel.
- **`SFractalSize = 512` for non-FP16 element types on A5** — the in-tree A5 MX matmul uses `512` for both AB Mat tiles regardless of element width ([tmatmul_mx_kernel.cpp:95-97](../tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp#L95-L97)). Whether that is intentional or a leftover from FP16 — Unknown.
- **Dynamic-region `SetValidShape` interaction with auto-sync** inside `__tf__` — Unknown. The header comment says PIPE_S sync is required ([pto_tile.hpp:1608](../include/pto/common/pto_tile.hpp#L1608)).
- **`ConvTile` dynamic-shape support** — auto mode's `GetShape` returns `staticShape` only ([pto_tile.hpp:1124-1129](../include/pto/common/pto_tile.hpp#L1124-L1129)). Some manual-mode tests (e.g., dynamic `srcG`/`srcN` in TTrans) appear to rely on the runtime branch — Unknown whether the same kernels are auto-mode-eligible without rewriting the shape to static.
- **`is_tile_data_v` vs `is_conv_tile_v`** — separate traits ([pto_tile.hpp:1738, 1759](../include/pto/common/pto_tile.hpp#L1738)). The current `TRESHAPE_IMPL` only accepts the former, blocking `ConvTile` aliasing pre-PR-852 (§10.1).
- **Whether the A5 file mirrors of the PR-852-fixed A3 headers contain the same bugs** — full audit Unknown (spot-checks negative; see [external_context/pr_852_notes.md "Scope"](external_context/pr_852_notes.md)).
- **`tload_gm2mat` (A3) and `tload_shape2d` (A5) tests** use raw `__cce_get_tile_ptr(tile.data())` chains and ARE in `ALL_TESTCASES`. Whether they actually compile under `--cce-enable-pto-passes` — Unknown (carried from [auto_mode_bad_patterns.md §3.2, §6.3](auto_mode_bad_patterns.md)).

---

## 13. Caveat on testcase inclusion as evidence

Inclusion in `ALL_TESTCASES` ([tests/npu/a2a3/src/st/testcase/CMakeLists.txt](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt), [tests/npu/a5/src/st/testcase/CMakeLists.txt](../tests/npu/a5/src/st/testcase/CMakeLists.txt)) is **build-coverage** evidence, not **clean-style** evidence. Several in-list test kernels carry manual-mode idioms that this doc still flags as risky:

- [tquant_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tquant/tquant_kernel.cpp) — `TASSIGN(srcTile, 0x0); TASSIGN(dstS8Tile, 0x0)` overlap (mixed). [auto_mode_bad_patterns.md §6.1](auto_mode_bad_patterns.md).
- [tdequant_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tdequant/tdequant_kernel.cpp) — `TLOAD(dstTile, dstGlobal)` on a write-only dst. [auto_mode_bad_patterns.md §6.2](auto_mode_bad_patterns.md).
- [tload_gm2mat_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tload_gm2mat/tload_gm2mat_kernel.cpp) and [tload_shape2d_kernel.cpp](../tests/npu/a5/src/st/testcase/tload_shape2d/tload_shape2d_kernel.cpp) — `__cce_get_tile_ptr(tile.data())` chains. [auto_mode_bad_patterns.md §6.3](auto_mode_bad_patterns.md).
- [textract_kernel.cpp](../tests/npu/a5/src/st/testcase/textract/textract_kernel.cpp) — direct `aTile.data()` in kernel body. [auto_mode_bad_patterns.md §6.4](auto_mode_bad_patterns.md).
- [tcolexpandmax_kernel.cpp](../tests/npu/a2a3/src/st/testcase/tcolexpandmax/tcolexpandmax_kernel.cpp) and siblings — unguarded kernel-scope `pipe_barrier(PIPE_ALL)`. [auto_mode_bad_patterns.md §2.2](auto_mode_bad_patterns.md).

When using a testcase as a tile-pattern reference, treat any of the above as **mixed / risky / manual-mode-curated** and prefer the patterns in [known_good_kernel_examples.md Group A](known_good_kernel_examples.md) instead.
