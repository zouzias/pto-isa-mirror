# AscendC API vs PTO (A2A3 ST) — usability comparison

Scope: Compare **API usability (shape/type deduction, boilerplate)** using existing ST kernels under `tests/npu/a2a3/src/st/testcase/`.
Focus cases:
- Elementwise: **tadd**
- Reduce: **tcolsum**
- Broadcast/expand: **trowexpand**
- Cast/convert: **tcvt**

> Note: AscendC snippets below are **conceptual** (based on CANN AscendC API list) to show typical usage pattern (pointer/struct-based), not exact compilable code. PTO snippets reference real code in this repo.

---

## 1) Elementwise Add (tadd)
**PTO (real code):** `tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp`
- Uses `Tile<TileType::Vec, T, kTRows_, kTCols_...>` to **deduce dtype/shape at compile time**.
- `GlobalTensor` + `Tile` objects carry stride/shape.
- Core is 1 line: `TADD(dstTile, src0Tile, src1Tile, event0)`.

**AscendC (conceptual):**
- Typical style uses `LocalTensor` / pointer + explicit shape/stride metadata.
- Requires explicit `Copy`/`DataCopy` or buffer binding, then `Add`.
- Mask/stride must be manually set if non‑contiguous.

**Usability verdict:** PTO is **more user‑friendly**: tile templates eliminate a lot of boilerplate and reduce shape/type mismatch risk. AscendC is **more verbose** and easy to mis‑specify strides.

---

## 2) Reduce Sum (tcolsum)
**PTO (real code):** `tests/npu/a2a3/src/st/testcase/tcolsum/tcolsum_kernel.cpp`
- Shape/stride in `GlobalTensor` and `Tile`.
- Core operation is `TCOLSUM(dstTile, srcTile, tmpTile, IsBinary)`.
- Only need to set UB addresses via `TASSIGN`.

**AscendC (conceptual):**
- Likely uses `ReduceSum / BlockReduceSum / WholeReduceSum` (CANN AscendC API list).
- Must specify **repeat counts, block stride, mask**, and potentially multiple calls for tail.
- More **structure‑heavy**: separate `SetMaskNorm/Count`, `SetVectorMask`, `Copy`, then `ReduceSum`.

**Usability verdict:** PTO wins for **code brevity and clarity**. AscendC offers more control but is **more error‑prone** (manual mask/stride settings).

---

## 3) Broadcast/Expand (trowexpand)
**PTO (real code):** `tests/npu/a2a3/src/st/testcase/trowexpand/trowexpand_kernel.cpp`
- Uses `TROWEXPAND(dstTile, srcTile)`.
- Special case uses BRCB when `dst_col * sizeof(T) == 32B`.
- Tile types carry layout info (RowMajor/ColMajor) and stride.

**AscendC (conceptual):**
- Would use `Brcb` (block broadcast) or elementwise ops + `Copy`.
- You need to explicitly manage data layout and strides.

**Usability verdict:** PTO is **simpler** for broadcast because the API is already at the vector‑tile level; AscendC requires more manual layout handling.

---

## 4) Cast/Convert (tcvt)
**PTO (real code):** `tests/npu/a2a3/src/st/testcase/tcvt/tcvt_kernel.cpp`
- Core call: `TCVT(dstTile, srcTile, RoundMode::CAST_RINT)`.
- Dtype is deduced by Tile template (`Tile<T...>` and `Tile<S...>`).

**AscendC (conceptual):**
- Uses `Cast` / `CastDeq` with explicit dtype info.
- Again requires pointer/shape/stride descriptors.

**Usability verdict:** PTO is **more user‑friendly** for dtype conversion because the template types avoid redundant dtype parameters.

---

# Summary — Which is easier for ST‑style kernels?

| Case | PTO (Tile‑template) | AscendC (pointer/struct) | Winner (usability) |
|---|---|---|---|
| Elementwise add | Minimal, type/shape inferred | More setup (tensor descriptors + mask) | **PTO** |
| Reduce sum | One API call; fewer manual strides | Manual mask/stride/repeat | **PTO** |
| Broadcast/expand | One API call; layout encapsulated | Brcb/Copy w/ manual layout | **PTO** |
| Cast | Dtype inferred from templates | Explicit dtype/descriptor | **PTO** |

**Overall:** For **ST‑style kernels** (single op + minimal plumbing), **PTO is more user‑friendly** because the template tiles **deduce shape/type/stride** and reduce boilerplate. AscendC is closer to raw HW control and gives more flexibility, but requires **more manual configuration** and is easier to mis‑specify.

---

## References (in this repo)
- tadd: `tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp`
- tcolsum: `tests/npu/a2a3/src/st/testcase/tcolsum/tcolsum_kernel.cpp`
- trowexpand: `tests/npu/a2a3/src/st/testcase/trowexpand/trowexpand_kernel.cpp`
- tcvt: `tests/npu/a2a3/src/st/testcase/tcvt/tcvt_kernel.cpp`

## AscendC API list (external)
- CANN AscendC API list: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/opdevgapi/atlasascendc_api_07_0004.html
