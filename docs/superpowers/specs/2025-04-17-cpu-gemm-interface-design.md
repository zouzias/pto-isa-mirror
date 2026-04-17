# CPU Backend Gemm Interface Design

**Date:** 2025-04-17
**Author:** Claude Code
**Status:** Approved

## Overview

This document describes the design for introducing a Gemm interface placeholder in the PTO-ISA CPU backend matmul implementation. The goal is to extract the core matrix multiplication logic into a simple, replaceable function that can be filled in later with optimized implementations.

## Objectives

- **Minimal Change:** Modify only the necessary code in a single file
- **Preserve Functionality:** Keep all existing features (bias, accumulation, type checking)
- **Placeholder Pattern:** Create a Gemm function that can be easily replaced later
- **C-Style Interface:** Use raw pointers for maximum flexibility
- **No External Impact:** Maintain all existing TMatmul interfaces unchanged

## Architecture

### Current Architecture

```
TMATMUL_IMPL → TMatmulNzZn → [Direct computation loop with parallel_for]
```

### New Architecture

```
TMATMUL_IMPL → TMatmulNzZn → Gemm(dst, acc, src0, src1, M, N, K) → [Placeholder computation]
```

The key change is extracting the core computation loop from `TMatmulNzZn` into a standalone `Gemm` function. All type checking, Tile encapsulation, and parameter extraction logic remain unchanged.

## Gemm Interface Design

### Function Signature

```cpp
namespace pto {
namespace cpu {

template <typename DType>
void Gemm(DType* dst, const DType* acc,
          const DType* src0, const DType* src1,
          uint16_t M, uint16_t N, uint16_t K);

} // namespace cpu
} // namespace pto
```

### Parameters

- **dst:** Output matrix pointer (row-major layout, M x N)
- **acc:** Accumulation pointer (nullptr = no accumulation, non-null = add to existing values)
- **src0:** Left matrix pointer (row-major layout, M x K)
- **src1:** Right matrix pointer (row-major layout, K x N)
- **M, N, K:** Matrix dimensions

### Supported Types

- `float` (f32 x f32 → f32)
- `int32_t` (s8 x s8 → s32, after accumulation)
- Can be extended to other types as needed

## Implementation Details

### Modified File

**File:** `include/pto/cpu/TMatmul.hpp`

### New Gemm Function

```cpp
namespace pto {
namespace cpu {

template <typename DType>
void Gemm(DType* dst, const DType* acc,
          const DType* src0, const DType* src1,
          uint16_t M, uint16_t N, uint16_t K)
{
    // Placeholder implementation - simple triple loop
    for (uint16_t i = 0; i < M; i++) {
        for (uint16_t j = 0; j < N; j++) {
            DType sum = acc ? acc[i * N + j] : static_cast<DType>(0);

            for (uint16_t k = 0; k < K; k++) {
                sum += src0[i * K + k] * src1[k * N + j];
            }

            dst[i * N + j] = sum;
        }
    }
}

} // namespace cpu
} // namespace pto
```

### Modified TMatmulNzZn Function

**Before:**
```cpp
template <typename TileAcc, typename TileLeft, typename TileRight>
void TMatmulNzZn(typename TileAcc::TileDType dst, typename TileAcc::TileDType acc,
                 typename TileLeft::TileDType src0, typename TileRight::TileDType src1,
                 uint16_t M, uint16_t N, uint16_t K)
{
    cpu::parallel_for_1d(0, M, static_cast<std::size_t>(M) * N * K, [&](std::size_t i) {
        for (uint16_t j = 0; j < N; j++) {
            typename TileAcc::DType mul_acc = 0;

            PTO_CPU_VECTORIZE_LOOP
            for (uint16_t k = 0; k < K; k++) {
                size_t src0Idx = GetTileElementOffset<TileLeft>(i, k);
                size_t src1Idx = GetTileElementOffset<TileRight>(k, j);
                mul_acc += static_cast<typename TileAcc::DType>(src0[src0Idx]) *
                           static_cast<typename TileAcc::DType>(src1[src1Idx]);
            }

            size_t dstIdx = GetTileElementOffset<TileAcc>(i, j);
            dst[dstIdx] = acc ? acc[dstIdx] + mul_acc : mul_acc;
        }
    });
}
```

**After:**
```cpp
template <typename TileAcc, typename TileLeft, typename TileRight>
void TMatmulNzZn(typename TileAcc::TileDType dst, typename TileAcc::TileDType acc,
                 typename TileLeft::TileDType src0, typename TileRight::TileDType src1,
                 uint16_t M, uint16_t N, uint16_t K)
{
    using DType = typename TileAcc::DType;
    cpu::Gemm<DType>(dst, acc, src0, src1, M, N, K);
}
```

### Changes Summary

1. **Removed:** `parallel_for_1d` and `PTO_CPU_VECTORIZE_LOOP` decorators
2. **Removed:** `GetTileElementOffset` calls (using linear indexing instead)
3. **Added:** Call to `cpu::Gemm()` with raw pointers
4. **Preserved:** All type checking in `CheckMadValid()`
5. **Preserved:** All wrapper functions (TMATMUL_IMPL, TGEMV_IMPL, etc.)

## Error Handling

**Strategy:** Minimal error handling

- Rely on existing `CheckMadValid()` for compile-time type checking
- No runtime checks in `Gemm` function (null pointer, dimension validation)
- Preserve all existing assertions in TMatmul layer

**Rationale:**
- CPU simulator inputs come from compile-time validated Tile objects
- Avoid redundant checking overhead
- Align with "minimal change" principle

## Testing Strategy

### Existing Tests

No test modifications required. All existing CPU test cases in `tests/cpu/st/testcase/tmatmul/` will continue to work.

**Verification:**
- Functional: Output should match pre-modification results exactly
- No performance requirements (placeholder implementation)

### Future Verification (After User Fills Gemm)

When you replace the Gemm implementation:
- Correctness: Numerical precision validation
- Performance: Comparison with naive implementation and/or reference libraries

## Git Workflow

### Branch Management

```bash
# Create feature branch
git checkout -b feature/cpu-gemm-interface

# Make changes to TMatmul.hpp
# Stage and commit
git add include/pto/cpu/TMatmul.hpp
git commit -m "Add Gemm interface placeholder for CPU backend matmul"

# Push to remote
git push origin feature/cpu-gemm-interface
```

### Commit Message Guidelines

- Use clear, descriptive commit messages
- Reference this design document in commit description
- Follow existing project commit message style

## Future Extension

### Replacing Gemm Implementation

To replace the placeholder with an optimized GEMM library:

```cpp
template <typename DType>
void Gemm(DType* dst, const DType* acc,
          const DType* src0, const DType* src1,
          uint16_t M, uint16_t N, uint16_t K)
{
    // Call optimized GEMM library
    // Example: cblas_sgemm(), your own implementation, etc.
}
```

**No further modifications needed to TMatmul.hpp**

### Extension Points

- Add multiple Gemm implementations with compile-time selection
- Add runtime switching between different backends
- Support for additional data types (half, bfloat16, int8)

## Design Rationale

### Why This Approach?

1. **Minimal Change:** Only ~30 lines changed in a single file
2. **Clear Separation:** Gemm function is isolated and self-contained
3. **Easy Replacement:** User can modify Gemm without touching Tile logic
4. **Preserves Compatibility:** All existing code continues to work
5. **Flexible Interface:** C-style raw pointers work with any GEMM library

### Alternative Approaches Considered

1. **Wrapper Layer:** Rejected - requires new file, more complex
2. **Macro Abstraction:** Rejected - harder to read and debug
3. **Strategy Pattern:** Rejected - over-engineering for simple use case

## Acceptance Criteria

- [x] Design approved by user
- [ ] Gemm function implemented in TMatmul.hpp
- [ ] TMatmulNzZn modified to call Gemm
- [ ] All existing CPU tests pass
- [ ] Code committed to feature branch
- [ ] Design document committed to Git

## References

- Original TMatmul implementation: `include/pto/cpu/TMatmul.hpp`
- CPU tests: `tests/cpu/st/testcase/tmatmul/`
- Project overview: `CLAUDE.md`
