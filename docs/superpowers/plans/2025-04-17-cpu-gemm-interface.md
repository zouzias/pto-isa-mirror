# CPU Backend Gemm Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-step. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gemm interface placeholder to the PTO-ISA CPU backend matmul implementation, enabling easy replacement of the core computation logic with optimized GEMM libraries.

**Architecture:** Extract the matrix multiplication computation loop from `TMatmulNzZn` into a standalone `cpu::Gemm()` template function. This function uses C-style raw pointers and can be replaced without modifying any Tile wrapper logic.

**Tech Stack:** C++20 templates, C-style pointer interface, existing PTO-ISA Tile infrastructure

---

## File Structure

**Modified files:**
- `include/pto/cpu/TMatmul.hpp` - Add Gemm function and simplify TMatmulNzZn

**No new files created**

**Test files (for validation only, no modifications):**
- `tests/cpu/st/testcase/tmatmul/main.cpp` - Existing matmul tests
- `tests/run_cpu.py` - CPU test runner

---

## Task 1: Create Feature Branch

- [ ] **Step 1: Create and checkout feature branch**

```bash
cd /path/to/pto-isa
git checkout -b feature/cpu-gemm-interface
```

Expected output: `Switched to a new branch 'feature/cpu-gemm-interface'`

- [ ] **Step 2: Verify branch**

```bash
git branch
```

Expected output: `* feature/cpu-gemm-interface` (asterisk indicates current branch)

---

## Task 2: Add Gemm Function to TMatmul.hpp

- [ ] **Step 1: Open TMatmul.hpp for editing**

File location: `include/pto/cpu/TMatmul.hpp`

- [ ] **Step 2: Locate the namespace**

Find the closing brace of namespace `pto` around line 211 (the end of the file). You'll see:

```cpp
} // namespace pto
#endif
```

- [ ] **Step 3: Add cpu::Gemm function before the closing pto namespace**

Insert the following code just before `} // namespace pto` (after line 210, before the closing namespace):

```cpp
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
```

**Important:** This must be inside `namespace pto { ... }` but creates a nested `namespace cpu { ... }`

- [ ] **Step 4: Verify the file structure**

After insertion, the end of your file should look like:

```cpp
    TGEMV_MX_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                  TileRightScale &bScaleMatrix, TileBias &biasData)
    {
        (void)Phase;
        (void)aScaleMatrix;
        (void)bScaleMatrix;
        TGEMV_BIAS_IMPL(cMatrix, aMatrix, bMatrix, biasData);
    }
} // namespace pto

namespace pto {
namespace cpu {

template <typename DType>
void Gemm(DType* dst, const DType* acc,
          const DType* src0, const DType* src1,
          uint16_t M, uint16_t N, uint16_t K)
{
    // ... (implementation)
}

} // namespace cpu
} // namespace pto
#endif
```

Wait - this is wrong. Let me fix this. The `cpu` namespace should be **inside** the `pto` namespace, not as a separate namespace block after pto closes.

**Corrected approach:** The file already has other cpu implementations. Check if there's already a `pto::cpu` namespace section in the file.

Let me check the actual file structure first:

```bash
grep -n "namespace" /path/to/include/pto/cpu/TMatmul.hpp | head -20
```

If you see `namespace pto {` at the top and `} // namespace pto` at the end, and no existing `namespace cpu`, then add the cpu namespace **inside** the pto namespace, not outside.

**Correct insertion:** Add this code **inside** namespace pto, but create a nested cpu namespace:

```cpp
} // namespace pto  <-- This is the OLD closing brace

// REPLACE with:

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

So the closing braces should be: `} // namespace cpu` then `} // namespace pto`

- [ ] **Step 5: Save the file**

- [ ] **Step 6: Verify syntax with a quick compile check**

```bash
cd /path/to/pto-isa
python3 tests/run_cpu.py --testcase tmatmul --gtest_filter 'TMatmulTest*' 2>&1 | head -50
```

Expected: Compilation should succeed (though tests may not have been modified yet, the new Gemm function should be syntactically valid)

---

## Task 3: Modify TMatmulNzZn to Use Gemm

- [ ] **Step 1: Locate TMatmulNzZn function**

In `include/pto/cpu/TMatmul.hpp`, find the `TMatmulNzZn` function around lines 18-38.

The current implementation looks like:

```cpp
template <typename TileAcc, typename TileLeft, typename TileRight>
void TMatmulNzZn(typename TileAcc::TileDType dst, typename TileAcc::TileDType acc, typename TileLeft::TileDType src0,
                 typename TileRight::TileDType src1, uint16_t M, uint16_t N, uint16_t K)
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

- [ ] **Step 2: Replace TMatmulNzZn implementation**

Replace the entire function body with:

```cpp
template <typename TileAcc, typename TileLeft, typename TileRight>
void TMatmulNzZn(typename TileAcc::TileDType dst, typename TileAcc::TileDType acc, typename TileLeft::TileDType src0,
                 typename TileRight::TileDType src1, uint16_t M, uint16_t N, uint16_t K)
{
    using DType = typename TileAcc::DType;
    cpu::Gemm<DType>(dst, acc, src0, src1, M, N, K);
}
```

**Key changes:**
- Remove `parallel_for_1d` wrapper
- Remove `PTO_CPU_VECTORIZE_LOOP` decorator
- Remove `GetTileElementOffset` calls
- Add `using DType` declaration
- Call `cpu::Gemm<DType>()`

- [ ] **Step 3: Save the file**

- [ ] **Step 4: Verify no other functions need modification**

The other functions (TMATMUL_IMPL, TGEMV_IMPL, TMATMUL_ACC_IMPL, etc.) all call TMatmulNzZn, so they don't need changes.

---

## Task 4: Run CPU Tests to Validate Changes

- [ ] **Step 1: Run tmatmul CPU tests**

```bash
cd /path/to/pto-isa
python3 tests/run_cpu.py --testcase tmatmul
```

Expected: All tests should pass with output similar to:

```
[PASS] tmatmul (XXXms)
```

- [ ] **Step 2: Run specific test case for detailed output**

```bash
python3 tests/run_cpu.py --testcase tmatmul --verbose 2>&1 | grep -A 5 "TMatmulTest"
```

Expected: Should see test cases running and passing

- [ ] **Step 3: Run all CPU tests to ensure no regressions**

```bash
python3 tests/run_cpu.py --clean
```

Expected: All CPU tests should pass. If any fail, check:
- Did you properly add `namespace cpu` inside `namespace pto`?
- Is the Gemm function template parameter correct?
- Are the pointer parameters matching the types?

---

## Task 5: Verify Gemm Interface Works Correctly

- [ ] **Step 1: Check that Gemm can be called directly (optional manual verification)**

Create a simple test file `test_gemm_interface.cpp` in `/tmp/`:

```cpp
#include "include/pto/cpu/TMatmul.hpp"
#include <iostream>
#include <iomanip>

int main() {
    // Simple test: 2x2 * 2x2 = 2x2
    // A = [1 2]    B = [5 6]    C = [19 22]
    //     [3 4]        [7 8]        [43 50]

    float A[4] = {1, 2, 3, 4};  // Row-major: [1,2,3,4]
    float B[4] = {5, 6, 7, 8};  // Row-major: [5,6,7,8]
    float C[4] = {0, 0, 0, 0};

    pto::cpu::Gemm<float>(C, nullptr, A, B, 2, 2, 2);

    std::cout << "Result:" << std::endl;
    std::cout << C[0] << " " << C[1] << std::endl;  // Expected: 19 22
    std::cout << C[2] << " " << C[3] << std::endl;  // Expected: 43 50

    return 0;
}
```

- [ ] **Step 2: Compile and run the test**

```bash
cd /path/to/pto-isa
g++ -std=c++20 -I./include /tmp/test_gemm_interface.cpp -o /tmp/test_gemm
/tmp/test_gemm
```

Expected output:
```
Result:
19 22
43 50
```

If correct, this confirms the Gemm interface is working as expected.

- [ ] **Step 3: Clean up test file**

```bash
rm /tmp/test_gemm_interface.cpp /tmp/test_gemm
```

---

## Task 6: Commit Changes

- [ ] **Step 1: Review changes**

```bash
cd /path/to/pto-isa
git diff include/pto/cpu/TMatmul.hpp
```

Expected changes:
- Added `namespace cpu` with `Gemm` template function (~20 lines)
- Simplified `TMatmulNzZn` function (~5 lines instead of ~20 lines)

- [ ] **Step 2: Stage the modified file**

```bash
git add include/pto/cpu/TMatmul.hpp
```

- [ ] **Step 3: Commit with descriptive message**

```bash
git commit -m "$(cat <<'EOF'
feat: Add Gemm interface placeholder for CPU backend matmul

Extract core matrix multiplication computation from TMatmulNzZn into
a standalone cpu::Gemm() template function using C-style raw pointers.

Changes:
- Add pto::cpu::Gemm<DType>() template function with placeholder impl
- Simplify TMatmulNzZn to delegate to cpu::Gemm()
- Remove parallel_for and vectorization decorators (now in Gemm)
- Remove GetTileElementOffset calls (use linear indexing)

This enables easy replacement of the core GEMM implementation with
optimized libraries (OpenBLAS, custom kernels, etc.) without modifying
Tile wrapper logic or external interfaces.

All existing tests pass. Gemm function can be replaced independently.

Ref: docs/superpowers/specs/2025-04-17-cpu-gemm-interface-design.md

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

Expected output: Commit hash and file summary

---

## Task 7: Push to Remote Repository

- [ ] **Step 1: Push feature branch to remote**

```bash
cd /path/to/pto-isa
git push origin feature/cpu-gemm-interface
```

Expected output:
```
...
 * [new branch]      feature/cpu-gemm-interface -> feature/cpu-gemm-interface
```

- [ ] **Step 2: Verify remote branch**

```bash
git branch -vv
```

Expected: Should show `feature/cpu-gemm-interface` tracking `origin/feature/cpu-gemm-interface`

---

## Task 8: Document Completion

- [ ] **Step 1: Update acceptance criteria in design doc**

Open `docs/superpowers/specs/2025-04-17-cpu-gemm-interface-design.md` and update the Acceptance Criteria section:

```markdown
## Acceptance Criteria

- [x] Design approved by user
- [x] Gemm function implemented in TMatmul.hpp
- [x] TMatmulNzZn modified to call Gemm
- [x] All existing CPU tests pass
- [x] Code committed to feature branch
- [x] Design document committed to Git
- [x] Implementation plan completed
```

- [ ] **Step 2: Stage and commit the documentation update**

```bash
git add docs/superpowers/specs/2025-04-17-cpu-gemm-interface-design.md
git commit -m "docs: Update acceptance criteria - implementation complete"
```

- [ ] **Step 3: Push documentation update**

```bash
git push origin feature/cpu-gemm-interface
```

---

## Post-Implementation Notes

### Future Work: Replacing Gemm Implementation

When you're ready to replace the placeholder Gemm implementation:

1. Open `include/pto/cpu/TMatmul.hpp`
2. Find the `cpu::Gemm` function (lines ~213-233 after modifications)
3. Replace the function body with your optimized implementation

Example with OpenBLAS:
```cpp
template <typename DType>
void Gemm(DType* dst, const DType* acc,
          const DType* src0, const DType* src1,
          uint16_t M, uint16_t N, uint16_t K)
{
    // Handle accumulation
    if (acc) {
        // Copy acc to dst first
        std::memcpy(dst, acc, M * N * sizeof(DType));
    }

    // Call optimized GEMM
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, src0, K, src1, N,
                acc ? 1.0f : 0.0f, dst, N);
}
```

**No other changes needed** - the Tile wrapper layer handles everything else.

### Testing After Gemm Replacement

After replacing Gemm:
1. Run `python3 tests/run_cpu.py --testcase tmatmul`
2. Verify numerical correctness (compare with reference implementation)
3. Measure performance improvement
4. Commit with message: "perf: Replace Gemm placeholder with <implementation>"

---

## Troubleshooting

### Common Issues

**Issue:** Compiler error "cpu::Gemm not found"
- **Fix:** Ensure `namespace cpu { ... }` is inside `namespace pto { ... }`, not after it

**Issue:** Template instantiation errors
- **Fix:** Check that `using DType = typename TileAcc::DType;` is present in TMatmulNzZn

**Issue:** Tests fail with incorrect results
- **Fix:** Verify linear indexing formula: `src0[i*K + k] * src1[k*N + j]`

**Issue:** Accumulation not working
- **Fix:** Ensure the ternary operator `acc ? acc[i*N+j] : 0` is correct

### Getting Help

If you encounter issues not covered here:
1. Check the design document: `docs/superpowers/specs/2025-04-17-cpu-gemm-interface-design.md`
2. Review the original TMatmul.hpp implementation before changes: `git diff HEAD~1 include/pto/cpu/TMatmul.hpp`
3. Run tests with verbose output: `python3 tests/run_cpu.py --testcase tmatmul --verbose`
