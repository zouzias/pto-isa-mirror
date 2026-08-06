# A5 VfSim Costmodel User Guide

The A5 VfSim costmodel is the vector function performance model used by PTO-ISA for the A5 generation under `__COSTMODEL`. It does not execute on real NPU hardware. Instead, it runs the A5 CCE mock PTO/tileop path in a normal C++ process, captures VF micro-ops and loop structure inside `__VEC_SCOPE__`, constructs `VfInfo`, and calls the VfSim native C++ model to predict VF cycles.

## Use Cases

This costmodel is intended for the following scenarios:

- Predicting the execution time of VF bodies in A5 vector tileops.
- Comparing performance trends across tile shapes, repeat counts, dtypes, and register/UB access patterns.
- Quickly validating A5 tileop costmodel results without NPU hardware or full NPU ST execution.

The prediction result is intended for performance modeling and trend analysis. It is not a replacement for hardware profiling.

## Basic Flow

The current A5 VfSim costmodel flow is:

```text
PTO/tileop execution
  -> A5 CCE mock captures VF micro-ops
  -> LLVM pass instrumentation captures for-loops inside __VEC_SCOPE__
  -> PTO-side VfInfo is constructed from micro-op events and loop events
  -> PredictVfCycles(...) is called when the PTO instruction finishes
  -> PTO VfInfo is converted to VfSim VfInfo
  -> VfSim native C++ model predicts cycles
  -> Predicted cycles are written back to the PTO trace / tile last cycle
```

Key points:

- Micro-op capture is implemented by CCE intrinsic mock stubs.
- For-loop capture is implemented by LLVM pass instrumentation.
- `VfInfo` is the interface data structure between PTO-ISA and the VfSim costmodel.
- One `__VEC_SCOPE__` corresponds to one `VfInfo`.
- If one PTO/tileop contains multiple `VfInfo` objects, each `VfInfo` is predicted independently and the final result is the sum of all predicted cycles.
- If no `VfInfo` is generated, the path does not enter VfSim and keeps the normal PTO costmodel cycle result.

## Quickly Run Existing Tests

The current A5 VfSim ST tests are located under:

```text
tests/costmodel/st_a5
```

Build and run:

```bash
cd /path/to/pto-isa

cmake -S tests/costmodel/st_a5 \
      -B /path/to/build/st_a5 \
      -DCMAKE_BUILD_TYPE=Debug \
      -DPTO_A5_LLVM_CONFIG=/path/to/llvm-config

cmake --build /path/to/build/st_a5 -j4
ctest --test-dir /path/to/build/st_a5 --output-on-failure
```

If `PTO_A5_LLVM_CONFIG` is not specified, the CMake script tries to find `llvm-config`, `llvm-config-20`, or `llvm-config-19`, and then locates the matching `clang++`.

Run a single group of tests, for example `tadd`:

```bash
ctest --test-dir /path/to/build/st_a5 \
      --output-on-failure \
      -R tadd
```

Currently covered basic vector tileop examples include:

```text
tadd, tadds, tsub, tsubs, tmul, tmuls, tdiv,
tmin, tmins, tmax, tand, tshls, tshrs
```

## Add a New A5 Costmodel Test Case

### 1. Use the Unified Entry Header

User code should include the PTO unified entry:

```cpp
#include <pto/pto-inst.hpp>
```

Do not directly include internal VfSim headers in test cases or user code.

Under `__COSTMODEL`, `pto/pto-inst.hpp` selects the PTO costmodel implementation. Under `__NPU_ARCH__=3101`, it enters the A5 CCE mock and VfSim settlement path.

### 2. Use Costmodel Compile Macros

A5 VfSim costmodel tests should at least define:

```bash
-D__COSTMODEL
-D__NPU_ARCH__=3101
```

In most host-only tests, also define:

```bash
-DPTO_COMM_NOT_SUPPORTED
```

Meaning of the macros:

| Macro | Meaning |
| --- | --- |
| `__COSTMODEL` | Enables the PTO costmodel path instead of the normal device execution path. |
| `__NPU_ARCH__=3101` | Selects the A5 architecture path. |
| `PTO_COMM_NOT_SUPPORTED` | Disables or bypasses communication/runtime paths that are not required in host-side costmodel tests. |

### 3. Clang/LLVM Pass Is Required

A5 VfSim requires loop information inside `__VEC_SCOPE__`.

Micro-ops are captured by CCE mock stubs, but loop structure is captured by LLVM pass instrumentation. Therefore, for cases that need real VfSim prediction, compile the test with `clang++` and the A5 loop pass.

Directly using `g++` is not recommended because:

- `g++` does not support Clang's `-fpass-plugin` option.
- The loop structure inside `__VEC_SCOPE__` cannot be captured.
- `VfInfo` may be incomplete or missing.
- If the native VfSim library is not linked, the path may fall back instead of using real VfSim prediction.

### 4. VfSim Must Be Built and Linked

Historically, most PTO-ISA features can be used in a header-only style: user code includes `pto/pto-inst.hpp`, and compile macros select the costmodel path.

The A5 VfSim costmodel is not fully header-only. The real VfSim prediction backend contains native C++ implementation files. PTO-ISA must build the `pto_a5_vfsim` library first, and the user test target or costmodel case must link against that library.

A complete setup therefore requires all of the following:

- Include the PTO unified entry: `#include <pto/pto-inst.hpp>`.
- Define A5 costmodel macros at compile time: `-D__COSTMODEL -D__NPU_ARCH__=3101`.
- Use Clang to load the loop instrumentation pass: `-fpass-plugin=/path/to/PtoLoopTracePass.so`.
- Build and link the native VfSim backend: `-lpto_a5_vfsim`.

If the code only includes headers but does not build or link `pto_a5_vfsim`, a single-file lit case or manual compile command usually cannot enter the full native VfSim path. It may fall back, or it may fail with an undefined reference to `PredictVfCyclesWithVfSim`.

### 5. Recommended CMake Integration

The recommended integration method is to use the existing A5 helper:

```text
cmake/a5_vf_mock.cmake
```

Example:

```cmake
include(${PTO_ISA_ROOT}/cmake/a5_vf_mock.cmake)

_pto_a5_find_llvm()
set(CMAKE_CXX_COMPILER ${PTO_A5_CLANGXX})

project(<your_project_name> CXX)

add_definitions(-D__COSTMODEL -D__NPU_ARCH__=3101 -DPTO_COMM_NOT_SUPPORTED)

add_executable(<your_a5_case_target> main.cpp)
target_include_directories(<your_a5_case_target> PRIVATE ${PTO_ISA_ROOT}/include)
target_enable_a5_vf_mock(<your_a5_case_target>)
```

`target_enable_a5_vf_mock(<your_a5_case_target>)` attaches the A5 VfSim costmodel compile, instrumentation, and link configuration to the user's own target. It is responsible for:

- Building and linking the `pto_a5_vfsim` native library.
- Building the `PtoLoopTracePass` LLVM pass.
- Adding compile options similar to `-O0 -g -fpass-plugin=$<TARGET_FILE:PtoLoopTracePass>`.
- Adding the required include paths and target dependencies.

For a complete example, refer to:

```text
tests/costmodel/st_a5/CMakeLists.txt
tests/costmodel/st_a5/testcase/CMakeLists.txt
tests/costmodel/st_a5/testcase/tadd/CMakeLists.txt
```

### 6. Manual Compile Command Reference

Manual compilation is not recommended for daily use, but the required options are roughly:

```bash
/path/to/clang++ main.cpp \
  -std=c++23 -O0 -g \
  -D__COSTMODEL \
  -D__NPU_ARCH__=3101 \
  -DPTO_COMM_NOT_SUPPORTED \
  -fpass-plugin=/path/to/PtoLoopTracePass.so \
  -I/path/to/pto-isa/include \
  -L/path/to/pto-isa/build/lib \
  -lpto_a5_vfsim \
  -Wl,-rpath,/path/to/pto-isa/build/lib \
  -o <your_a5_case_binary>
```

Notes:

- The `clang++` major version should match the LLVM major version used to build the pass plugin.
- `-fpass-plugin` is required for loop capture.
- `-lpto_a5_vfsim` is required for the native VfSim backend.
- If the environment relies on the `PTO_A5_VFSIM_LINKED` guard, make sure the linked-library path defines it consistently. The exact behavior depends on the current branch implementation.

## Write Test Cases

### 1. Basic Test Structure

A typical test case should follow this structure:

```cpp
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

#include "common/a5_vfsim_tileop_check.hpp"

TEST(A5VfSimTAdd, Basic)
{
    pto::mocker::ResetTrace();

    // Construct tile/global/tensor arguments and call the PTO tileop.
    // For example: TADD(dst, src0, src1);

    const auto cycles = pto::mocker::GetLastPtoInstrCycles();
    EXPECT_GT(cycles, 0U);
}
```

For a concrete example, refer to:

```text
tests/costmodel/st_a5/testcase/tadd/main.cpp
```

### 2. ResetTrace

`::pto::mocker::ResetTrace()` clears the costmodel trace of the current thread.

Call it before each test case to avoid the previous case's PTO instruction, VF information, or pipeline state affecting the current case.

Example:

```cpp
TEST(A5VfSimTAdd, Case1)
{
    pto::mocker::ResetTrace();
    // call tileop
    EXPECT_GT(pto::mocker::GetLastPtoInstrCycles(), 0U);
}

TEST(A5VfSimTAdd, Case2)
{
    pto::mocker::ResetTrace();
    // call another tileop
    EXPECT_GT(pto::mocker::GetLastPtoInstrCycles(), 0U);
}
```

Without `ResetTrace()`, `GetTrace().executed_pto` may contain PTO instructions from both Case1 and Case2, making it difficult to determine which instruction is being checked.

### 3. Get the Predicted Cycle

After the PTO/tileop call finishes, get the last PTO instruction cycle with:

```cpp
uint64_t cycles = ::pto::mocker::GetLastPtoInstrCycles();
```

If the tile object supports `SetLastCycle(...)`, `MAP_INSTR_IMPL` writes the current costmodel cycle to the tile object, so test helpers can read it from the tile as well.

### 4. Check the Expected Micro-op Sequence

The existing A5 tests use helper functions such as:

```cpp
pto::test::a5::ExpectLastBinaryVecTileOp({"vlds", "vlds", "vadd", "vsts"}, repeat);
```

This line checks that the VF micro-op sequence captured by the last tileop is the expected sequence, and that the repeat count matches expectation.

Its purpose is to avoid tests that only check `cycles > 0` but do not verify whether the case actually entered the expected VF path.

## Fallback Strategy

VfSim does not currently cover all A5 tileops or all micro-ops.

The A5 costmodel may use fallback in the following cases:

- The tileop or VF micro-op is not supported by VfSim yet.
- The constructed `VfInfo` is incomplete, for example missing loop information.
- The LLVM pass is not enabled, so loop structure is not captured.
- The native VfSim library is not linked, so the single-file path uses inline fallback.
- The test enters a full host runtime, communication, or syncall path, but the corresponding mock intrinsic is incomplete.

Fallback usually returns:

```text
repeat times * single-instruction cycle
```

or a conservative estimate from the formula costmodel.

Fallback ensures that tests and basic prediction can continue, but it is not equivalent to native VfSim prediction.

## How to Tell Whether VfSim Is Really Used

Check the following items:

1. The CMake log contains a message similar to:

   ```text
   a5_vf_mock: <target> enabled pass instrumentation (-O0 -g -fpass-plugin)
   ```

2. The compile command contains:

   ```text
   -fpass-plugin=...
   ```

3. The target links:

   ```text
   pto_a5_vfsim
   ```

4. The test checks the VF micro-op sequence, for example:

   ```cpp
   ExpectLastBinaryVecTileOp(...)
   ```

5. `GetLastPtoInstrCycles()` changes with repeat count, shape, or VF structure.

If the compile command only uses `g++`, does not contain `-fpass-plugin`, or does not link `pto_a5_vfsim`, it usually does not use the full native VfSim prediction path.

## Relationship with Perf-Sim

Perf-Sim is the pipeline-level operator simulation framework. It focuses on whole-operator execution behavior across AIC, AIV, MTE, CUBE, VEC, and MTE3.

The A5 VfSim costmodel focuses on VF execution time inside A5 vector functions. In PTO-ISA, it can be used as the backend model for the VF part of the costmodel. When a PTO instruction finishes, the VF cycle predicted by VfSim is written back to the PTO instruction result.

Conceptually:

| Component | Scope |
| --- | --- |
| Perf-Sim | Whole PTO kernel or pipeline-level timing. |
| A5 VfSim costmodel | Execution time of one or more VF structures. |
| Fallback | Conservative estimate when VfSim is unsupported or information is insufficient. |

## FAQ

### Why can't I use only g++?

A5 VfSim needs LLVM pass instrumentation to capture loop structure inside `__VEC_SCOPE__`. `g++` cannot load Clang LLVM pass plugins through `-fpass-plugin`, so it cannot produce complete loop information. In that case, the path usually falls back.

### What is `-fpass-plugin`?

`-fpass-plugin` is a Clang option used to load an LLVM pass plugin. In this project, it loads the A5 loop instrumentation pass and inserts recording logic for loops inside `__VEC_SCOPE__` at compile time.

### What is `-lpto_a5_vfsim`?

`-lpto_a5_vfsim` links the native C++ VfSim model library embedded in PTO-ISA.

The A5 VfSim backend is not a purely header-only implementation. If a single-file lit case compiles without linking this library, it may enter inline fallback or produce an undefined reference to `PredictVfCyclesWithVfSim`.

### Why do I see errors such as `aclrtMallocHost` or `aclrtFreeHost` not declared?

This usually means a full host wrapper path entered the costmodel compile flow, but the ACL runtime mock or include order is incomplete.

Short-term workaround: force pre-including the costmodel entry header in the compile command.

Long-term solution: complete the ACL runtime stub in:

```text
include/pto/costmodel/common/aclrt_stub.hpp
```

### Why do I see errors such as `ld_dev`, `st_atomic`, or `ATOMIC_SUM` not declared?

These are CCE/device-side intrinsics or constants. They are normally provided by the real CCE compiler environment, but host-side costmodel tests need mock definitions.

If the full syncall or communication path is pulled into the A5 costmodel build, add the corresponding intrinsic mocks to the runtime stub, or guard that path under `PTO_COMM_NOT_SUPPORTED`.

### Why do I see TLOAD/TSTORE intrinsic errors?

The real A5 `TLoad.hpp` and `TStore.hpp` may pull in MTE intrinsics. Host-side A5 costmodel tests generally should not include the real MTE implementation directly. They should use A5 costmodel-specific TLOAD/TSTORE stubs instead.

### Why does the result always look like fallback?

Check whether:

- The compiler is `clang++`, not `g++`.
- `__COSTMODEL` and `__NPU_ARCH__=3101` are defined.
- `-fpass-plugin` is enabled.
- `pto_a5_vfsim` is linked.
- The tileop and VF micro-ops are supported by VfSim.
- The executed code path actually enters `__VEC_SCOPE__`.

### How are multiple VFs handled?

For a single `VfInfo`, `PredictVfCycles(...)` returns one predicted cycle value.

For multiple `VfInfo` objects, PTO-ISA calls VfSim for each `VfInfo` independently, then sums all predicted cycle values and returns one scalar result.
