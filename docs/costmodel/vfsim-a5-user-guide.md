# A5 VfSim Costmodel User Guide

The A5 VfSim costmodel is the vector function performance model used by PTO-ISA for the A5 generation under `__COSTMODEL`. It does not execute on real NPU hardware. Instead, it runs the A5 CCE mock PTO/tileop path in a normal C++ process, captures VF micro-ops and loop structure inside `__VEC_SCOPE__`, constructs PTO `VfInfo`, lowers it directly to `CanonicalVfInfo`, and calls the VfSim native C++ model to predict VF cycles.

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
  -> predictVfCycles(...) is called when the PTO instruction finishes
  -> PTO VfInfo is lowered directly to VfSim CanonicalVfInfo
  -> VfSim native C++ model predicts cycles and a structured hit status
  -> The result is written to the PTO trace and Perf-Sim instruction record
```

Key points:

- Micro-op capture is implemented by CCE intrinsic mock stubs.
- For-loop capture is implemented by LLVM pass instrumentation.
- PTO `VfInfo` is the internal capture structure; `CanonicalVfInfo` is the only native VfSim input.
- One `__VEC_SCOPE__` corresponds to one `VfInfo`.
- If one PTO/tileop contains multiple `VfInfo` objects, each `VfInfo` is predicted independently and the final result is the sum of all predicted cycles.
- If no `VfInfo` is generated, the path does not enter VfSim and keeps the normal PTO costmodel cycle result.
- VF cycles become the Vector pipeline latency in Perf-Sim and are not charged twice.

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

LLVM/Clang 19 is recommended. If `PTO_A5_LLVM_CONFIG` is not specified, the CMake script searches for `llvm-config` and the matching-major `clang++`. Other LLVM/Clang major versions are not rejected, but Clang and the LLVM used to build the pass plugin must have the same major version.

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

The suite also contains adapter golden, memory/sync routing, ACL Host Runtime, unchanged full-Host TADD, and relocatable-config regression tests.

“Supported tileop” refers to the exact dtype/form combinations covered by these regressions; it does not imply that every dtype of the same tileop hits VfSim. An uncovered form returns an observable `UnsupportedForm` or `InvalidTrace` status and uses the A5 fallback.

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

**Clang/LLVM 19 is recommended.** The currently validated loop-capture boundary is regular `for` loops compiled with `-O0` and `-g`. Other matching-major Clang/LLVM toolchains may be used, but their compatibility should be covered by the integrator's regression tests. The current pass relies on loop `DebugLoc` and a preheader; a loop missing either may be skipped. Complex control flow, other optimization levels, and loop ownership beyond the `__VEC_SCOPE__` destructor boundary are not currently claimed as supported scenarios.

Directly using `g++` is not recommended because:

- `g++` does not support Clang's `-fpass-plugin` option.
- The loop structure inside `__VEC_SCOPE__` cannot be captured.
- `VfInfo` may be incomplete or missing.
- If the native VfSim library is not linked, the real prediction symbol is unavailable and linking fails.

### 4. VfSim Must Be Built and Linked

Historically, most PTO-ISA features can be used in a header-only style: user code includes `pto/pto-inst.hpp`, and compile macros select the costmodel path.

The A5 VfSim costmodel is not fully header-only. The real VfSim prediction backend contains native C++ implementation files. PTO-ISA must build the `pto_a5_vfsim` library first, and the user test target or costmodel case must link against that library.

The `pto_a5_vfsim` static archive contains both the PTO adapter and all VfSim native-core object files. A consumer only needs to link `libpto_a5_vfsim.a`; `libvfsim_native_core.a` is not an additional link dependency.

A complete setup therefore requires all of the following:

- Include the PTO unified entry: `#include <pto/pto-inst.hpp>`.
- Define A5 costmodel macros at compile time: `-D__COSTMODEL -D__NPU_ARCH__=3101`.
- Use Clang to load the loop instrumentation pass: `-fpass-plugin=/path/to/PtoLoopTracePass.so`.
- Build and link the native VfSim backend: `-lpto_a5_vfsim`.

If the code only includes headers but does not build or link `pto_a5_vfsim`, a single-file lit case or manual compile command fails with an undefined reference to `predictVfCyclesWithVfSim`.

### 5. Recommended CMake Integration

The recommended integration method is to use the existing A5 helper:

```text
pkg_inc/pto/costmodel/vfsim/cmake/a5_vf_mock.cmake
```

Example:

```cmake
include(${PTO_ISA_ROOT}/pkg_inc/pto/costmodel/vfsim/cmake/a5_vf_mock.cmake)

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
- Fully synchronizing VfSim JSON into the relative build layout `pkg_inc/pto/costmodel/vfsim/configs` on every build, preventing stale configurations when a build directory is reused.

This helper lives in the internal `pkg_inc` tree and is installed together with the VfSim sources and configurations. `PTO_ISA_ROOT` must name the architecture directory that contains both `include/` and `pkg_inc/`. External projects should enable the A5 costmodel through this helper instead of depending directly on the native VfSim C++ API.

Standard Clang cannot parse Ascend `kernel<<<...>>>` syntax. The current A5 costmodel accepts the same Host-compilable cases as CPU_SIM: the launch wrapper must call the kernel function directly. NPU test sources that still contain `<<<...>>>` are not accepted by this build entry, and the costmodel does not rewrite user source code.

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

To distinguish a real hit from fallback, inspect the structured result:

```cpp
const auto& result = ::pto::mocker::GetTrace().executed_pto.back().vfPrediction;
EXPECT_EQ(result.status, ::pto::mocker::vf::VfPredictionStatus::VfSimHit);
EXPECT_GT(result.vfsimHitCount, 0U);
EXPECT_EQ(result.fallbackCount, 0U);
```

The native VfSim does not currently model predicate registers. The PTO adapter explicitly filters
`plt_b8`, `plt_b16`, and `plt_b32`, while the remaining supported instructions may still report
`VfSimHit`. This is an approximate prediction: inspect `result.ignoredInstructionCount` and
`result.diagnostics` to identify filtered instructions. PLT cycles and predicate dependencies are not
included yet.

Other predicate, carry, or multi-output instructions that are captured completely but have no native VfSim
model return `UnsupportedForm` and use the A5 formula fallback. `InvalidTrace` remains reserved for missing
or contradictory capture information.

Logging and configuration lookup are controllable as well:

```cpp
::pto::mocker::vf::VfPredictionOptions options;
options.logLevel = ::pto::mocker::vf::VfSimLogLevel::Summary;
options.configDir = "/path/to/vfsim-or-configs";
::pto::mocker::vf::SetVfPredictionOptions(options);
```

Log levels are `Off`, `Errors`, `Summary`, and `Detailed`. Configuration lookup order is `options.configDir`, `PerfSimConfig::vfsim_config_dir`, `PTO_VFSIM_CONFIG_DIR`, and finally a relocatable installation layout near the executable or current directory.

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
- Configuration JSON is missing or the simulator throws an exception.
- Capture information for the current dtype/form is incomplete.

Fallback usually returns:

```text
repeat times * single-instruction cycle
```

or a conservative estimate from A5 `vec_cycle_generated.hpp`. A5 fallback does not read A2/A3 formula parameters.

Fallback keeps basic prediction available, but it is not equivalent to native VfSim prediction. Callers should inspect the structured status instead of checking only `cycles > 0`.

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

4. The test calls `ExpectLastVfSimHit()` or directly asserts `VfPredictionStatus::VfSimHit`.

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

The A5 VfSim backend is not a purely header-only implementation. A single-file lit case that does not link this library produces an undefined reference to `predictVfCyclesWithVfSim`.

### How is ACL Runtime handled by a full Host main?

`include/pto/costmodel/common/aclrt_stub.hpp` provides Host mocks for initialization, device and stream state, Host/Device allocation, Memcpy/Memset, synchronization, and release. These APIs only keep control flow valid so execution reaches the kernel; they add zero cycles and do not model real ACL asynchronous scheduling. `target_enable_a5_vf_mock` adds the costmodel stub include path.

### Why do I see errors such as `ld_dev`, `st_atomic`, or `ATOMIC_SUM` not declared?

These are CCE/device-side intrinsics or constants. They are normally provided by the real CCE compiler environment, but host-side costmodel tests need mock definitions.

The A5 costmodel uses a dedicated sync mock: VF-internal `mem_bar` enters VfSim, while external `pipe_barrier`, events, and SyncAll enter Perf-Sim. It does not include the device-atomic path from `syncall_soft.hpp`.

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

For a single `VfInfo`, `predictVfCycles(...)` returns one predicted cycle value.

For multiple `VfInfo` objects, PTO-ISA calls VfSim for each `VfInfo` independently, then sums all predicted cycle values and returns one scalar result.
