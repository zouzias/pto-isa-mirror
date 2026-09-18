# TMATMUL_MX SPLITK tail regression tests

This change repairs the A5 ST kernels `RunTMATMULMX_SPLIT_K` and `RunTGEMVMX_SPLIT_K`, together with their host tests. It does not change the PTO CPU simulator, the public `TLOAD` implementation, or the CANN CAModel binary. CAModel executes compiled NPU instructions on the host; it is a different execution path from PTO's CPU backend.

## Tail handling

`RunTMATMULMX_SPLIT_K` retains a 64-element compute block. The final operand loads use the remaining K for both GM shape and L1 valid shape, while preserving the compact GM row strides. ND2NZ zeros the partial C0 block; `TFILLPAD` zeros unused complete A C0 blocks and B rows before moving the operands to L0. Scale tensors retain their existing aligned ZZ/NN storage.

Previously, K=62 issued an A load with width 64 and row stride 62 (32B/31B for FP4). This crosses source rows and can crash or stall CAModel. Loading a full final B block can read beyond the input allocation.

`RunTGEMVMX_SPLIT_K` retains a 1024-element compute block. Its tail loads only the remaining data, `ceil(tailK/32)` ND A scales, and `ceil(tailK/64)*2` NN B scales. The latter includes the stored NN alignment group when needed. Before tail loads, `TEXPANDS` zeros A's L1 buffer and initializes B's L1 scale storage to E8M0 127 (scale 1). `TFILLPAD` zeros B's unused rows. The A scale load pads its 32-byte L1 block with E8M0 0; no extra ND A scale entries are read from GM.

## Coverage and assertions

There are **55 tests: 19 existing tests and 36 new `splitk_*` regressions**.

The following 15 MATMUL cases use M=47, N=128. Each also has a `_dynamic` variant, for 30 new MATMUL tests in total.

| Cases | Static count | Coverage |
| --- | --- | --- |
| FP4 K=62 | 8 | Every E1M2/E2M1 A/B combination, with and without bias |
| FP8 K=62 | 1 | E4M3/E5M2 without bias |
| FP4/FP8 K=72 | 2 | E1M2/E2M1 and E4M3/E5M2; full split plus 8-element tail; accumulation with bias |
| FP4/FP8 K=64 | 2 | Exact full block; E1M2/E2M1 without bias, E4M3/E5M2 with bias |
| FP8 K=96 | 1 | E4M3/E5M2 without bias; unused complete A C0 block |
| FP4 K=126 | 1 | E2M1/E2M1 without bias; reused L1 with a nearly full final block |

The dynamic variants pass K as a kernel argument and use dynamic GM K dimensions, A row stride, and L1 valid K dimensions for operand loads. Each launcher passes its case's fixed K; loop count, scale layout, M and N remain compile-time values. These tests exercise dynamic operand descriptors, not arbitrary runtime GEMM shapes.

Six new `splitk_gemv_*` cases use M=1, N=64, all with bias: FP4 E1M2/E2M1 at K=62/1024/1032/1056 and FP8 E4M3/E5M2 at K=62/1032. Existing `case16`, `case17`, and `case19` cover multiple complete MATMUL/GEMV splits; `case19` is an unguarded K=2048 control.

New MATMUL tests allocate compact inputs without extra K padding. New GEMV tests append nonzero guard bytes after each compact A/B and scale payload, excluding bias. These bytes lie outside the logical tensors and must not affect the result; they expose full-block tail overreads even when CAModel permits access to that memory. The ND A scale payload has exactly `ceil(K/32)` entries. Generated ZZ/NN alignment padding uses E8M0 127 so tiny scales do not hide stale tail data.

All 36 new tests compare every output element against NumPy over the original K using `ASSERT_FLOAT_EQ` (GTest's floating-point comparison, not bitwise equality). Existing tests retain `ResultCmp`. The host checks Runtime calls, new input payload sizes, golden file size, and output file writing; resources are released on early assertion failures.

## Run

Use manual mode with an A5 CANN environment, a compatible GTest installation, and Python packages `numpy`, `en_dtypes`, and `ml_dtypes`. The repository's A5 auto-mode target list does not include `tmatmul_mx`.

From the repository root, the standard runner builds, generates data, and runs the **36 new regressions**:

```bash
timeout -k 10s 1800s python tests/script/run_st.py \
  -r sim -v a5 -t tmatmul_mx -g 'TMATMULMXTest.splitk_*'
```

Use `-g 'TMATMULMXTest.*'` for all 55 tests. The timeout bounds the whole build/generation/test command; it is not a per-case timeout.

For a prebuilt simulator executable, use the CANN installation and SoC used to build it. The example below assumes the build directory is `build/camodel_splitk_fix` and CANN provides the simulator libraries under `lib`:

```bash
# Run from the repository root, after sourcing the installed CANN environment.
export SOC_VERSION=Ascend950PR_9599
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
```

Generate data in the build directory, where `GetGoldenDir()` expects it relative to `bin`:

```bash
cd build/camodel_splitk_fix
python ../../tests/npu/a5/src/st/testcase/tmatmul_mx/gen_data.py
cd bin
mkdir -p camodel_log
export CAMODEL_LOG_PATH="$PWD/camodel_log"
timeout -k 10s 1800s ./tmatmul_mx --gtest_filter='TMATMULMXTest.*'
```

The host links `runtime_camodel` in sim mode and `runtime` in npu mode. Unused ACL dependencies are excluded to avoid also loading the hardware Runtime in sim mode. The same tests can run with `-r npu` on matching A5 hardware. A build success or an initialization failure is not evidence of a passing hardware test.

## Validation recorded on 2026-09-18

All 55 formal GTest cases passed in a single audit batch using the final executable and freshly generated inputs on aarch64 CANN 9.0.0 / Ascend950PR_9599 CAModel. Each case ran in a separate process with a 90-second timeout. The original GEMV implementation failed the guarded K=1032 test; the repaired implementation passed. Truncated golden data and failed output writes were also confirmed to fail the host test.

All 55 formal GTest cases also passed on Ascend950PR hardware with x86_64 CANN 9.2.0 (compiler build dated 2026-07-14). The 19 existing cases and 36 new regressions ran sequentially in one process in 374.931 seconds, with no failures or skipped cases. Device health remained OK after execution.

Both sim and npu targets compiled. The original external `pto_test` repository and the report's specific CANN 9.2 nightly were not verified. These results establish the tested caller-side fix, not a repair of CAModel itself.
