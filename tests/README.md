# tests/

Tests and examples for PTO Tile Lib, covering both CPU simulation and NPU (including `sim` and on-board `npu` modes).

## Test Entry Points

> All ST suites (CPU / NPU / Comm / CostModel) are built on GoogleTest. A system
> installation is preferred and required for the NPU suites; see
> [Prerequisites: GTest](#prerequisites-gtest) below.

Common test entry points:

- Full CPU Simulator run: `python3 tests/run_cpu.py --clean --verbose`
- GEMM demo: `python3 tests/run_cpu.py --demo gemm --verbose`
- Flash Attention demo: `python3 tests/run_cpu.py --demo flash_attn --verbose`
- Single ST testcase: `python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] -t [TEST_CASE] -g [GTEST_FILTER_CASE]`
- One-click scripts: `./tests/run_st.sh --a3 --sim --all` (see note below), `./tests/run_cpu_tests.sh`

> `run_st.sh` requires a platform flag (`--a3`/`--a5`/`--a3_a5`/`--kirin9030`) **and**, for `--a3`/`--a5`, a mode flag (`--simple` or `--all`); optionally a run mode (`--sim`/`--npu`, defaults to on-board `npu`). Run `./tests/run_st.sh` with no/invalid arguments to print the full usage. Note the `--` prefixes are required.

## A5 UB ND-to-L1 NZ Regression Tests

Conversion cases explicitly use `TileCopyMode::ND2NZ`. Default calls are checked against
byte-exact legacy copy results (2 Null cases per instruction), including nonzero window offsets.
The CPU cases check the explicit overloads against element-wise layout semantics.

Cases belong to their instruction directories under `npu/a5/src/st/testcase/` and are
registered in the existing test targets:

| Instruction | Test target | GTest filter | ND-to-NZ cases |
|-------------|-------------|--------------|----------------|
| TMOV | `tmov_ub2l1` | `TMovUb2l1Test.nd2nz_*` | 14 |
| TEXTRACT | `textract` | `TEXTRACTTest.nd2nz_*` | 14 |
| TINSERT | `tinsert` | `TInsertTest.nd2nz_*` | 20 |

```bash
python3 tests/script/run_st.py -r npu -v a5 -t tmov_ub2l1
python3 tests/script/run_st.py -r npu -v a5 -t textract -g 'TEXTRACTTest.nd2nz_*:TEXTRACTTest.legacy_null_*'
python3 tests/script/run_st.py -r npu -v a5 -t tinsert -g 'TInsertTest.nd2nz_*:TInsertTest.legacy_null_*'
```

The first command also runs the 9 NZ-input cases in `tmov_ub2l1`, for 63 cases across
these commands, including 6 default Null-copy cases.
Each instruction registers its cases and generates inputs and byte-exact golden data in its existing `main.cpp`,
including checks that data outside the valid window is preserved.
Kernel implementations and explicit launch specializations live in `tmov_ub2l1_kernel.cpp`,
`textract_acc2mat_kernel.cpp`, and `tinsert_kernel.cpp`, respectively.
TEXTRACT reuses the existing mixed-kernel target for its AIV/AIC transfers.
Large NZ results are initialized and read back in chunks of at most 128 KiB,
with synchronization before reusing the UB buffer.

TMOV and TEXTRACT cover all 10 supported data types; TINSERT also covers `int32_t` (11 types).
Boundary cases include FP4 offsets, static and dynamic valid shapes, zero rows and columns,
extraction at a smaller source valid-window edge, insertion at a smaller destination valid-window edge,
dual-AIV FP4 insertion, and static/dynamic
FP4 widths of 65536 columns, including a larger source row stride.
The manual A5 `--simple` and `--all` entry points include this regression coverage.
These cases require manual Tile address aliasing and are not registered in auto mode.
In auto mode, `tmov_ub2l1` retains its 9 NZ-input cases; `textract` and `tinsert` are
excluded by the existing auto-mode target list.

CPU reference cases cover float and both packed FP4 types with byte-exact checks, including padding.
Debug builds also check insertion windows that exceed the destination valid shape while fitting physical storage.
Run them with:

```bash
python3 tests/run_cpu.py --rebuild --build-type Debug --testcase tmov --gtest_filter 'TMOVTest.nd2nz_*'
```

Four `TINSERT` ND copy cases cover a `CompactMode::Normal` Mat destination, dynamic valid shapes,
row and column offsets, and unchanged data outside the copied window. Run them with:

```bash
python3 tests/script/run_st.py -r npu -v a5 -t tinsert -g 'TInsertTest.case_nd_compact_normal_*'
```

## Layout

- `script/`: Recommended entry scripts
  - `run_st.py`: Build and run NPU ST
  - `build_st.py`: Build NPU ST only
  - `all_cpu_tests.py`: Build and run CPU ST suites in batch
  - `cpu_bfloat16.py`: CPU bfloat16 test script
  - `README.md`: Script usage
- `cpu/`: CPU-side ST tests (gtest + CMake)
  - `st/`: CPU compute ST projects and testcase data generation scripts
  - `comm/st/`: CPU communication ST
- `npu/`: NPU-side ST tests split by SoC
  - `a2a3/src/st/`: A2/A3 compute ST
  - `a2a3/src/common/`: A2/A3 shared test resources
  - `a2a3/comm/st/`: A2/A3 communication ST
  - `a5/src/st/`: A5 compute ST
  - `a5/src/common/`: A5 shared test resources
  - `a5/comm/st/`: A5 communication ST
  - `kirin9030/src/st/`: Kirin9030 compute ST
  - `kirin9030/src/common/`: Kirin9030 shared test resources
- `costmodel/`: Cost model tests
  - `st/`: Cost model ST (operator cost measurement)
  - `st_fit/`: Cost model fitting tests
- `common/`: Shared test resources
- `run_cpu.py`: CPU simulator full run entry point
- `run_cpu_tests.sh`: CPU ST one-click script
- `run_st.sh`: NPU ST one-click script (requires `--<platform>` and, for a3/a5, `--simple`/`--all`; run with no args for usage)
- `run_comm_test.sh`: Communication ST one-click script (see below)
- `run_costmodel.py`: Cost model run script
- `run_costmodel_tests.sh`: Cost model one-click script
- `validate_op_coverage.py`: Operator coverage validation script
- `validate_testcase_names.py`: Testcase name validation script

## Prerequisites: GTest

All ST suites use GoogleTest:

- **CPU Simulator / CostModel ST** prefer a system GTest (`libgtest-dev`) and fall
  back to fetching googletest v1.14.0 automatically at configure time when it is
  missing.
- **NPU / Comm ST** are compiled with the custom `bisheng` compiler and **require a
  system GTest** (the nested googletest build cannot be reused there, so CMake
  fails with an explicit error if GTest is missing).

```bash
# Ubuntu / Debian
sudo apt install libgtest-dev

# CentOS / RHEL / EulerOS
sudo yum install gtest-devel
```

## Communication Tests (Comm ST)

Communication tests verify multi-device PTO communication primitives (Put / Get / Broadcast / Gather / Scatter / Reduce / Notify / Wait / Test), built on MPI + HCCL.

### Prerequisites: MPI Installation

Communication tests require an MPI environment (MPICH or OpenMPI). Two components are needed at runtime:

1. **`mpirun`**: launches multi-process execution
2. **`libmpi.so`**: loaded at runtime via `dlopen`

#### Install MPICH (Recommended)

```bash
# Ubuntu / Debian
sudo apt install mpich libmpich-dev

# CentOS / RHEL / EulerOS
sudo yum install mpich mpich-devel
# May need to load a module or add to PATH manually:
export PATH=/usr/lib64/mpich/bin:$PATH
```


#### Build MPICH from Source (No Root)

```bash
wget https://www.mpich.org/static/downloads/4.2.3/mpich-4.2.3.tar.gz
tar xzf mpich-4.2.3.tar.gz && cd mpich-4.2.3
./configure --prefix=$HOME/mpich --disable-fortran
make -j$(nproc) && make install
export MPI_HOME=$HOME/mpich
export PATH=$MPI_HOME/bin:$PATH
```

#### Environment Variables

| Variable | Description |
|----------|-------------|
| `MPI_HOME` | MPI installation root; the script searches `$MPI_HOME/bin/mpirun` |
| `MPI_LIB_PATH` | Direct path to `libmpi.so` (overrides default search) |

If `mpirun` is already on `PATH` and `libmpi.so` is in a standard library path, these variables are not required.

#### Verify Installation

```bash
mpirun --version
mpirun -n 2 echo "MPI OK"
```

### Sync vs Async Instruction Tests

Communication tests are split into **sync instructions** (e.g. `tput`, `tget`) and **async instructions** (e.g. `tput_async`, `tget_async`):

| Type | Test Case Examples | CANN Version Required |
|------|-------------------|----------------------|
| Sync | `tput`, `tget`, `treduce`, `tbroadcast`, etc. | CANN 8.x and above |
| Async | `tput_async`, `tget_async` | **CANN 9.0 and above** |

Async instructions depend on the SDMA opapi interface (e.g. `aclnnShmemSdmaStarsQuery`) introduced in CANN 9.0. They will fail on lower CANN versions due to missing symbols. Therefore, `run_comm_test.sh` **excludes async tests by default** — use the `-a` flag to enable them.

### Quick Start

```bash
# Run all tests with 8 NPUs (default A2/A3, excludes async)
./run_comm_test.sh

# Include async instruction tests (requires CANN 9.0+)
./run_comm_test.sh -a

# Run only the async tput testcase
./run_comm_test.sh -t tput_async

# A5 SoC, 2 NPUs
./run_comm_test.sh -v a5 -n 2

# Run only the tput testcase
./run_comm_test.sh -t tput

# Enable debug logging
./run_comm_test.sh -d -t tput
```

You can also run directly via `run_st.py`, which automatically splits execution by rank count:

```bash
# Auto-split tput_async across 2/4/8 ranks
python3 tests/script/run_st.py -r npu -v a3 -t comm/tput_async

# Limit to 2 ranks
python3 tests/script/run_st.py -r npu -v a3 -t comm/tput_async -n 2
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-n` | Number of available NPUs: 2, 4, or 8 | 8 |
| `-v` | SoC version: `a3` (Ascend910B) or `a5` (Ascend950) | a3 |
| `-t` | Run specific testcase(s) (repeatable), e.g. `tput`, `treduce` | all |
| `-a` | Include async instruction tests (names containing `_async`), requires CANN 9.0+ | off |
| `-d` | Enable debug mode with verbose init/sync logging | off |

### How It Works

The script automatically runs each testcase at each applicable rank count (2 / 4 / 8, up to `-n`), using GTest filters to select only the tests matching the current rank count. For example, with `-n 4` it first runs default tests at 2 ranks, then tests with the `4Ranks` suffix at 4 ranks, skipping 8-rank tests.

## Suggested Reading

- Getting started (recommended: CPU first, then NPU): [docs/getting-started.md](../docs/getting-started.md)
