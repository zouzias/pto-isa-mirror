# tests/

Tests and examples for PTO Tile Lib, covering both CPU simulation and NPU (including `sim` and on-board `npu` modes).

## Test Entry Points

Common test entry points:

- Full CPU Simulator run: `python3 tests/run_cpu.py --clean --verbose`
- GEMM demo: `python3 tests/run_cpu.py --demo gemm --verbose`
- Flash Attention demo: `python3 tests/run_cpu.py --demo flash_attn --verbose`
- Single ST testcase: `python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] -t [TEST_CASE] -g [GTEST_FILTER_CASE]`
- One-click scripts: `./tests/run_st.sh --a3 --sim --all` (see note below), `./tests/run_cpu_tests.sh`

> `run_st.sh` requires a platform flag (`--a3`/`--a5`/`--a3_a5`/`--kirin9030`) **and**, for `--a3`/`--a5`, a mode flag (`--simple` or `--all`); optionally a run mode (`--sim`/`--npu`, defaults to on-board `npu`). Run `./tests/run_st.sh` with no/invalid arguments to print the full usage. Note the `--` prefixes are required.

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
- `run_pipeline.sh`: Pipeline test script
- `validate_op_coverage.py`: Operator coverage validation script
- `validate_testcase_names.py`: Testcase name validation script

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

Engine-specific test names containing `_async` follow the same default exclusion. The HNS1825 GET target must be
selected explicitly with `-t tget_async_hns1825`.

### HNS1825 RDMA Async Tests (A5)

The HNS1825 tests require A5, an HNS1825 RDMA NIC and driver, HCOMM, and reachable RDMA NIC IPv4
addresses. Select the backend before the build is configured:

```bash
export PTO_RDMA_BACKEND=HNS_1825
python3 tests/script/run_st.py -r npu -v a5 -t comm/tput_async_hns1825 -d -n 2
```

`PTO_RDMA_BACKEND` is read by CMake during configuration and translated into identical host/kernel compile
definitions. It is not read by the generated test binary. Unset, empty, and unsupported values build without RDMA.
`run_st.py` rebuilds from a fresh `build/` directory unless `-w/--without-build` is supplied; do not use `-w` after
changing the backend.

The test bootstrap resolves each rank's physical device id, RDMA NIC IPv4 address, and registered-buffer device
address, then exchanges the values over MPI. Its variables are:

| Variable | Scope | Description |
|---|---|---|
| `PTO_RDMA_BACKEND` | Configure time | Only `HNS_1825` is supported; other values disable RDMA for this build. |
| `PTO_ROCE_ROOTINFO` | ST only | Root-info JSON path; defaults to `/etc/hccl_rootinfo.json`. |
| `PTO_ROCE_PHYIDS` | ST only | Optional comma-separated physical device ids indexed by rank. |
| `PTO_ROCE_LOCAL_IP` | ST only | Fallback local RDMA IPv4 for a rank when root-info resolution fails. |
| `PTO_ROCE_IPS` | ST only | Optional comma-separated RDMA IPv4 list indexed by rank. |
| `PTO_ROCE_BASE_PORT` | ST only | Common channel base port; defaults to `60032`. |
| `PTO_ROCE_VERBOSE` | Control plane/ST | Set to `1` for detailed endpoint, MR, channel, and cleanup progress. |
| `HCCL_RDMA_TC` | HCOMM/RoCE | Traffic class; PTO defaults to `132`. |
| `HCCL_RDMA_SL` | HCOMM/RoCE | Service level; PTO defaults to `4`. |

If the HNS1825 verbs provider is not discovered from a default provider path, set `IBV_EXTEND_DRIVERS` to the
driver-provided `libhrn5-rdmav34.so`. This is a deployment requirement of the HCOMM/libibverbs stack, not a PTO backend
selector.

Both HNS1825 PUT and GET have passed in the target environment. GET remains a dedicated target so READ and WRITE
regressions can be run and diagnosed independently. It can be built with:

```bash
export PTO_RDMA_BACKEND=HNS_1825
cmake -S tests/npu/a5/comm/st -B tests/npu/a5/comm/st/build \
  -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_9599 -DTEST_CASE=tget_async_hns1825
cmake --build tests/npu/a5/comm/st/build -j
```

Run it explicitly with:

```bash
python3 tests/script/run_st.py -r npu -v a5 -t comm/tget_async_hns1825 -d -n 2
```

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

`run_st.py -w/--without-build` skips compilation and runs an existing binary. It is a
`run_st.py` option, not a `run_comm_test.sh` option; configure-time environment
changes such as `PTO_RDMA_BACKEND` do not affect a reused binary.

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-n` | Number of available NPUs: 2, 4, or 8 | 8 |
| `-v` | SoC version: `a3` (Ascend910B) or `a5` (Ascend950) | a3 |
| `-t` | Run specific testcase(s) (repeatable), e.g. `tput`, `treduce` | all |
| `-a` | Include regular async instruction tests (names containing `_async`); HNS1825 GET remains an explicit target | off |
| `-d` | Enable debug mode with verbose init/sync logging | off |

### How It Works

The script automatically runs each testcase at each applicable rank count (2 / 4 / 8, up to `-n`), using GTest filters to select only the tests matching the current rank count. For example, with `-n 4` it first runs default tests at 2 ranks, then tests with the `4Ranks` suffix at 4 ranks, skipping 8-rank tests.

## Suggested Reading

- Getting started (recommended: CPU first, then NPU): [docs/getting-started.md](../docs/getting-started.md)
