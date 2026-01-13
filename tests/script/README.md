# tests/script/

Entry scripts for building and running the repository test suites.

## NPU ST (sim / npu)

- Build + run: `tests/script/run_st.py`
- Build only: `tests/script/build_st.py`
- Parallel (NPU + CPU): `tests/script/run_st_parallel.py`

Common arguments:

- `-r, --run-mode`: `sim` or `npu`
- `-v, --soc-version`: `a2`, `a3`, or `a5` (mapped to an internal `SOC_VERSION`)
- `-t, --testcase`: testcase name (e.g., `tmatmul`)
- `-d, --debug-enable`: optional debug build (only in `run_st.py`)

Examples:

```bash
python3 tests/script/run_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
python3 tests/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1
```

### Parallel NPU ST

`run_st_parallel.py` builds once, generates goldens, then runs test binaries in
parallel (default: 1 worker process per healthy NPU). For each NPU testcase run,
it also runs the matching CPU ST binary (when available) and reports both times.

```bash
python3 tests/script/run_st_parallel.py -v a3 --timeout-sec 30
python3 tests/script/run_st_parallel.py -v a5 --timeout-sec 30
```

`run_st_parallel.py` runs all NPU ST testcases by default.
Use `--testcases` to restrict the run, or `--include-long` (reserved for future long/perf exclusions).
Use `--skip-cpu` to disable the CPU side run.

Deadlock handling:

- `--timeout-sec` (or `PTO_ST_TIMEOUT_SEC`) is the **base timeout per gtest**:
  - each testcase binary uses `base * (# gtests in the binary)`
- `--abort-on-timeout` (default: enabled) stops scheduling new tasks after the first timeout and marks remaining tasks as `SKIP`.
- Optional: `PTO_ST_DEADLOCK_REGEX` (default: `deadlock`) kills a test early if the pattern appears in its stdout.

`tests/run_st.sh` uses the parallel runner by default for `npu all`.

```bash
PTO_ST_TIMEOUT_SEC=30 ./tests/run_st.sh a3 npu all
PTO_ST_SERIAL=1 ./tests/run_st.sh a3 npu all
PTO_ST_PARALLEL_ARGS="--timeout-sec 180" ./tests/run_st.sh a3 npu all
```

## CPU ST

- Batch build + run: `tests/script/all_cpu_tests.py`

Options:

- `-v, --verbose`: print build/run output
- `-b, --build-folder`: build directory (default: `build_tests`)

Example:

```bash
python3 tests/script/all_cpu_tests.py --verbose
```

## Convenience Wrappers

- Recommended suites: `run_st.sh`
- CPU tests: `run_cpu_tests.sh`

For the latest arguments, run `python3 <script> -h`.
