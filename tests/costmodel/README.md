/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

# PTO Costmodel Tests

These tests are small host executables that run one PTO path under `__COSTMODEL` and print trace results.

## Arch Selection

There are two ways to select the costmodel arch.

### 1. Use `run.sh`

`run.sh` reads `PTO_COSTMODEL_TEST_ARCH` from the environment.

Valid values:

- `a2a3`
- `a5`

Examples:

```bash
# default: a2a3
./tests/costmodel/run.sh all

# explicit A2/A3
PTO_COSTMODEL_TEST_ARCH=a2a3 ./tests/costmodel/run.sh all

# A5
PTO_COSTMODEL_TEST_ARCH=a5 ./tests/costmodel/run.sh all
```

Arch-specific build outputs go to:

```text
tests/costmodel/build/a2a3
tests/costmodel/build/a5
```

### 2. Use CMake directly

`tests/CMakeLists.txt` accepts the cache variable `PTO_COSTMODEL_TEST_ARCH`.

Examples:

```bash
cmake -S tests/costmodel -B tests/costmodel/build/a2a3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DPTO_COSTMODEL_TEST_ARCH=a2a3 \
  -DPTO_COSTMODEL_CASES=all

cmake -S tests/costmodel -B tests/costmodel/build/a5 \
  -DCMAKE_BUILD_TYPE=Release \
  -DPTO_COSTMODEL_TEST_ARCH=a5 \
  -DPTO_COSTMODEL_CASES=all
```

## Running All Tests

Use:

```bash
./tests/costmodel/run.sh all
PTO_COSTMODEL_TEST_ARCH=a5 ./tests/costmodel/run.sh all
```

This does three things:

1. configures CMake for the selected arch
2. builds the requested cases
3. runs each built executable and prints its trace

## Running Selected Tests

Pass testcase names to `run.sh`:

```bash
./tests/costmodel/run.sh tload tadd tstore
PTO_COSTMODEL_TEST_ARCH=a5 ./tests/costmodel/run.sh tdiv tands txor
```

This only builds the requested targets and then runs them.

If you use CMake directly, select cases with `PTO_COSTMODEL_CASES`:

```bash
cmake -S tests/costmodel -B tests/costmodel/build/a5 \
  -DCMAKE_BUILD_TYPE=Release \
  -DPTO_COSTMODEL_TEST_ARCH=a5 \
  -DPTO_COSTMODEL_CASES="tload;tadd;tstore"

cmake --build tests/costmodel/build/a5 --target tload tadd tstore

tests/costmodel/build/a5/bin/tload
tests/costmodel/build/a5/bin/tadd
tests/costmodel/build/a5/bin/tstore
```

## Available Testcases

### `a2a3`

```text
runtime_stub
tload tadd tadds
tsub tsubs
tmul tmuls
tmatmul tmatmul_acc tmatmul_bias
tgemv tgemv_acc tgemv_bias
tmax tmaxs
tmin tmins
tabs tneg
texp tlog tsqrt trsqrt
tnot tand tor
tlrelu trelu
taxpy
tdivs
tstore
```

### `a5`

```text
runtime_stub
tload tadd tadds
tsub tsubs
tmul tmuls
tdiv tdivs
tmax tmaxs
tmin tmins
tabs tneg
texp tlog tsqrt trsqrt
tnot
tand tands
tor tors
txor txors
tshl tshls
tshr tshrs
tlrelu trelu
taxpy
tstore
```

## Adding a New Testcase

Add the testcase under the correct arch folder:

```text
tests/costmodel/a2a3/testcase/<name>/
tests/costmodel/a5/testcase/<name>/
```

Minimal pattern:

- `main.cpp`
- `<name>_kernel.cpp` if needed
- `CMakeLists.txt` with `pto_costmodel_st(<name>)`

Then register the name in:

- `tests/costmodel/<arch>/testcase/CMakeLists.txt`
- `tests/costmodel/run.sh`

## Notes

- Default build type is `Release`.
- `run.sh` uses a per-arch build directory, so `a2a3` and `a5` builds do not overwrite each other.
- Current tests are trace-only; they are intended to show PTO and CCE call flow, not numeric correctness.
