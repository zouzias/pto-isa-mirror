/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

# PTO Costmodel

`include/pto/costmodel` is the host-side trace backend for PTO when `__COSTMODEL` is enabled.

It is intended for:

- compiling NPU PTO headers on a normal host toolchain
- replacing CCE/CANN intrinsics with trace-oriented stubs
- running small host-side demos that print PTO and CCE traces

It is not a numeric simulator. If you need host-side numerical behavior, use the CPU backend under
`include/pto/cpu/`.

## Folder Layout

```text
include/pto/costmodel/
  common/        shared host runtime pieces
  a2a3/          A2/A3-specific CCE stubs
  a5/            A5-specific CCE stubs and compat helpers
  evaluator/     trace-based latency evaluator
  runtime_stub.hpp
  trace.hpp
  arch_config.hpp
```

Main entry points:

- `runtime_stub.hpp`: common host runtime facade included by PTO costmodel builds
- `trace.hpp`: shared PTO/CCE trace recording
- `a2a3/cce_stub.hpp`: A2/A3 trace-only intrinsic stubs
- `a5/cce_stub.hpp`: A5 trace-only intrinsic stubs

## Arch Selection

Costmodel arch is selected the same way as the NPU backend: by `__NPU_ARCH__`.

Current test mappings:

- `a2a3` -> `__NPU_ARCH__=2201`
- `a5` -> `__NPU_ARCH__=3101`

The test build system exposes this through the CMake cache variable:

```text
PTO_COSTMODEL_TEST_ARCH=a2a3
PTO_COSTMODEL_TEST_ARCH=a5
```

The test runner also accepts the environment variable:

```bash
PTO_COSTMODEL_TEST_ARCH=a2a3
PTO_COSTMODEL_TEST_ARCH=a5
```

If unset, the default is `a2a3`.

## Running Tests

See [tests/README.md](/home/lc/pto-isa/tests/costmodel/README.md) for the exact commands.

Short version:

```bash
# default arch: a2a3
./tests/costmodel/run.sh all

# select A5
PTO_COSTMODEL_TEST_ARCH=a5 ./tests/costmodel/run.sh all

# run a few cases only
PTO_COSTMODEL_TEST_ARCH=a5 ./tests/costmodel/run.sh tload tadd tstore
```

## What the Tests Do

Each testcase is a small host executable that:

1. resets the shared trace
2. runs one PTO path or direct stub calls
3. prints the recorded PTO/CCE trace

The tests do not use `EXPECT_*`. They are trace demos intended to verify:

- the selected arch compiles
- the expected PTO path is reachable
- the expected intrinsic stub calls are emitted

## Adding New Arch Support

When adding a new NPU arch to the costmodel backend, keep the split consistent:

- put shared host runtime code in `common/`
- put arch-specific CCE stubs in `<arch>/cce_stub.hpp`
- keep `trace.hpp` shared
- add tests under `/tests/costmodel/<arch>/`
- teach `/tests/costmodel/run.sh` and `/tests/costmodel/CMakeLists.txt` how to select the new arch

## Current Scope

The current costmodel backend focuses on readable trace output and lightweight host compilation. Some PTO/NPU paths may
still need additional stub coverage before they can compile under `__COSTMODEL`.

codex resume 019d3d48-250d-7e93-bb17-c45990213ffa
