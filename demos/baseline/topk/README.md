# Basic Topk Operator Example

## Overview

This example demonstrates how to implement a Topk operator using PTO, including project setup, build, and execution.

## Supported AI Processors

- A2/A3

## Directory Layout

```
kernels/topk/
├── scripts/
│   └── gen_data.py              # Generates input and golden output
├── CMakeLists.txt               # Build configuration
├── topk_kernel.cpp        # Kernel implementation
├── main.cpp                     # Host-side entry point
└── run.sh                       # Convenience script
```

## Operator Description

### Function

This example implements topk with fixed dimensions `[rows, cols] = [48, 1024]`:

### Specification

| Item        | Value |
| ----------- | ----- |
| OpType      | `topk` |
| Inputs      | `[rows, cols] = [48, 1024]` |
| Output      | `data`, `index` |
| Kernel name | `topk_kernel` |

### Tiling Parameters

The validation platform has 24 cores. The workload is split across cores.

Per-core shape:

- `rows = 2`, `cols = 1024`

## Implementation Notes

### Type definitions

The implementation defines topk representations. Load input data in GM in two tiles, generate index using TCI, use TSort32 to sort each 32 data, use TMrgsort for each tile. Merge sort between two tiles to get a single sorted tile, extract data and index, then store back to gm seperately.

### Pipeline scheduling

This example overlaps data movement and compute using double buffering in L1 and L0 to improve utilization. Synchronization points ensure correct dependencies, including:

- Forward sync: `MTE2 -> Vec -> Vec`
- Reverse sync: `Vec -> MTE2`

Pipeline overview:

![Topk pipeline](../../../docs/figures/topk_pipeline.png)

## Build and Run

1. Configure your Ascend CANN environment (example path):

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Run the example:

```bash
cd ${git_clone_path}/demos/baseline/topk
bash run.sh -r npu -v Ascend910B1
```

If the run succeeds, the output prints:

```text
test success
```
