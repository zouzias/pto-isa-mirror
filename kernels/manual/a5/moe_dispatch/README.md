# A5 moe_dispatch

`moe_dispatch` is a dispatch-only A5 / Ascend950 PTO project. It covers token routing metadata, guarded HCCL
remote-window layout, packed input rows, and dispatch-ready signaling. It does not implement FFN/GMM, combine, restore,
or an HCCL collective AllToAllV path.

## Scope

Input:

```text
inputA[M, K] + expertIdx[M, topK]
```

Outputs:

```text
tokenPerExpert
expandedRowIdx
packedA in the owner rank remote window
dispatch-ready signal
```

## AIV Blocks

`--aiv-blocks` is a runtime argument. If it is omitted or set to `0`, the A5 resource table chooses the default.

Current default:

```text
Ascend950 / Ascend950DT_* / Ascend950PR_* -> 40 AIV blocks
```

This follows the current `dispatch_combine_moe` A5 baseline of `20 AIC x 2 AIV ratio = 40 AIV`. This project is
AIV-only, so launch `blockDim` is `aivBlocks` in the first version.

## HCCL Remote Window

The default layout is `GuardedDispatchOnly`:

```text
window + 0
  [headGuard]                 4 KiB, no live payload
  [packedA]                   half[maxOutputSize, K], 512B aligned
  [expandedRowIdx]            int32[M * topK], 512B aligned
  [reservedScratch]
  [tokenPerExpert]            windowBytes - 2 MiB
  [signal]                    windowBytes - 1 MiB
```

HCCL returns window base and size; this project does not assume HCCL automatically skips padding. The head guard is a
project-owned layout rule, based on the A5 `gemm_ar` guarded window pattern. The existing A5 `dispatch_combine_moe`
fusion layout uses `offsetA = 0`; a future fusion-compatible mode must be named explicitly instead of changing this
default silently.

## Build

```bash
source /home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh
cmake -S kernels/manual/a5/moe_dispatch -B /tmp/moe_dispatch_a5_build -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_958b
cmake --build /tmp/moe_dispatch_a5_build --target moe_dispatch -j8
```

Current local A3 machines should use host-only validation:

```bash
bash kernels/manual/a5/moe_dispatch/run.sh --host-golden-only 1
```
