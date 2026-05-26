# dispatch_combine_tile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable PTO-style, HCCL-backed, multi-process/multi-card MoE dispatch/combine example with exactly two
device compute kernels.

**Architecture:** Host code follows `kernels/manual/a2a3/gemm_ar` for MPI/HCCL bootstrap and continuous rank-to-device
mapping. Device code uses PTO tile and comm primitives only: dispatch packs local tokens, publishes count rows, pulls owner
expert payload by `TGET`; combine pushes expert rows back by `TPUT`, waits for peer completion, then restores local output.

**Tech Stack:** C++17 host, Bisheng CCE kernel build, PTO `TLOAD/TSTORE/TGET/TPUT/TNOTIFY/TWAIT`, ACL/HCCL/MPI runtime,
shell `run.sh` orchestration.

---

## Status And Feedback Tracking

This file is also the task tracker. Do not create a separate `TODO.md` unless the project scope changes.

Use the task checkboxes below as the source of truth:

- `[ ]` not started
- `[~]` in progress
- `[x]` completed and locally verified
- `[!]` blocked or needs user decision

When a task finishes, add one short feedback note under that task:

```text
Feedback:
- YYYY-MM-DD: result, verification command/log, remaining risk if any.
```

Current task status:

| Task | Status | Notes |
| --- | --- | --- |
| Task 0: PTO Project Initialization | `[x]` | Initialization contract files created and locally verified |
| Task 1: Buildable Project Scaffold | `[x]` | Buildable scaffold verified |
| Task 2: Explicit Parameters and Layout Calculation | `[x]` | Explicit args and layout summaries verified |
| Task 3: MPI/HCCL Runtime and Continuous Card Mapping | `[x]` | MPI/HCCL window bootstrap verified for 2/3/4 ranks |
| Task 4: Deterministic Data and CPU Golden | `[x]` | Host deterministic data and CPU golden verified |
| Task 5: PTO Kernel Views, Remote Pointer Helper, and Signal Helpers | `[x]` | Kernel view and comm helper scaffold verified |
| Task 6: Dispatch Metadata and Count Publication | `[x]` | Dispatch metadata and count publication verified |
| Task 7: Dispatch Pack and Payload Gather | `[x]` | Dispatch pack and TGET gather verified |
| Task 8: Host Expert Output Preparation and Dispatch Timing | `[x]` | Host identity expert output and dispatch timing verified |
| Task 9: Combine Return Path | `[x]` | TPUT combine return path verified |
| Task 10: Combine Restore and Output Verification | `[x]` | Restore and output verification passed for 2-rank acceptance shapes |
| Task 11: E2E Timing, Debug Dumps, and Run Matrix | `[x]` | 2/3/4-rank matrix passed after per-iteration signal epoch fix |
| Task 12: Hardening and Final Review | `[x]` | Static checks and final default 2-rank run passed |

## Validation Rules

Apply these rules to every task unless the task says otherwise:

- Every run command must finish without hang. Use a local timeout wrapper during manual execution. Small/debug communication
  commands should use `timeout 60s`; default-shape commands should start with `timeout 90s`. If the timeout fires, mark the
  task `[!]`, record the last rank log, and diagnose the last completed stage before retrying. Do not increase timeout until
  logs prove the command is making progress rather than waiting on a missing signal.
- A task is not complete just because it builds. If hardware execution is unavailable, record that explicitly in the task
  `Feedback` and complete only the build/static/code-review checks listed for that task.
- Debug comparisons must print the compared buffer name, element count, mismatch count, and first mismatch tuple:
  `rank, buffer, index/row/col, actual, expected`.
- For head dumps, compare at least the first `min(16, rows)` rows and `min(64, K)` columns, plus the last row of each non-empty
  segment when the segment metadata is available.
- Multi-rank commands must print one start line and one completion line per rank. Missing completion from any rank is a failure.
- Stage-gated runs must print rank-scoped stage markers before and after each blocking region, for example
  `rank=1 stage=wait_count begin` and `rank=1 stage=wait_count done`. A timeout is diagnosed from the last missing `done`
  marker, not by re-running with a longer timeout.
- Static forbidden-API grep is part of completion for every task that touches kernel code.
- After completing or blocking a task, update the task status table and add a `Feedback` note under that task.

## Design Review Result

The current `DESIGN.md` is implementable as the first version. It has the important boundaries fixed:

- exactly two device compute kernels: `DispatchCombineTileDispatch` and `DispatchCombineTileCombine`;
- real multi-process/multi-card acceptance path, no single-process multi-rank simulation;
- continuous device mapping only: `device = deviceBase + rank`;
- no SHMEM in host or device code;
- kernel side has no AscendC/Catlass/SHMEM dependency and uses PTO public headers only;
- HCCL window is the only cross-rank memory substrate;
- dispatch/combine are separate CPU-launched operators, so overlap requirements are intra-kernel only.

Two implementation cautions should stay visible during coding:

- `GlobalTensor`/`Tile` exact template signatures must be verified against local PTO headers before writing the final kernel
  helpers. Treat the aliases in `DESIGN.md` as intent, not as blindly copyable code.
- Several communication stages are hard to validate with black-box e2e alone. Each task below includes an observation path:
  compile, forbidden-API grep, deterministic debug dumps, per-rank logs, or focused code review.

## File Structure

Create these project files:

- `kernels/manual/a2a3/dispatch_combine_tile/CMakeLists.txt`  
  Build host binary and one vector-only kernel shared library.
- `kernels/manual/a2a3/dispatch_combine_tile/run.sh`  
  Parse explicit parameters, set CANN/MPI/HCCL env, build, and launch `mpirun -n ${pes}`.
- `kernels/manual/a2a3/dispatch_combine_tile/common.h`  
  Shared shape, layout, parameter, alignment, and small POD structs used by host and kernel.
- `kernels/manual/a2a3/dispatch_combine_tile/args.h`  
  Host command-line parsing and validation helpers.
- `kernels/manual/a2a3/dispatch_combine_tile/layout.h`  
  Host-side layout byte calculation matching the POD layout structs.
- `kernels/manual/a2a3/dispatch_combine_tile/golden.h`  
  Deterministic data generation and CPU reference dispatch/combine logic.
- `kernels/manual/a2a3/dispatch_combine_tile/hccl_context.h`  
  Copy/adapt `gemm_ar` `HcclDeviceContext` and host extraction helpers without SHMEM.
- `kernels/manual/a2a3/dispatch_combine_tile/comm_mpi.h`  
  Copy/adapt MPI wrapper from `gemm_ar`.
- `kernels/manual/a2a3/dispatch_combine_tile/kernel_launchers.h`  
  Kernel declarations and host launch wrappers.
- `kernels/manual/a2a3/dispatch_combine_tile/dispatch_combine_tile_kernel.cpp`  
  PTO-only device helpers plus the two kernels.
- `kernels/manual/a2a3/dispatch_combine_tile/main.cpp`  
  Runtime orchestration, memory allocation, HCCL setup, kernel launches, timing, debug dump, verification.

Do not create additional device compute kernel source files for this first version.

## Task 0: PTO Project Initialization

**Files:**

- Verify/Create directory: `kernels/manual/a2a3/dispatch_combine_tile/`
- Keep: `kernels/manual/a2a3/dispatch_combine_tile/DESIGN.md`
- Keep: `kernels/manual/a2a3/dispatch_combine_tile/IMPLEMENTATION_PLAN.md`
- Create placeholder: `kernels/manual/a2a3/dispatch_combine_tile/CMakeLists.txt`
- Create placeholder: `kernels/manual/a2a3/dispatch_combine_tile/run.sh`
- Create placeholder: `kernels/manual/a2a3/dispatch_combine_tile/common.h`
- Create placeholder: `kernels/manual/a2a3/dispatch_combine_tile/args.h`
- Create placeholder: `kernels/manual/a2a3/dispatch_combine_tile/layout.h`
- Create placeholder: `kernels/manual/a2a3/dispatch_combine_tile/golden.h`
- Create placeholder: `kernels/manual/a2a3/dispatch_combine_tile/hccl_context.h`
- Create placeholder: `kernels/manual/a2a3/dispatch_combine_tile/comm_mpi.h`
- Create placeholder: `kernels/manual/a2a3/dispatch_combine_tile/kernel_launchers.h`
- Create placeholder: `kernels/manual/a2a3/dispatch_combine_tile/dispatch_combine_tile_kernel.cpp`
- Create placeholder: `kernels/manual/a2a3/dispatch_combine_tile/main.cpp`

**Scope:**

- [x] Create the project directory under `kernels/manual/a2a3/`, matching the `gemm_ar` manual PTO project location.
- [x] Keep `DESIGN.md` as the design baseline and `IMPLEMENTATION_PLAN.md` as the task/status tracker.
- [x] Initialize the file layout from `DESIGN.md` section 14.1, not from the original SHMEM project layout:
  - root-level `CMakeLists.txt`, `run.sh`, `main.cpp`;
  - root-level host/kernel interface headers;
  - no `include/`, `utils/`, `scripts/`, `out/`, or `build/` directories at initialization time.
- [x] Record in `CMakeLists.txt` comments that Task 1 will follow `gemm_ar`: Bisheng, PTO include first, host executable,
  one vector-only kernel shared library, HCCL/ACL host link.
- [x] Record in `run.sh` comments that Task 1/2 will follow `gemm_ar`: CANN env search, MPI search, `mpirun -n ${PES}`,
  `HCCL_BUFFSIZE` auto sizing, explicit shape args only.
- [x] Record in `common.h` the shared ABI names that all later tasks must use:
  `DispatchCombineTileShape`, `WorkspaceLayout`, `PeerWindowLayout`, and `DispatchCombineTileRuntimeConfig`.
- [x] In `common.h`, predeclare the exact `DispatchCombineTileShape` fields from `DESIGN.md` section 11:
  `ep`, `m`, `k`, `topK`, `expertPerRank`, `expertNum`, `maxOutputSize`, `aivBlocks`, `tileCols`, `rowChunk`,
  and `metadataPad`.
- [x] In `common.h`, predeclare the exact layout field names from `DESIGN.md` section 12.0.1:
  `localTokenPerExpert`, `blockTokenPerExpert`, `blockPrefixPerExpert`, `cumsumPerExpert`, `dispatchOffset`,
  `prevSumBeforeRank`, `localSync`, `floatScratch`, `dispatchedA`, `ptrDLocal`, `peerTokenPerExpert`,
  `expandedRowIdx`, `packedA`, `ptrD`, `countReadySignal`, `combineDoneSignal`, and `totalBytes`.
- [x] Record in `layout.h` the exported layout entry points that Task 2 must implement:
  `ComputeWorkspaceLayout`, `ComputePeerWindowLayout`, and `EstimateHcclBuffSizeMb`.
- [x] Record in `args.h` the exported parser/validator entry points that Task 2 must implement:
  `ParseArgs`, `ValidateArgs`, and `PrintRunSummary`.
- [x] Record in `golden.h` the exported data/golden entry points that Task 4 must implement:
  `GenerateOrLoadInputs`, `ComputeCpuGolden`, and `CompareOutputs`.
- [x] Record in `hccl_context.h` that Task 3 must adapt `gemm_ar` `HcclDeviceContext`, MESH/RING `windowsIn[]`
  extraction, and fixed-offset `peerWindow` slicing.
- [x] Record in `comm_mpi.h` that Task 3 must adapt `gemm_ar` dlopen MPI wrapper.
- [x] Record in `kernel_launchers.h` the fixed launch wrapper names that later host code must call:
  `LaunchDispatchCombineTileDispatch` and `LaunchDispatchCombineTileCombine`.
- [x] In `kernel_launchers.h`, predeclare launch wrapper parameters in the same order as the kernel ABI:
  shape, rank, input/output pointers, `peerWindow`, `hcclCtx`, `workspace`, stream, and launch block count.
- [x] Record in `dispatch_combine_tile_kernel.cpp` the only two device kernel names:
  `DispatchCombineTileDispatch` and `DispatchCombineTileCombine`.
- [x] In `dispatch_combine_tile_kernel.cpp`, predeclare the two kernel ABI parameter lists from `DESIGN.md` section 11 so
  Task 1 can fill empty bodies without changing the host/device contract.
- [x] Record in `main.cpp` the staged host flow that later tasks must fill:
  `ParseArgs -> InitMpiAndRank -> BindDeviceContinuous -> InitHcclWindowContext -> ComputeLayouts ->
  AllocateLocalBuffers -> SlicePeerWindow -> GenerateOrLoadData -> RunDispatch -> PrepareExpertOutputIdentity ->
  RunCombine -> VerifyAndDump -> Cleanup`.
- [x] In host placeholders, record that host may use ACL/HCCL/MPI but no SHMEM.
- [x] In kernel placeholder, record that kernel may include only PTO/C++ headers and must not include AscendC/Catlass/SHMEM
  headers.
- [x] Preserve the original `dispatch_gmm_combine` semantics only as interface requirements: `-pes/-M/-K/-expertPerPe`,
  `data-dir/out` style files, per-rank output verification, warmup/iters/profile logs.
- [x] Do not copy code from `dispatch_gmm_combine_v2` or `dispatch_ffn_combine_v3`.
- [x] Do not copy original `dispatch_gmm_combine` SHMEM bootstrap, Catlass/GMM/SwiGLU/quant code, or Python data generation
  scripts.
- [x] Do not introduce build products, generated data, or a `TODO.md`.

**Observation/Validation:**

- [x] Run:

  ```bash
  find kernels/manual/a2a3/dispatch_combine_tile -maxdepth 1 -type f | sort
  ```

  Expected: the output contains exactly the planned md/source/script files and no build/data/generated files.

- [x] Run:

  ```bash
  rg -n "aclshmem|shmem_|symmetricPtr|dispatch_gmm_combine_v2|dispatch_ffn_combine_v3|kernel_operator|AscendC::|Catlass" \
    kernels/manual/a2a3/dispatch_combine_tile
  ```

  Expected: matches are allowed only in `DESIGN.md`/`IMPLEMENTATION_PLAN.md` as forbidden/reference text, not in source
  placeholders.

- [x] Run:

  ```bash
  rg -n "DispatchCombineTileShape|WorkspaceLayout|PeerWindowLayout|DispatchCombineTileRuntimeConfig|\
LaunchDispatchCombineTileDispatch|LaunchDispatchCombineTileCombine|DispatchCombineTileDispatch|DispatchCombineTileCombine|\
ComputeWorkspaceLayout|ComputePeerWindowLayout|EstimateHcclBuffSizeMb|ParseArgs|ValidateArgs|PrintRunSummary|\
GenerateOrLoadInputs|ComputeCpuGolden|CompareOutputs|InitMpiAndRank|BindDeviceContinuous|InitHcclWindowContext|\
PrepareExpertOutputIdentity" \
    kernels/manual/a2a3/dispatch_combine_tile/CMakeLists.txt \
    kernels/manual/a2a3/dispatch_combine_tile/run.sh \
    kernels/manual/a2a3/dispatch_combine_tile/common.h \
    kernels/manual/a2a3/dispatch_combine_tile/args.h \
    kernels/manual/a2a3/dispatch_combine_tile/layout.h \
    kernels/manual/a2a3/dispatch_combine_tile/golden.h \
    kernels/manual/a2a3/dispatch_combine_tile/hccl_context.h \
    kernels/manual/a2a3/dispatch_combine_tile/comm_mpi.h \
    kernels/manual/a2a3/dispatch_combine_tile/kernel_launchers.h \
    kernels/manual/a2a3/dispatch_combine_tile/dispatch_combine_tile_kernel.cpp \
    kernels/manual/a2a3/dispatch_combine_tile/main.cpp
  ```

  Expected: source placeholders mention all required ABI/stage names so Task 1-12 can fill implementation without renaming
  entry points.

- [x] Run:

  ```bash
  test ! -f kernels/manual/a2a3/dispatch_combine_tile/TODO.md
  ```

  Expected: command exits with status 0.

- [x] Update the status table: mark Task 0 `[x]` only after placeholders exist and the grep checks pass.

Feedback:
- 2026-05-25: Created Task0 initialization contract files at the project root, preserved DESIGN.md/IMPLEMENTATION_PLAN.md,
  and verified with `find kernels/manual/a2a3/dispatch_combine_tile -maxdepth 1 -type f | sort`,
  forbidden-reference grep, ABI/stage-name grep, and `test ! -f kernels/manual/a2a3/dispatch_combine_tile/TODO.md`.
  Remaining risk: Task0 intentionally does not validate CMake/build execution; that starts in Task1.

## Task 1: Buildable Project Scaffold

**Files:**

- Modify: `kernels/manual/a2a3/dispatch_combine_tile/CMakeLists.txt`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/run.sh`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/common.h`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/kernel_launchers.h`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/dispatch_combine_tile_kernel.cpp`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/main.cpp`

**Scope:**

- [x] Preserve standard copyright headers from Task 0.
- [x] Add CMake target `dispatch_combine_tile_kernel` compiled for `dav-c220-vec`.
- [x] Add host executable `dispatch_combine_tile`.
- [x] Define `DispatchCombineTileShape`, `WorkspaceLayout`, and `PeerWindowLayout` PODs in `common.h`.
- [x] Add two empty PTO kernel entry points with the exact names from `DESIGN.md`.
- [x] Add host launch wrappers that compile and call the two kernel symbols.
- [x] Add `main.cpp` skeleton that parses no real args yet and prints `dispatch_combine_tile scaffold`.
- [x] Add `run.sh` skeleton with explicit defaults and no `--case all`.

**Observation/Validation:**

- [x] Run:

  ```bash
  cd kernels/manual/a2a3/dispatch_combine_tile
  bash run.sh --skip-run 1
  ```

  Expected: CMake and `make` produce `build/dispatch_combine_tile`, `build/libdispatch_combine_tile_kernel.so`,
  and `run.sh` prints `skip_run=1`.

- [x] Run:

  ```bash
  rg -n "DispatchCombineTile[A-Za-z0-9_]*\\(" dispatch_combine_tile_kernel.cpp
  ```

  Expected: exactly the two public kernel entry points plus local helper calls, no extra `__global__ AICORE` compute kernels.

- [x] Run:

  ```bash
  rg -n "kernel_operator|AscendC::|LocalTensor|DataCopy|Catlass|aclshmem|shmem_|symmetricPtr" .
  ```

  Expected: no matches outside `DESIGN.md`/`IMPLEMENTATION_PLAN.md`.

Feedback:
- 2026-05-25: `bash run.sh --skip-run 1` passed after loading the CANN 8.5/ltr_pto environment from `CLAUDE.md`;
  produced `build/dispatch_combine_tile` and `build/libdispatch_combine_tile_kernel.so`. `./build/dispatch_combine_tile`
  prints `dispatch_combine_tile scaffold`. Static greps confirmed only the two public kernel entries plus launcher calls,
  and forbidden API matches are confined to `DESIGN.md`/`IMPLEMENTATION_PLAN.md`; source-only forbidden grep has no matches.
  Remaining risk: kernel bodies are intentionally empty until later tasks.

## Task 2: Explicit Parameters and Layout Calculation

**Files:**

- Create: `kernels/manual/a2a3/dispatch_combine_tile/args.h`
- Create: `kernels/manual/a2a3/dispatch_combine_tile/layout.h`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/main.cpp`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/run.sh`

**Scope:**

- [x] Implement command-line parsing for all parameters in `DESIGN.md` sections 5.1-5.4.
- [x] Support short aliases: `-pes`, `-M`, `-K`, `-topK`, `-expertPerPe`, `-debug`, `-iters`, `-warmup`,
  `-device-base`, `-aivBlocks`, `-tileCols`.
- [x] Add explicit observability gates used by this plan: `--skip-run`, `--skip-kernels`, `--host-golden-only`,
  `--dispatch-metadata-only`, `--dispatch-only`, and `--combine-return-only`. These gates must not select hidden shapes;
  they only stop after a named stage for compile/debug/code-review validation.
- [x] Confirm no case preset path is implemented; `--case`, `--case-all`, and `--case all` are not documented or required
  run modes.
- [x] Implement continuous mapping validation: `deviceBase + pes <= ndevices`.
- [x] Implement `AlignUp`, `expertNumPadded`, workspace bytes, peer window bytes, and `HCCL_BUFFSIZE` estimate.
- [x] Print a complete parameter summary from both `run.sh` and `main.cpp`.
- [x] Keep `maxOutputSize == 0` mapped to `EP * M * topK`.
- [x] Enforce first-version constraints: `run-mode == npu`, `K % tileCols == 0`, nonzero shape fields, no capacity/drop.

**Observation/Validation:**

- [x] Run:

  ```bash
  bash run.sh -pes 3 -M 8 -K 64 -topK 2 -expertPerPe 1 --device-base 1 --ndevices 8 --skip-run 1
  ```

  Expected logs include `PES=3`, `DEVICE_BASE=1`, `NDEVICES=8`, `M=8`, `K=64`, `TOPK=2`, and computed
  `peer_window_bytes`, `workspace_bytes`, `MAX_OUTPUT_SIZE=48`, and an auto `HCCL_BUFFSIZE` value.

- [x] Code-review checkpoint: confirm `run.sh` uses only explicit shape parameters, has no hidden shape list, and does not
  document `--case` usage.
- [x] Run invalid-shape checks:

  ```bash
  bash run.sh -pes 2 -M 8 -K 128 -topK 2 -expertPerPe 1 --tile-cols 64 --skip-run 1
  bash run.sh -pes 4 -M 8 -K 64 -topK 2 -expertPerPe 1 --device-base 6 --ndevices 8 --skip-run 1
  ```

  Expected: first command passes `K % tileCols == 0`; second command fails with `deviceBase + pes > ndevices`.

Feedback:
- 2026-05-25: Implemented explicit host/run.sh parsing and aligned layout byte estimates. Sequential validation passed:
  `bash run.sh -pes 3 -M 8 -K 64 -topK 2 -expertPerPe 1 --device-base 1 --ndevices 8 --skip-run 1`,
  `bash run.sh -pes 2 -M 8 -K 128 -topK 2 -expertPerPe 1 --tile-cols 64 --skip-run 1` passed,
  and `bash run.sh -pes 4 -M 8 -K 64 -topK 2 -expertPerPe 1 --device-base 6 --ndevices 8 --skip-run 1`
  failed with `deviceBase + pes > ndevices`. Code review confirmed no hidden case preset list is documented as a supported
  run mode. `run.sh --skip-build 1` also exercised `main.cpp` summary printing through the intended CANN environment.
  Remaining risk: `run.sh` resolves omitted `tileCols` to `min(1024, K)` so the plan's small `K=64` validation can pass
  while explicit `--tile-cols` remains strict.

## Task 3: MPI/HCCL Runtime and Continuous Card Mapping

**Files:**

- Create: `kernels/manual/a2a3/dispatch_combine_tile/comm_mpi.h`
- Create: `kernels/manual/a2a3/dispatch_combine_tile/hccl_context.h`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/main.cpp`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/run.sh`

**Scope:**

- [x] Copy/adapt `gemm_ar/comm_mpi.h` for MPI init, rank, size, barrier, broadcast, and finalize.
- [x] Copy/adapt `gemm_ar/hccl_context.h` `HcclDeviceContext`.
- [x] Implement host ACL init, `aclrtSetDevice(deviceBase + rank)`, stream creation, and teardown.
- [x] Implement HCCL root-info broadcast through MPI.
- [x] Allocate/extract HCCL window resources following `gemm_ar`; expose device-visible `HcclDeviceContext`.
- [x] Slice `peerWindow = windowsIn[rank] + peerWindowOffset` with identical offset on all ranks.
- [x] Verify `peerWindowOffset + peerWindowBytes <= hcclCtx.winSize`.
- [x] Add rank logs:

  ```text
  rank=... size=... device=... window_base=... peer_window=... win_size=...
  ```

**Observation/Validation:**

- [x] Run small multi-process scaffold:

  ```bash
  timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 1 --skip-kernels 1
  ```

  Expected: two MPI ranks start and finish; rank 0 binds device base; rank 1 binds `deviceBase + 1`; HCCL context logs print
  nonzero `window_base`, `peer_window`, and `win_size`; every rank prints `skip_kernels_done`.

- [x] For 3-card and 4-card mapping:

  ```bash
  timeout 60s bash run.sh -pes 3 -M 8 -K 64 -topK 2 -expertPerPe 1 --device-base 0 --ndevices 8 --debug 1 --skip-kernels 1
  timeout 60s bash run.sh -pes 4 -M 8 -K 64 -topK 2 -expertPerPe 1 --device-base 0 --ndevices 8 --debug 1 --skip-kernels 1
  ```

  Expected: devices are exactly `0,1,2` and `0,1,2,3`; no rank reports a non-contiguous device id.

- [x] Run forbidden API grep:

  ```bash
  rg -n "aclshmem|shmem_|symmetricPtr" .
  ```

  Expected: no matches outside docs/plans.

Feedback:
- 2026-05-25: Implemented header-only MPI dlopen wrapper, continuous rank-to-device binding, ACL stream lifecycle,
  HCCL root-info MPI broadcast, `gemm_ar`-style MESH/RING window extraction, device-visible `HcclDeviceContext`,
  and fixed-offset `peerWindow` slicing. Verified sequentially with
  `timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 1 --skip-kernels 1`,
  `timeout 60s bash run.sh -pes 3 -M 8 -K 64 -topK 2 -expertPerPe 1 --device-base 0 --ndevices 8 --debug 1 --skip-kernels 1`,
  and
  `timeout 60s bash run.sh -pes 4 -M 8 -K 64 -topK 2 -expertPerPe 1 --device-base 0 --ndevices 8 --debug 1 --skip-kernels 1`.
  All ranks printed contiguous `device=0..N-1`, nonzero `window_base`/`peer_window`, `win_size=68157440`, and
  `skip_kernels_done`. `rg -n "aclshmem|shmem_|symmetricPtr" .` matched only `DESIGN.md`/`IMPLEMENTATION_PLAN.md`.
  Remaining risk: this task intentionally validates runtime bootstrap only; actual dispatch/combine kernels are still gated
  behind later tasks.

## Task 4: Deterministic Data and CPU Golden

**Files:**

- Create: `kernels/manual/a2a3/dispatch_combine_tile/golden.h`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/main.cpp`

**Scope:**

- [x] Generate deterministic per-rank `inputA`, `expertIdx`, and `probs` from `seed + rank`.
- [x] Generate expert ids across all `EP * expertPerRank` experts so every rank receives at least some rows in small tests.
- [x] Support `--gen-data 0` loading the documented `data-dir` files.
- [x] Implement CPU reference:
  - route local rank tokens by global expert;
  - build `peerTokenPerExpert[src, expert]`;
  - build packed row ids;
  - build owner-rank `dispatchedA`;
  - use identity `expertOutput = dispatchedA`;
  - return rows to owner `ptrD`;
  - restore `outputC[token, col] = sum_slot probs[token, slot] * ptrD[row, col]`.
- [x] Write documented binary files when `--debug > 0`.
- [x] Add host-only `--host-golden-only 1` path for validating data/golden without launching kernels.

**Observation/Validation:**

- [x] Run:

  ```bash
  timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --host-golden-only 1
  ```

  Expected: rank files are generated under `data-dir`; logs show nonzero counts for both owner ranks; CPU golden prints
  `golden_total_routes=32`, `golden_invalid_routes=0`, and one `golden_rank_done` line per rank.

- [x] Code-review checkpoint: CPU golden must use the same row-layout formulas as `DESIGN.md` sections 9 and 10.

Feedback:
- 2026-05-25: Implemented deterministic host input generation/loading, CPU dispatch/combine golden, identity
  `expertOutput`, debug binary dumps, and an MPI-only `--host-golden-only 1` path that exits before ACL/HCCL setup.
  Verified with
  `timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --host-golden-only 1 --skip-build 1`:
  both ranks printed `golden_total_routes=32`, `golden_invalid_routes=0`, `compare_mismatches=0`, and
  `golden_rank_done`. `find out -maxdepth 1 -type f | sort` showed rank input, golden output, and debug metadata files.
  Also verified `--gen-data 0` reads the documented files with the same command plus `--gen-data 0`.

## Task 5: PTO Kernel Views, Remote Pointer Helper, and Signal Helpers

**Files:**

- Modify: `kernels/manual/a2a3/dispatch_combine_tile/dispatch_combine_tile_kernel.cpp`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/common.h`

**Scope:**

- [x] Add kernel-side view structs for workspace and peer window fields.
- [x] Add `RemotePtr(ctx, localPeerWindowBase, peerRank)` that only computes `windowsIn[peer] + offset`.
- [x] Add `GlobalTensor` construction helpers for 1D/2D half, int32, and float views.
- [x] Add local row copy helper using `TLOAD/TSTORE` and ping-pong Vec tiles.
- [x] Add remote row helpers using `pto::comm::TGET` and `pto::comm::TPUT` ping-pong overloads.
- [x] Add `TNOTIFY/TWAIT` signal helpers for count and combine-done signals.
- [x] Keep metadata scalar reads/writes inside helper functions only.

**Observation/Validation:**

- [x] Build:

  ```bash
  bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --skip-run 1
  ```

  Expected: kernel compiles with PTO headers.

- [x] Run:

  ```bash
  rg -n "#include|AscendC::|kernel_operator|LocalTensor|GlobalTensor<|TQue|TBuf|TPipe|DataCopy|Catlass|aclshmem|shmem_" dispatch_combine_tile_kernel.cpp
  ```

  Expected: includes are PTO/C++ only; `GlobalTensor<` references are PTO namespace or imported PTO aliases.

- [x] Code-review checkpoint: `RemotePtr` has no communication, sync, runtime query, or non-window pointer handling.

Feedback:
- 2026-05-25: Added shared `HcclDeviceContext` ABI, kernel-side workspace/window view builders, dynamic
  `GlobalTensor` helpers, local row copy, `TGET`/`TPUT` row wrappers, `TNOTIFY`/`TWAIT` signal wrappers, and scalar
  metadata helpers. Verified with
  `bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --skip-run 1`, which built the PTO kernel and host binary.
  Static grep
  `rg -n "#include|AscendC::|kernel_operator|LocalTensor|GlobalTensor<|TQue|TBuf|TPipe|DataCopy|Catlass|aclshmem|shmem_" dispatch_combine_tile_kernel.cpp`
  showed only PTO/C++/local includes plus PTO `GlobalTensor`; no forbidden APIs. Code review confirmed `RemotePtr` only
  computes `windowsIn[peerRank] + (localPtr - windowsIn[rankId])` and performs no communication or runtime query.

## Task 6: Dispatch Metadata and Count Publication

**Files:**

- Modify: `kernels/manual/a2a3/dispatch_combine_tile/dispatch_combine_tile_kernel.cpp`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/main.cpp`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/golden.h`

**Scope:**

- [x] Implement dispatch state clearing for metadata and signals.
- [x] Implement contiguous token-shard local count into `blockTokenPerExpert[block, expert]`.
- [x] Implement block-prefix scan and `localTokenPerExpert`.
- [x] Implement packed expert offsets from `localTokenPerExpert`.
- [x] Publish count rows to every peer with `TPUT(remote.peerTokenPerExpert[myRank, :])`.
- [x] Notify `remote.countReadySignal[myRank]` only after count row publication.
- [x] Wait on local `countReadySignal[src]` with `TWAIT`.
- [x] Build `cumsumPerExpert`, `dispatchOffset`, and `prevSumBeforeRank`.
- [x] Add debug dump and rank logs:

  ```text
  rank=... dispatch_counts local_total=... owner_rows=...
  rank=... count_wait_done peers=...
  ```

**Observation/Validation:**

- [x] Run:

  ```bash
  timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 1 --dispatch-metadata-only 1
  ```

  Expected: no hang; every rank prints `dispatch_metadata_done`; `peerTokenPerExpert`, `cumsumPerExpert`,
  `dispatchOffset`, and `prevSumBeforeRank` match CPU golden with zero mismatches.

- [x] If hardware execution is unavailable, build plus code-review is acceptable for this task:
  - `TPUT` is used for count row exchange;
  - `TNOTIFY` happens after count row `TPUT`;
  - `TWAIT` is device-side, not replaced by host barrier.
  - count wait loops cover all `src in [0, EP)`, including `src == myRank`.

Feedback:
- 2026-05-26: Implemented dispatch metadata/count exchange in the PTO dispatch kernel and host
  `--dispatch-metadata-only` verification. Verified with
  `timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 1 --dispatch-metadata-only 1 --skip-build 1`:
  both ranks printed `dispatch_metadata_done`; `localTokenPerExpert`, `peerTokenPerExpert`, `cumsumPerExpert`,
  `dispatchOffset`, and `prevSumBeforeRank` all reported zero mismatches. Static forbidden API grep over source files
  had no matches. Debugging note: the kernel must not clear remote-visible `peerTokenPerExpert/countReadySignal` while peer
  ranks may publish into them; host clears `peerWindow` before launch, and kernel clears only local workspace metadata plus
  local owner-only fields. `SoftSyncAiv` uses an explicit UB address for its sync tile.

## Task 7: Dispatch Pack and Payload Gather

**Files:**

- Modify: `kernels/manual/a2a3/dispatch_combine_tile/dispatch_combine_tile_kernel.cpp`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/main.cpp`

**Scope:**

- [x] Implement stable local pack:
  - contiguous token shard;
  - per-block cursor initialized from `blockPrefixPerExpert`;
  - write `expandedRowIdx[token * topK + slot]`;
  - copy `inputA[token, :]` into `peerWindow.packedA[packedRow, :]` through PTO row copy helper.
- [x] Implement payload gather for `ownerRank == myRank` local experts only.
- [x] Use peer rank round-robin assignment: `src = blockId + n * blockNum`.
- [x] Use `TGET(remote(src).packedA segment -> workspace.dispatchedA segment)` with ping-pong staging.
- [x] Skip zero-row segments.
- [x] Add debug logs:

  ```text
  rank=... dispatch_gather local_expert=... src=... rows=... dst_start=...
  ```

**Observation/Validation:**

- [x] Run:

  ```bash
  timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --dispatch-only 1
  ```

  Expected: every rank prints `dispatch_pack_done` and `dispatch_gather_done`; `expandedRowIdx`, `packedA_head`,
  and `dispatchedA_head` match CPU golden with zero mismatches for the dumped rows.

- [x] Run:

  ```bash
  rg -n "TGET|TLOAD|TSTORE|expandedRowIdx|blockPrefixPerExpert" dispatch_combine_tile_kernel.cpp
  ```

  Expected: pack uses `TLOAD/TSTORE`, gather uses `TGET`, and row order is based on `blockPrefixPerExpert`.

Feedback:
- 2026-05-26: Implemented stable local payload pack and owner-rank dispatch gather. Verified with
  `timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --dispatch-only 1`:
  both ranks printed `dispatch_pack_done` and `dispatch_gather_done`; `expandedRowIdx`, `packedA`, and `dispatchedA`
  all reported zero mismatches. Static grep confirmed pack uses `TLOAD/TSTORE`, gather uses `TGET`, and row ordering uses
  `blockPrefixPerExpert`. Debugging note: the local PTO row copy must synchronize MTE2 -> MTE3 between `TLOAD` and
  `TSTORE`; otherwise `packedA` is corrupted even when routing metadata is correct.

## Task 8: Host Expert Output Preparation and Dispatch Timing

**Files:**

- Modify: `kernels/manual/a2a3/dispatch_combine_tile/main.cpp`

**Scope:**

- [x] After dispatch stream sync, copy `workspace.dispatchedA` to `expertOutput`.
- [x] Keep this as host verification helper, not a third device kernel.
- [x] Time and print `dispatch_e2e_us` and `prepare_host_us`.
- [x] Add option to compare `workspace.dispatchedA` against CPU golden before combine when `--debug >= 2`.
- [x] Ensure host barrier is used only for process-level iteration alignment and window reuse protection.

**Observation/Validation:**

- [x] Run:

  ```bash
  timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --dispatch-only 1
  ```

  Expected logs include positive `dispatch_e2e_us` and `prepare_host_us`; debug compare reports `expertOutput` equals
  `dispatchedA` for the copied row range.

- [x] Code-review checkpoint: no `PrepareExpertOutput` device kernel exists.

Feedback:
- 2026-05-26: Implemented host-side identity expert output preparation by copying `workspace.dispatchedA` to
  `expertOutput`, with `dispatch_e2e_us` and `prepare_host_us` logs. Verified with
  `timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --dispatch-only 1`: both ranks printed
  positive timing values, `expertOutput_vs_dispatchedA` zero mismatches, and `expertOutput` vs CPU golden zero
  mismatches. Code review/static grep confirmed there is no `PrepareExpertOutput` device kernel; this remains a host
  verification helper.

## Task 9: Combine Return Path

**Files:**

- Modify: `kernels/manual/a2a3/dispatch_combine_tile/dispatch_combine_tile_kernel.cpp`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/main.cpp`

**Scope:**

- [x] Implement combine state clearing for `combineDoneSignal` and relevant output rows.
- [x] Implement peer rank round-robin return loop: `src = blockId + n * blockNum`.
- [x] For each source peer, iterate local experts in order and compute:
  - `rows = peerTokenPerExpert[src, globalExpert]`;
  - `srcStart = dispatchOffset[localExpert] + prevSumBeforeRank[src, localExpert]`;
  - `dstStart = cumsumPerExpert[src, globalExpert - 1]` or zero.
- [x] Use `TPUT(remote(src).ptrD segment <- expertOutput segment)` with ping-pong staging.
- [x] Notify `remote(src).combineDoneSignal[myRank]` once after all local expert segments for that peer are written.
- [x] Wait on local `combineDoneSignal[peer]` with `TWAIT`.
- [x] Add debug logs:

  ```text
  rank=... combine_return dst=... local_expert=... rows=... src_start=... dst_start=...
  rank=... combine_wait_done peers=...
  ```

**Observation/Validation:**

- [x] Run:

  ```bash
  timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --combine-return-only 1
  ```

  Expected: no hang; every rank prints `combine_return_done` and `combine_wait_done`; dumped `ptrD_head` matches CPU
  golden with zero mismatches for returned rows.

- [x] If black-box execution hangs, use logs to identify the last peer/rank signal and inspect the `TNOTIFY/TWAIT` pairing before
  changing data movement.

Feedback:
- 2026-05-26: Implemented combine return path in the existing `DispatchCombineTileCombine` kernel with peer-sharded
  `TPUT` into owner `ptrD`, one `combineDoneSignal[myRank]` notify per destination peer, and device-side `TWAIT` for peer
  completion. Verified with
  `timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --combine-return-only 1`: no hang; both
  ranks printed `combine_return_done`, `combine_wait_done`, `combineDoneSignal=2,2`, and `ptrD` zero mismatches.
  State clearing for `ptrD`, `combineDoneSignal`, and `outputC` is done on host before launch to avoid device-side
  remote-visible signal races.

## Task 10: Combine Restore and Output Verification

**Files:**

- Modify: `kernels/manual/a2a3/dispatch_combine_tile/dispatch_combine_tile_kernel.cpp`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/main.cpp`

**Scope:**

- [x] Restore output by contiguous token shard; each block owns disjoint token rows.
- [x] For each token row, clear `outputC[token, :]`.
- [x] For each `topK` slot, read `expandedRowIdx`; skip `-1`.
- [x] Implement `outputC += probs[token, slot] * peerWindow.ptrD[row, :]` with PTO Vec tile operations.
- [x] Use `floatScratch` if direct half-by-float accumulation is not supported cleanly by available PTO instructions.
- [x] Copy `outputC` to host and compare against CPU golden with `rtol/atol`.
- [x] Print mismatch count and first mismatch details when verification fails.

**Observation/Validation:**

- [x] Run:

  ```bash
  timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2
  ```

  Expected: every rank prints `restore_done`; `outputC` compare prints `verify=PASS`, `mismatch_count=0`.

- [x] Run:

  ```bash
  timeout 60s bash run.sh -pes 2 -M 128 -K 256 -topK 2 -expertPerPe 2 --debug 1
  ```

  Expected: `verify=PASS`, positive timing fields, and no debug-level-2 payload dump unless `debug=2`.

Feedback:
- 2026-05-26: Implemented combine restore in `DispatchCombineTileCombine`: return rows are written to `ptrD`, peers are
  waited with device-side `TWAIT`, then each AIV block restores a disjoint contiguous token shard with PTO Vec
  `TEXPANDS/TLOAD/TAXPY/TSTORE`. Verified with
  `timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --clean-build 0 --keep-hccl-shm 1`
  and
  `timeout 60s bash run.sh -pes 2 -M 128 -K 256 -topK 2 -expertPerPe 2 --debug 1 --clean-build 0 --keep-hccl-shm 1`;
  both ranks printed `restore_done`, `verify=PASS`, and `mismatch_count=0`. During validation, the medium shape initially
  timed out in restore; root cause was a hand-written flag sequence around two `TLOAD`s and `TAXPY`. Switching restore to
  PTO `Event<Op::TLOAD, Op::TAXPY>` and `Event<Op::TAXPY, Op::TSTORE_VEC>` dependency chaining fixed the hang.

## Task 11: E2E Timing, Debug Dumps, and Run Matrix

**Files:**

- Modify: `kernels/manual/a2a3/dispatch_combine_tile/main.cpp`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/run.sh`
- Modify: `kernels/manual/a2a3/dispatch_combine_tile/README.md` if a README is added during implementation

**Scope:**

- [x] Implement warmup/iters loop.
- [x] Print required timing fields:
  - `dispatch_e2e_us`;
  - `prepare_host_us`;
  - `combine_e2e_us`;
  - `total_e2e_us`.
- [x] Print required performance config fields:
  - `aiv_blocks`;
  - `tile_cols`;
  - `peer_window_bytes`;
  - `workspace_bytes`;
  - `dispatch_peer_shards`;
  - `combine_peer_shards`.
- [x] Implement debug levels 0, 1, and 2 exactly as described in `DESIGN.md`.
- [x] Ensure `run.sh` can run 2, 3, and 4 ranks through continuous mapping.
- [x] Ensure `--hccl-buffsize-mb 0` auto-raises `HCCL_BUFFSIZE` based on peer window bytes.

**Observation/Validation:**

- [x] Run acceptance matrix:

  ```bash
  timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 -debug 2
  timeout 60s bash run.sh -pes 2 -M 128 -K 256 -topK 2 -expertPerPe 2 -debug 1
  timeout 60s bash run.sh -pes 2 -M 128 -K 256 -topK 4 -expertPerPe 2 -debug 1
  timeout 90s bash run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 -debug 0
  ```

  Expected: all pass verification; all print positive timing fields; default-shape run prints `debug=0` logs only and no
  head payload dump.

- [x] Run multi-card mapping if hardware is available:

  ```bash
  timeout 60s bash run.sh -pes 3 -M 8 -K 64 -topK 2 -expertPerPe 1 --device-base 0 --ndevices 8 -debug 1
  timeout 60s bash run.sh -pes 4 -M 8 -K 64 -topK 2 -expertPerPe 1 --device-base 0 --ndevices 8 -debug 1
  ```

  Expected: all ranks bind continuous devices, all ranks print completion lines, and verification passes.

Feedback:
- 2026-05-26: Implemented warmup/iters measurement loop, rank-scoped timing, performance config logging, debug gating, and
  auto `HCCL_BUFFSIZE` sizing. The 2-rank acceptance matrix passed:
  `timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 -debug 2 --clean-build 0 --keep-hccl-shm 1`,
  `timeout 60s bash run.sh -pes 2 -M 128 -K 256 -topK 2 -expertPerPe 2 -debug 1 --clean-build 0 --keep-hccl-shm 1`,
  `timeout 60s bash run.sh -pes 2 -M 128 -K 256 -topK 4 -expertPerPe 2 -debug 1 --skip-build 1 --clean-build 0 --keep-hccl-shm 1`,
  and
  `timeout 90s bash run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 -debug 0 --skip-build 1 --clean-build 0 --keep-hccl-shm 1`.
  All printed positive `dispatch_e2e_us`, `prepare_host_us`, `combine_e2e_us`, `total_e2e_us`, and
  `verify=PASS mismatch_count=0`; default debug=0 produced no payload head dump and printed `HCCL_BUFFSIZE=79`.
  Multi-rank E2E initially exposed a stale-signal/window-reuse issue because all iterations reused fixed device signal
  values. The fix adds `shape.signalValue = iter + 1` at host launch and uses that value for both dispatch count
  `TNOTIFY/TWAIT` and combine-done `TNOTIFY/TWAIT`; second-iteration debug logs now show `count_ready_signal=2,...`
  instead of reusing `1`. After the fix,
  `timeout 60s bash run.sh -pes 3 -M 8 -K 64 -topK 2 -expertPerPe 1 --device-base 0 --ndevices 8 -debug 1 --clean-build 0 --keep-hccl-shm 1`
  passed, and
  `timeout 60s bash run.sh -pes 4 -M 8 -K 64 -topK 2 -expertPerPe 1 --device-base 0 --ndevices 8 -debug 1 --skip-build 1 --clean-build 0 --keep-hccl-shm 1`
  passed; all ranks bound continuous devices, completed warmup and measure iterations, and printed
  `verify=PASS mismatch_count=0`. Focused 4-rank diagnostics also passed for `--dispatch-metadata-only`,
  `--dispatch-only`, and `--combine-return-only`.

## Task 12: Hardening and Final Review

**Files:**

- Modify as needed under `kernels/manual/a2a3/dispatch_combine_tile/`

**Scope:**

- [x] Run `clang-format -i -style=file` on C++ headers and sources.
- [x] Run `shellcheck run.sh` if available; otherwise manually review `set -euo pipefail`, quoting, and getopt handling.
- [x] Confirm only two device compute kernel entry points exist.
- [x] Confirm no SHMEM/AscendC/Catlass device dependency appears in kernel code.
- [x] Confirm all expected command-line parameters are visible and documented through `run.sh --help`.
- [x] Confirm host barriers do not replace device-side `TWAIT` readiness.
- [x] Confirm debug logs are guarded by `debug` level and not too noisy at `debug=0`.
- [x] Record any hardware-only limitations or unverified paths in final notes.

**Observation/Validation:**

- [x] Run final static checks:

  ```bash
  rg -n "kernel_operator|AscendC::|LocalTensor|DataCopy|Catlass|aclshmem|shmem_|symmetricPtr" .
  rg -n "__global__ AICORE void" dispatch_combine_tile_kernel.cpp
  rg -n "case all|case-all|--case" run.sh main.cpp args.h
  ```

  Expected: forbidden API matches only in docs/plans; exactly two kernel entry points; `--case` only appears in rejection/help
  logic.

- [x] Run task-state check:

  ```bash
  awk '/Current task status:/,/## Validation Rules/' IMPLEMENTATION_PLAN.md | rg -n "\\[ \\]|\\[~\\]|\\[!\\]"
  ```

  Expected: no output from the status table. Unchecked substeps may remain only if they document optional hardware paths not run
  in the current environment and are called out in final notes.

- [x] Run final default:

  ```bash
  timeout 90s bash run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 -debug 0 -iters 5 -warmup 3
  ```

  Expected: build succeeds, multi-process run succeeds, verification passes, and timing fields print.

Feedback:
- 2026-05-26: Ran `clang-format -i -style=file` on modified C++ files and final static checks:
  source-only forbidden API grep had no matches, `rg -n "__global__ AICORE void" dispatch_combine_tile_kernel.cpp`
  reported exactly `DispatchCombineTileDispatch` and `DispatchCombineTileCombine`,
  `rg -n "case all|case-all|--case" run.sh main.cpp args.h` showed only rejection/help logic,
  `bash -n run.sh` passed, `bash run.sh --help` documented the explicit parameters, and
  `git diff --check -- kernels/manual/a2a3/dispatch_combine_tile` passed. `shellcheck` is not installed in this
  environment, so `run.sh` was reviewed manually for `set -euo pipefail`, quoting, and option handling. Final default
  `timeout 90s bash run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 -debug 0 -iters 5 -warmup 3 --clean-build 0 --keep-hccl-shm 1`
  passed within timeout after rebuilding unchanged targets; both ranks completed three warmup iterations and five measured
  iterations with `verify=PASS mismatch_count=0` and timing fields. After the per-iteration signal epoch fix and 3/4-rank
  reruns, task-state check reports no unfinished or blocked status-table entries.

## Execution Notes

- Prefer one task per implementation session. Tasks 6, 7, 9, and 10 are the riskiest and should each end with a code-review
  checkpoint even if hardware execution is available.
- When a black-box run hangs, do not immediately rewrite the algorithm. First use the rank logs to identify whether the issue is
  count publication, count wait, combine return, or combine wait.
- Keep task commits focused. Do not refactor unrelated `gemm_ar` code.
- Do not add a third device kernel to make validation easier. Use host debug paths, D2H dumps, and CPU golden instead.
