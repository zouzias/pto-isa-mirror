# A5 Moe Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an A5 `moe_dispatch` project with configurable AIV block count and guarded HCCL remote-window layout, while fixing the A2/A3 `moe_combine` default AIV mismatch.

**Architecture:** Keep `moe_dispatch` as a dispatch-only PTO project under `kernels/manual/a5/moe_dispatch`. Host code owns args, layout validation, CPU golden, and runtime context setup; device code exposes a minimal AIV-only kernel that uses PTO Tile/GlobalTensor payload operations and A5 `PtoRemoteWindow` signal helpers.

**Tech Stack:** C++17, Ascend CCE, PTO headers, ACL/HCCL host APIs, MPI-style launcher shell.

---

### Task 1: Fix A2/A3 AIV Default

**Files:**
- Modify: `kernels/manual/a2a3/moe_combine/args.h`

- [ ] **Step 1: Write a parser regression check**

Run this temporary compile check after editing `args.h`:

```bash
tmp_cpp=/tmp/moe_combine_args_check.cpp
printf '%s\n' \
'#include "kernels/manual/a2a3/moe_combine/args.h"' \
'#include <cassert>' \
'int main() {' \
'  char prog[] = "check";' \
'  char *argv0[] = {prog};' \
'  auto defaults = moe_combine::ParseArgs(1, argv0);' \
'  assert(defaults.shape.aivBlocks == 24);' \
'  char opt[] = "--aiv-blocks";' \
'  char val[] = "8";' \
'  char *argv1[] = {prog, opt, val};' \
'  auto explicit_args = moe_combine::ParseArgs(3, argv1);' \
'  assert(explicit_args.shape.aivBlocks == 8);' \
'}' > "${tmp_cpp}"
g++ -std=c++17 -I. "${tmp_cpp}" -o /tmp/moe_combine_args_check
/tmp/moe_combine_args_check
```

Expected after implementation: command exits `0`.

- [ ] **Step 2: Change `ChooseDefaultAivBlocks()`**

Replace the return value in `kernels/manual/a2a3/moe_combine/args.h`:

```cpp
inline uint32_t ChooseDefaultAivBlocks(const MoeCombineShape &)
{
    return 24;
}
```

- [ ] **Step 3: Verify explicit override still wins**

Run the temporary compile check from Step 1. Expected: exit `0`.

### Task 2: Build A5 Dispatch Shared Host ABI

**Files:**
- Create: `kernels/manual/a5/moe_dispatch/common.h`
- Create: `kernels/manual/a5/moe_dispatch/layout.h`
- Create: `kernels/manual/a5/moe_dispatch/args.h`
- Create: `kernels/manual/a5/moe_dispatch/golden.h`

- [ ] **Step 1: Add `common.h`**

Define `MoeDispatchShape`, runtime config, resource config, window layout mode, workspace layout, and guarded window layout structs. Include `kMoeDispatchWindowHeadGuardBytes = 4096`, `kMoeDispatchTailControlBytes = 2 MiB`, and `kMoeDispatchSignalBytes = 1 MiB`.

- [ ] **Step 2: Add `layout.h`**

Implement checked `AlignUp`, `CheckedMul`, `ComputeWorkspaceLayout()`, `ComputeGuardedWindowLayout()`, `ValidateWindowLayout()`, and `EstimateHcclBuffSizeMb()`. The guarded layout must put `packedA` after the 4 KiB head guard and put `tokenPerExpert` at `windowBytes - 2 MiB`, `signal` at `windowBytes - 1 MiB`.

- [ ] **Step 3: Add `args.h`**

Implement `DefaultArgs()`, `ParseArgs()`, `ValidateArgs()`, `ChooseDefaultAivBlocks(socVersion, shape)`, and `DumpArgs()`. `--aiv-blocks` must override the SoC default. A5 default must be 40 for `Ascend950*`.

- [ ] **Step 4: Add `golden.h`**

Implement deterministic host data generation and CPU dispatch golden for `inputA[M,K]`, `expertIdx[M,topK]`, `tokenPerExpert`, `expandedRowIdx`, and `packedA`.

- [ ] **Step 5: Compile header-only checks**

Run small `g++ -std=c++17 -I.` checks for `args.h`, `layout.h`, and `golden.h`. Expected: all exit `0`.

### Task 3: Add A5 Runtime And Kernel Scaffolding

**Files:**
- Create: `kernels/manual/a5/moe_dispatch/op_host/runtime_context.hpp`
- Create: `kernels/manual/a5/moe_dispatch/op_host/runtime_context.cpp`
- Create: `kernels/manual/a5/moe_dispatch/op_kernel/utils/const_args.hpp`
- Create: `kernels/manual/a5/moe_dispatch/op_kernel/utils/hccl_context.hpp`
- Create: `kernels/manual/a5/moe_dispatch/op_kernel/utils/hccl_window.hpp`
- Create: `kernels/manual/a5/moe_dispatch/op_kernel/moe_dispatch_kernel.cpp`
- Create: `kernels/manual/a5/moe_dispatch/kernel_launchers.h`

- [ ] **Step 1: Add runtime context wrappers**

Port the minimal `PtoRemoteWindowContext`, stream/runtime structs, and loader helpers from A5 `dispatch_combine_moe`, preserving `windowIn/windowOut/windowBytes` validation.

- [ ] **Step 2: Add device constants and window helpers**

Create A5 constants and `PtoRemoteWindow` helper with signal region at `segmentBytes - 1 MiB`, 64B signal stride, `NotifyRemoteTokenReady()`, and `WaitTokenReady()`.

- [ ] **Step 3: Add minimal AIV kernel**

Expose `LaunchMoeDispatchKernel(...)` and an AIV-only kernel entry. The first implementation may use scalar metadata and simple payload copy loops, but the payload path must include PTO `Tile`, `GlobalTensor`, `TLOAD`, `TSTORE`, and a visible comm signal path.

### Task 4: Add Host Runner, CMake, Run Script, README

**Files:**
- Create: `kernels/manual/a5/moe_dispatch/main.cpp`
- Create: `kernels/manual/a5/moe_dispatch/CMakeLists.txt`
- Create: `kernels/manual/a5/moe_dispatch/run.sh`
- Create: `kernels/manual/a5/moe_dispatch/README.md`

- [ ] **Step 1: Add host runner**

Implement parse/validate/dump, layout calculation, golden-only mode, and build/run orchestration placeholders that do not claim A5 runtime success on A3 machines.

- [ ] **Step 2: Add CMake**

Follow A5 `dispatch_combine_moe` include/link structure. Define `PTO_NPU_ARCH_A5`, build `moe_dispatch_kernel` and `moe_dispatch`.

- [ ] **Step 3: Add run script**

Load `/home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh` when present, accept `--aiv-blocks`, compute/default `HCCL_BUFFSIZE`, and forward args to host binary.

- [ ] **Step 4: Add README**

Document scope, guarded HCCL window layout, AIV parameter chain, and compile-only limitation on current A3 machine.

### Task 5: Verification

**Files:**
- Check: `kernels/manual/a2a3/moe_combine/args.h`
- Check: `kernels/manual/a5/moe_dispatch/**`

- [ ] **Step 1: Run A2/A3 args regression**

Run the temporary `g++` parser check from Task 1. Expected: exit `0`.

- [ ] **Step 2: Run header-only A5 checks**

Compile temporary host checks for `moe_dispatch/args.h`, `layout.h`, and `golden.h`. Expected: exit `0`.

- [ ] **Step 3: Run static source checks**

Run:

```bash
rg -n "WINDOW_HEAD_GUARD|GuardedDispatchOnly|--aiv-blocks|TLOAD|TSTORE|TNOTIFY|TWAIT" kernels/manual/a5/moe_dispatch
```

Expected: all key terms appear in the new project.

- [ ] **Step 4: Run A5 compile if environment exists**

Run:

```bash
source /home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh
cmake -S kernels/manual/a5/moe_dispatch -B /tmp/moe_dispatch_a5_build -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_958b
cmake --build /tmp/moe_dispatch_a5_build --target moe_dispatch -j8
```

Expected: compile succeeds on a valid A5/CANN 9 environment. If unavailable locally, report that only static/header checks were run.
