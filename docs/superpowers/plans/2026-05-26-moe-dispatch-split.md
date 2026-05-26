# Moe Dispatch Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the existing dispatch half of `dispacth_tile` into a standalone `kernels/manual/a2a3/moe_dispatch` operator and get it building/running before PTO style cleanup.

**Architecture:** Start from the current runnable `dispacth_tile` code, preserve the dispatch data path, and remove combine-only kernel/host gates. Keep routing metadata, packed payload, count publication, prefix metadata, and TGET owner gather unchanged for this pass.

**Tech Stack:** C++17 host, Bisheng CCE A2/A3 device kernel, HCCL window communication, MPI launcher, existing PTO headers.

---

### Task 1: Acceptance Checks

**Files:**
- Create: `kernels/manual/a2a3/moe_dispatch/`
- Modify: `kernels/manual/a2a3/moe_dispatch/CMakeLists.txt`
- Modify: `kernels/manual/a2a3/moe_dispatch/run.sh`
- Modify: `kernels/manual/a2a3/moe_dispatch/common.h`
- Modify: `kernels/manual/a2a3/moe_dispatch/args.h`
- Modify: `kernels/manual/a2a3/moe_dispatch/kernel_launchers.h`
- Modify: `kernels/manual/a2a3/moe_dispatch/moe_dispatch_kernel.cpp`
- Modify: `kernels/manual/a2a3/moe_dispatch/main.cpp`

- [ ] **Step 1: Verify the new operator does not exist yet**

Run: `test -d kernels/manual/a2a3/moe_dispatch`
Expected: FAIL before implementation.

- [ ] **Step 2: Scaffold from the current dispatch project**

Copy source files from `kernels/manual/a2a3/dispacth_tile`, excluding `out/` and `build/`.

- [ ] **Step 3: Rename project-facing symbols**

Use `moe_dispatch`, `MoeDispatchShape`, `MoeDispatchArgs`, `MoeDispatchRuntimeConfig`, `LaunchMoeDispatchKernel`, and `MoeDispatchKernel` for the standalone operator. Keep low-risk layout field names unchanged.

- [ ] **Step 4: Remove combine-only paths**

Delete `LaunchDispatchCombineTileCombine`, `DispatchCombineTileCombine`, `RunCombine`, `PrepareExpertOutputIdentity`, `VerifyAndDump`, `--combine-return-only`, and combine timing/reporting from the new project.

- [ ] **Step 5: Keep dispatch verification**

Keep `CopyDispatchMetadataToHost`, payload comparison, debug dumps, `dispatch_e2e`, and `total_e2e` reporting.

- [ ] **Step 6: Verify statically and build**

Run:
```bash
rg -n "LaunchDispatchCombineTileCombine|DispatchCombineTileCombine|RunCombine|PrepareExpertOutputIdentity|combine-return-only|combine_e2e" kernels/manual/a2a3/moe_dispatch
bash kernels/manual/a2a3/moe_dispatch/run.sh --skip-run 1 --clean-build 1
```
Expected: the grep has no matches, and the build exits 0.
