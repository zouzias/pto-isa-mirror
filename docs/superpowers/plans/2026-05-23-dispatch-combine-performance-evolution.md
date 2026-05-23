# Dispatch Combine Performance Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved `kernels/manual/a2a3/dispatch_combine/design.md` Section 10 performance evolution while preserving README semantics, dynamic shape support, CPU golden validation, and pure PTO-ISA source constraints.

**Architecture:** Keep the single-process multi-rank correctness harness as the observable contract. Internally split the current two-kernel pipeline into metadata/pack, segment dispatch, segment return, and token restore kernels, with segment descriptors as the handoff protocol. P1/P2 replace data-plane helpers with PTO vector primitive and ping-pong-style row helpers; P3/P4 add descriptor-driven multi-block local simulation; P5 adds explicit local communication and identity expert backend boundaries without changing output semantics.

**Tech Stack:** C++17 host harness, ACL runtime, CCE direct kernel launch, PTO-ISA `Tile` / `GlobalTensor` / `TLOAD` / `TSTORE` / `TMULS` / `TADD`, A2/A3 `dav-c220-vec`, existing `./run.sh` black-box validation.

---

## File Structure

- Modify `kernels/manual/a2a3/dispatch_combine/dispatch_combine_kernel.cpp`
  - Add `SegmentDesc` and descriptor/ready views.
  - Replace restore scalar accumulation with `ScaleAddFloatRow` using `TMULS` + `TADD`.
  - Replace row-copy helper with explicit ping/pong tile storage helper boundary.
  - Split stage graph into metadata/pack, segment dispatch, segment return, and restore kernels.
  - Add local simulation communication backend helpers and identity expert backend helper.
- Modify `kernels/manual/a2a3/dispatch_combine/kernel_launchers.h`
  - Extend launcher ABI with `segment_desc`, `segment_ready`, and `return_ready` GM workspaces.
- Modify `kernels/manual/a2a3/dispatch_combine/main.cpp`
  - Add host-side `SegmentDescHost` golden descriptor generation and comparison.
  - Allocate/copy back segment workspace.
  - Add black-box benchmark timing around the launcher using ACL events or host timing if ACL events are unavailable.
- Modify `kernels/manual/a2a3/dispatch_combine/design.md`
  - Record what is implemented in P1-P5 local-simulation scope and what remains real multi-process/HCCL follow-up.
- Modify `kernels/manual/a2a3/dispatch_combine/task.md`
  - Add performance implementation verification evidence.
- Modify `kernels/manual/a2a3/dispatch_combine/todo.md`
  - Add and update P1-P5 implementation checklist.

## Verification Commands

Run fresh after each task that changes code:

```bash
bash kernels/manual/a2a3/dispatch_combine/run.sh --case all
bash kernels/manual/a2a3/dispatch_combine/run.sh --case smoke --ranks 1 --experts-per-rank 1 --tokens 3 --hidden 2 --topk 1
bash kernels/manual/a2a3/dispatch_combine/run.sh --case smoke --ranks 2 --experts-per-rank 2 --tokens 4 --hidden 65 --topk 2
bash kernels/manual/a2a3/dispatch_combine/run.sh --case smoke --ranks 2 --experts-per-rank 2 --tokens 4 --hidden 1024 --topk 2
bash kernels/manual/a2a3/dispatch_combine/run.sh --case smoke --ranks 2 --experts-per-rank 4 --tokens 8 --hidden 128 --topk 4
```

Run source constraints after code changes:

```bash
rg "AscendC::" kernels/manual/a2a3/dispatch_combine
rg "template <int Hidden>|switch \\(hidden\\)|LaunchPhases<|CopyGmVector|AddScaledGmVector|ClearVector<float>" kernels/manual/a2a3/dispatch_combine/dispatch_combine_kernel.cpp
```

Expected source constraint result: no matches.

---

### Task 1: P1 Vector Primitive Restore

**Files:**
- Modify: `kernels/manual/a2a3/dispatch_combine/dispatch_combine_kernel.cpp:227-293`

- [ ] **Step 1: Establish current black-box baseline**

Run:

```bash
bash kernels/manual/a2a3/dispatch_combine/run.sh --case smoke --ranks 2 --experts-per-rank 2 --tokens 4 --hidden 8 --topk 2
```

Expected: current code prints `[SUMMARY] All dispatch_combine cases passed.` before implementation changes.

- [ ] **Step 2: Add vector helper wrappers**

Insert after `FillVecTile`:

```cpp
template <typename Element, int TileElems = kPtoVectorTileElems>
PTO_INTERNAL void MulVecTile(uint64_t dst_ub_offset_bytes, uint64_t src_ub_offset_bytes, uint32_t elem_num,
                             Element scalar)
{
    using TileData = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elem_num; offset += TileElems) {
        uint32_t cur = (elem_num - offset > TileElems) ? TileElems : (elem_num - offset);
        TileData dst_tile(1, cur);
        TileData src_tile(1, cur);
        AssignUbTile<TileData, Element>(dst_tile, dst_ub_offset_bytes, offset);
        AssignUbTile<TileData, Element>(src_tile, src_ub_offset_bytes, offset);
        TMULS(dst_tile, src_tile, scalar);
    }
    pipe_barrier(PIPE_V);
}

template <typename Element, int TileElems = kPtoVectorTileElems>
PTO_INTERNAL void AddVecTile(uint64_t dst_ub_offset_bytes, uint64_t src0_ub_offset_bytes, uint64_t src1_ub_offset_bytes,
                             uint32_t elem_num)
{
    using TileData = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elem_num; offset += TileElems) {
        uint32_t cur = (elem_num - offset > TileElems) ? TileElems : (elem_num - offset);
        TileData dst_tile(1, cur);
        TileData src0_tile(1, cur);
        TileData src1_tile(1, cur);
        AssignUbTile<TileData, Element>(dst_tile, dst_ub_offset_bytes, offset);
        AssignUbTile<TileData, Element>(src0_tile, src0_ub_offset_bytes, offset);
        AssignUbTile<TileData, Element>(src1_tile, src1_ub_offset_bytes, offset);
        TADD(dst_tile, src0_tile, src1_tile);
    }
    pipe_barrier(PIPE_V);
}
```

- [ ] **Step 3: Replace scalar restore helper**

Replace `AddScaledFloatRow` with:

```cpp
PTO_INTERNAL void ScaleAddFloatRow(__gm__ float *dst_row, __gm__ float *src_row, int hidden, float scale)
{
    for (int h0 = 0; h0 < hidden; h0 += kPtoVectorTileElems) {
        int cur = (hidden - h0 > kPtoVectorTileElems) ? kPtoVectorTileElems : (hidden - h0);
        LoadVec<float>(kVecUb, src_row + h0, static_cast<uint32_t>(cur));
        LoadVec<float>(kAccUb, dst_row + h0, static_cast<uint32_t>(cur));
        MulVecTile<float>(kVecUb, kVecUb, static_cast<uint32_t>(cur), scale);
        AddVecTile<float>(kAccUb, kAccUb, kVecUb, static_cast<uint32_t>(cur));
        StoreVec<float>(dst_row + h0, kAccUb, static_cast<uint32_t>(cur));
    }
}
```

Update `RestoreOutputRows` to call `ScaleAddFloatRow`.

- [ ] **Step 4: Verify P1**

Run the five verification commands listed above. Expected: all print `[SUMMARY] All dispatch_combine cases passed.`.

- [ ] **Step 5: Verify P1 source constraints**

Run the two `rg` commands listed above. Expected: no matches. Also run:

```bash
rg "AddScaledFloatRow|GetTileValue<float>|SetTileValue<float>" kernels/manual/a2a3/dispatch_combine/dispatch_combine_kernel.cpp
```

Expected: no matches for `AddScaledFloatRow`, `GetTileValue<float>`, or `SetTileValue<float>`.

---

### Task 2: P2 Row Copy Ping-Pong Helper Boundary

**Files:**
- Modify: `kernels/manual/a2a3/dispatch_combine/dispatch_combine_kernel.cpp:8-10,262-269`

- [ ] **Step 1: Add ping/pong UB offsets**

Replace the UB constants with:

```cpp
constexpr uint64_t kVecPingUb = 0x00000;
constexpr uint64_t kVecPongUb = 0x04000;
constexpr uint64_t kVecUb = kVecPingUb;
constexpr uint64_t kAccUb = 0x08000;
constexpr int kPtoVectorTileElems = 1024;
```

- [ ] **Step 2: Replace row copy helper with ping/pong boundary**

Replace `CopyFloatRow` with:

```cpp
PTO_INTERNAL void CopyFloatRow(__gm__ float *dst_row, __gm__ float *src_row, int hidden)
{
    int chunk_id = 0;
    for (int h0 = 0; h0 < hidden; h0 += kPtoVectorTileElems) {
        int cur = (hidden - h0 > kPtoVectorTileElems) ? kPtoVectorTileElems : (hidden - h0);
        uint64_t ub = ((chunk_id & 1) == 0) ? kVecPingUb : kVecPongUb;
        LoadVec<float>(ub, src_row + h0, static_cast<uint32_t>(cur));
        StoreVec<float>(dst_row + h0, ub, static_cast<uint32_t>(cur));
        ++chunk_id;
    }
}
```

This keeps the correctness event model conservative while making the ping/pong storage contract explicit for subsequent event-chain tightening.

- [ ] **Step 3: Verify P2**

Run:

```bash
bash kernels/manual/a2a3/dispatch_combine/run.sh --case smoke --ranks 2 --experts-per-rank 2 --tokens 4 --hidden 2048 --topk 2
bash kernels/manual/a2a3/dispatch_combine/run.sh --case all
```

Expected: both print `[SUMMARY] All dispatch_combine cases passed.`.

---

### Task 3: P3 Segment Descriptor Workspace

**Files:**
- Modify: `kernels/manual/a2a3/dispatch_combine/kernel_launchers.h`
- Modify: `kernels/manual/a2a3/dispatch_combine/dispatch_combine_kernel.cpp`
- Modify: `kernels/manual/a2a3/dispatch_combine/main.cpp`

- [ ] **Step 1: Add host and device descriptor shape**

Use exactly seven `int32_t` fields in both host and kernel:

```cpp
struct SegmentDesc {
    int32_t src;
    int32_t dst;
    int32_t local_expert;
    int32_t global_expert;
    int32_t src_row_base;
    int32_t dispatch_row_base;
    int32_t rows;
};
```

- [ ] **Step 2: Extend launcher ABI**

Update `kernel_launchers.h` launcher signature by adding:

```cpp
int32_t *segment_desc, int32_t *segment_ready, int32_t *return_ready,
```

between `return_y` and `ranks`.

- [ ] **Step 3: Add descriptor view and builder in kernel**

Add a `SegmentRows` view:

```cpp
struct SegmentRows {
    __gm__ int32_t *base;
    PTO_INTERNAL int SegmentCount(const DispatchShape &shape) const
    {
        return shape.ranks * shape.experts_per_rank * shape.ranks;
    }
    PTO_INTERNAL int Id(const DispatchShape &shape, int dst, int local_expert, int src) const
    {
        return (dst * shape.experts_per_rank + local_expert) * shape.ranks + src;
    }
    PTO_INTERNAL __gm__ int32_t *Row(int segment_id) const
    {
        return base + segment_id * 7;
    }
};
```

Add `BuildSegmentDescriptors(shape, views)` after prefix generation. For each `(dst, localExpert, src)`, write fields in this order: `src,dst,localExpert,globalExpert,srcRowBase,dispatchRowBase,rows`.

- [ ] **Step 4: Consume descriptors in dispatch/return**

Refactor `BuildDispatchRows` and `BuildReturnRows` to iterate `segment_id` and read descriptor rows instead of recomputing nested loops internally. Keep the same row-copy semantics.

- [ ] **Step 5: Add host golden descriptor generation and comparison**

In `main.cpp`, add `segment_desc`, `segment_ready`, `return_ready` vectors/device buffers. Generate descriptor rows with the same formula as kernel and compare `segmentDesc` as int vector after execution.

- [ ] **Step 6: Verify P3**

Run `--case all` and `H=1024` smoke. Expected: descriptor comparison and all existing tensor comparisons pass.

---

### Task 4: P4 Multi-Block Segment Dispatch/Return and Token Restore

**Files:**
- Modify: `kernels/manual/a2a3/dispatch_combine/dispatch_combine_kernel.cpp`

- [ ] **Step 1: Split kernels**

Replace the current two-kernel launch with these kernels, in this stream order:

```text
MetadataPackKernel<<<1>>>
DispatchSegmentsKernel<<<segmentCount>>>
ReturnSegmentsKernel<<<segmentCount>>>
RestoreTokensKernel<<<R * M>>>
```

- [ ] **Step 2: Implement segment block mapping**

In segment kernels:

```cpp
int segment_id = get_block_idx();
if (segment_id >= segment_count || get_subblockid() != 0) return;
```

Read descriptor fields and copy `rows` rows for that segment. Write `segmentReady[segment_id] = 1` after dispatch and `returnReady[segment_id] = 1` after return.

- [ ] **Step 3: Implement token restore block mapping**

In restore kernel:

```cpp
int token_block = get_block_idx();
int src = token_block / shape.tokens;
int token = token_block % shape.tokens;
```

Zero only `out[src, token, :]`, then loop topK for that token and call `ScaleAddFloatRow`. No two blocks write the same out row.

- [ ] **Step 4: Verify P4**

Run `--case all`, `H=1024`, and `topK=4`. Expected: all pass and ready arrays equal all ones for existing segment count.

---

### Task 5: P5 Local Communication and Expert Backend Boundaries

**Files:**
- Modify: `kernels/manual/a2a3/dispatch_combine/dispatch_combine_kernel.cpp`
- Modify: `kernels/manual/a2a3/dispatch_combine/design.md`
- Modify: `kernels/manual/a2a3/dispatch_combine/task.md`

- [ ] **Step 1: Add explicit local backend helpers**

Wrap segment data movement in:

```cpp
PTO_INTERNAL void DispatchSegmentLocal(const DispatchShape &shape, const SegmentDesc &desc, const DispatchViews &views);
PTO_INTERNAL void ExpertIdentitySegmentLocal(const DispatchShape &shape, const SegmentDesc &desc, const DispatchRows &dispatch);
PTO_INTERNAL void ReturnSegmentLocal(const DispatchShape &shape, const SegmentDesc &desc, const CombineViews &views);
```

`ExpertIdentitySegmentLocal` intentionally preserves rows in-place for current identity expert semantics.

- [ ] **Step 2: Route segment kernels through backend helpers**

`DispatchSegmentsKernel` must call `DispatchSegmentLocal`. `ReturnSegmentsKernel` must call `ExpertIdentitySegmentLocal` then `ReturnSegmentLocal`.

- [ ] **Step 3: Document P5 scope honestly**

Update `design.md` Section 10.7 and `task.md` to state the current implementation has the P5 backend seam and local simulation backend; real multi-process HCCL/PTO window wiring remains a follow-up because the current executable is single-process/single-NPU.

- [ ] **Step 4: Verify P5 local backend**

Run all verification commands. Expected: all pass. Do not claim real HCCL communication is complete unless a multi-process HCCL test has been implemented and run.

---

### Task 6: Documentation, Task Tracking, and Final Verification

**Files:**
- Modify: `kernels/manual/a2a3/dispatch_combine/todo.md`
- Modify: `kernels/manual/a2a3/dispatch_combine/task.md`
- Modify: `kernels/manual/a2a3/dispatch_combine/design.md`

- [ ] **Step 1: Update todo.md**

Append a section `## 性能版本实现` with P1-P5 implementation checkboxes and update progress count.

- [ ] **Step 2: Update task.md**

Append a `## 性能版本实现记录` section containing exact commands and observed results.

- [ ] **Step 3: Run final verification**

Run all five black-box commands and both source-grep commands listed in this plan.

- [ ] **Step 4: Self-review**

Check:

```bash
rg "AscendC::" kernels/manual/a2a3/dispatch_combine
rg "template <int Hidden>|switch \\(hidden\\)|LaunchPhases<|CopyGmVector|AddScaledGmVector|ClearVector<float>" kernels/manual/a2a3/dispatch_combine/dispatch_combine_kernel.cpp
rg "TODO|TBD" kernels/manual/a2a3/dispatch_combine/design.md kernels/manual/a2a3/dispatch_combine/task.md kernels/manual/a2a3/dispatch_combine/todo.md
```

Expected: first two source constraints have no matches; docs have no `TODO` or `TBD` placeholders.

## Self-Review of This Plan

- Spec coverage: P1 maps to Task 1, P2 to Task 2, P3 to Task 3, P4 to Task 4, P5 local-simulation/backend seam to Task 5, verification/docs to Task 6.
- Scope note: real multi-process HCCL/PTO window communication is not implemented by this plan because the existing harness is explicitly single-process/single-NPU; Task 5 preserves a concrete backend seam and documents the residual follow-up rather than manufacturing a fake completion claim.
- Placeholder scan: no `TBD`, no `TODO`, no unspecified tests.
- Type consistency: descriptor field order is fixed as seven `int32_t` values across host and kernel; launcher extension uses raw `int32_t*` to avoid C ABI struct layout ambiguity.
