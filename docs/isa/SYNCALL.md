# SYNCALL

## Instruction Diagram

> No `SYNCALL.svg` is provided (unlike most vector ops). `SYNCALL` is a **cross-core control-plane** primitive rather than a per-element Tile transform; semantically it means "all selected participants rendezvous at this point before any may proceed."

The following conceptual diagram distinguishes the hardware (FFTS) and software (GM polling) paths:

```mermaid
flowchart TB
  subgraph hard [Hard Mode / FFTS]
    H1[Each participant reaches call site] --> H2[ffts_cross_core_sync etc.]
    H2 --> H3[wait_flag_dev etc.]
    H3 --> H4[Barrier complete]
  end
  subgraph soft [Soft Mode / GM Polling]
    S1[Write local GM slot counter] --> S2[Poll all slots until threshold]
    S2 --> S3[Barrier complete]
  end
```

## Summary

`SYNCALL` is a cross-core synchronization barrier supporting A2/A3 and A5 NPU backends. The template parameter `SyncCoreType` selects the core-type mode:

- **AIV-only** (default): `SYNCALL()` synchronizes all AIV cores.
- **AIC-only**: `SYNCALL<SyncCoreType::AICOnly>()` synchronizes all AIC cores (A2/A3 supports both hardware and software modes; A5 supports hardware mode only).
- **MIX (AIC+AIV)**: `SYNCALL<SyncCoreType::Mix>()` synchronizes mixed AIC and AIV cores.

`SyncAllMode` (specified explicitly in workspace-bearing overloads) selects **hardware mode (FFTS)** or **software mode (GM polling)**. The workspace-free overload corresponds to the hardware path.

## Mathematical Semantics

Not applicable as an elementwise arithmetic operation. `SYNCALL` expresses a **barrier arrival** relation:

- At a given dynamic program point, every core in the participant set defined by the current `SyncCoreType` must execute past the `SYNCALL` call before any participant may proceed beyond that point.
- Hardware mode: cross-core visibility is guaranteed by FFTS flags and device-side `wait_flag_dev` primitives.
- Software mode: each participant owns a monotonically-increasing counter slot in GM; `dcci`/`dsb` coherency primitives and polling determine "all participants have reached the current generation."

This semantic does **not** provide additional guarantees on GM or other buffer contents after the barrier; callers must still ensure data visibility explicitly or follow existing data-plane conventions.

## C++ Built-in Interface

Declared in `include/pto/common/pto_instr.hpp`. Software-mode interfaces use type-safe `GlobalTensor` and `Tile` parameters (constrained via SFINAE):

```cpp
// Hardware mode (all CoreType variants)
template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
PTO_INST void SYNCALL();

// Software mode — AIV-only (GlobalTensor + Vec Tile)
template <SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::AIVOnly,
          typename GlobalData, typename TileData,
          std::enable_if_t<is_global_data_v<GlobalData> &&
                           is_tile_data_v<TileData> && TileData::Loc == TileType::Vec, int> = 0>
PTO_INST void SYNCALL(GlobalData &gmWorkspace, TileData &ubWorkspace, int32_t usedCores = 0);

// Software mode — AIC-only (GlobalTensor + Mat Tile)
template <SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::AICOnly,
          typename GlobalData, typename TileData,
          std::enable_if_t<is_global_data_v<GlobalData> &&
                           is_tile_data_v<TileData> && TileData::Loc == TileType::Mat, int> = 0>
PTO_INST void SYNCALL(GlobalData &gmWorkspace, TileData &l1Workspace, int32_t usedCores = 0);

// Software mode — MIX (GlobalTensor + Vec Tile + Mat Tile)
template <SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::Mix,
          typename GlobalData, typename UbTileData, typename L1TileData,
          std::enable_if_t<is_global_data_v<GlobalData> &&
                           is_tile_data_v<UbTileData> && UbTileData::Loc == TileType::Vec &&
                           is_tile_data_v<L1TileData> && L1TileData::Loc == TileType::Mat, int> = 0>
PTO_INST void SYNCALL(GlobalData &gmWorkspace, UbTileData &ubWorkspace, L1TileData &l1Workspace,
                       int32_t usedCores = 0);
```

## Parameters

- `gmWorkspace`: `GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>>` (when `using namespace pto` coexists with Ascend C headers, qualify with `pto::` to avoid name collision with the compiler-intrinsic `Stride` enum). GM workspace for software mode; must be zero-initialized before the call. Each participating core occupies 8 `int32_t` values (cache-line-isolated sync counter).
- `ubWorkspace`: `Tile<TileType::Vec, int32_t, 1, SYNCALL_SOFT_SLOT_INT32>`. UB scratch for AIV-only and MIX software mode; capacity must be at least `usedCores * 8 * sizeof(int32_t)`.
- `l1Workspace`: `Tile<TileType::Mat, int32_t, 1, SYNCALL_SOFT_SLOT_INT32>`. L1 (cbuf) scratch for AIC-only and MIX software mode; used by `create_cbuf_matrix` to fill a sync value then DMA-transfer to GM.
- `usedCores`: Number of cores participating in the software barrier. When 0, automatically inferred — AIV-only uses `get_block_num()`, AIC-only uses `get_block_num()`, MIX uses `SYNCALL_GET_MIX_PARTICIPANT_COUNT()` (i.e. `AIC blocks × (1 + AIV ratio)`).

## Kernel Meta Macros

MIX-mode kernels must embed `.ascend.meta` information in the ELF for the runtime to correctly schedule AIC/AIV sub-kernels. Macros are defined in `include/pto/common/kernel_meta.hpp`:

```cpp
// AIV-side kernel (marked as MIX_AIV_MAIN, ratio fixed 0:1)
PTO_SYNCALL_AIV_KERNEL_META(kernelName);

// AIC-side kernel (marked as MIX_AIC_MAIN, specify AIC:AIV ratio)
PTO_SYNCALL_MIX_AIC_KERNEL_META(kernelName, aicRatio, aivRatio);
```

Usage example (1:2 mixed mode):

```cpp
PTO_SYNCALL_MIX_AIC_KERNEL_META(MyKernel_mix_aic, 1, 2);  // AIC kernel ELF
PTO_SYNCALL_AIV_KERNEL_META(MyKernel_mix_aiv);             // AIV kernel ELF
```

> Legacy aliases `PTO_SYNCALL_AIV_KERNEL_META` / `PTO_SYNCALL_MIX_AIC_KERNEL_META` are still usable and equivalent to the above macros.

> **dav-c220 auto-split**: When compiled with `--cce-aicore-arch=dav-c220`, Bisheng automatically generates AIC/AIV sub-kernels and corresponding `.ascend.meta` at a fixed physical ratio of **1:2** (two AIV subblocks per AIC block). In this case you do **not** need to hand-write `PTO_SYNCALL_MIX_AIC_KERNEL_META`, and you **cannot** override the ratio to 1:1 via meta alone (see "MIX 1:1" below).

## Compilation and Launch Guide (A2/A3)

This section uses the ST suite [`tests/npu/a2a3/src/st/testcase/syncall_mix_chevron/`](../../tests/npu/a2a3/src/st/testcase/syncall_mix_chevron/) as the reference for which **compile arch**, **kernel meta**, and **host launch** pattern to use for each `SyncCoreType` / mode / AIC:AIV ratio. The legacy [`syncall/`](../../tests/npu/a2a3/src/st/testcase/syncall/) directory retains the full register-ELF path; new development should prefer the per-scenario patterns in the chevron directory.

### Scenario Quick Reference

| Scenario | Sync Mode | Participants (910B4) | Compile `--cce-aicore-arch` | Kernel Meta | Host Launch | Reference Source |
|----------|-----------|------------------------|----------------------------|-------------|-------------|------------------|
| AIV-only | Hard | 40 AIV | `dav-c220-vec` | `PTO_SYNCALL_AIV_KERNEL_META` | chevron `<<<40>>>` | `syncall_aiv_hard_kernel.cpp` |
| AIV-only | Soft | 40 AIV | `dav-c220-vec` | none | chevron `<<<40>>>` | `syncall_aiv_soft_kernel.cpp` |
| AIC-only | Hard | 20 AIC | **`dav-c220`** (MIX auto-split, empty AIV stub) | auto-generated by Bisheng | chevron `<<<20>>>` | `syncall_aic_hard_kernel.cpp` |
| AIC-only | Soft | 20 AIC | `dav-c220-cube` | none | chevron `<<<20>>>` | `syncall_aic_soft_kernel.cpp` |
| MIX 1:2 | Hard / Soft | 60 (20 AIC + 40 AIV) | **`dav-c220`** (`pto_mix_st`) | auto-generated by Bisheng | chevron `<<<20>>>` | `syncall_mix_chevron_kernel.cpp` |
| MIX 1:1 | Soft | 40 (20 AIC + 20 AIV) | cube + vec `.o` each, one `.so` | optional (soft does not need FFTS context) | **dual-stream** chevron: AIC `<<<20>>>` + AIV `<<<20>>>` on separate streams | `syncall_mix11_soft_kernel.cpp` |
| MIX 1:1 | Hard | 40 (20 AIC + 20 AIV) | cube + vec `.o` each | **`PTO_SYNCALL_MIX_AIC_KERNEL_META(..., 1, 1)`** | **register ELF** + `rtKernelLaunchWithHandleV2` | `syncall_mix11_hard_kernel.cpp` |

### Path Details

#### 1. Chevron single-arch (AIV-only / AIC-only soft)

- **Compile**: one source file + matching arch (`dav-c220-vec` or `dav-c220-cube`) → separate `.so`.
- **Launch**: `kernel<<<blockDim, nullptr, stream>>>(...)`; `blockDim` equals the number of participating cores.
- **Hard AIV-only**: declare `PTO_SYNCALL_AIV_KERNEL_META` so the runtime schedules as `KERNEL_TYPE_MIX_AIV_1_0`.

#### 2. Chevron MIX auto-split (MIX 1:2, hard AIC-only)

- **Compile**: `--cce-aicore-arch=dav-c220`; Bisheng splits one source into AIC/AIV device code with embedded meta. CMake helper: `pto_mix_st(<target>)`.
- **Launch**: single chevron `<<<AIC block count>>>` (20 on 910B4); runtime brings up 20 AIC + 40 AIV at 1:2.
- **Hard AIC-only exception**: pure `dav-c220-cube` cannot establish the FFTS context required for AIC-only hard sync (hangs). Use `dav-c220` MIX compile instead: AIC runs `SYNCALL<AICOnly>()`, AIV is an empty stub, MIX launch supplies FFTS.

#### 3. Dual-arch dual-stream (MIX 1:1 soft)

- **Why**: on the ccec/bisheng path `GetTaskRation()` is always **2**; `dav-c220` auto-split is physically fixed at **1:2** and cannot be overridden to 1:1 via meta.
- **Compile**: same source compiled once as `dav-c220-cube` (`-DSYNCALL_MIX_BUILD_AIC`) and once as `dav-c220-vec` (`-DSYNCALL_MIX_BUILD_AIV`), linked into one `.so`.
- **Launch**: AIC and AIV kernels chevron `<<<20>>>` on separate `aclrtStream`s; soft sync uses GM polling and does **not** require a unified MIX FFTS context.

#### 4. Register ELF (MIX 1:1 hard)

- **Why**: hard MIX sync needs a **single** MIX FFTS context; dual-stream hard is invalid; chevron auto-split cannot produce true 1:1 on ccec (observed all-zero output). CANN built-in ops using `KERNEL_TYPE_MIX_AIC_1_1` only get `GetTaskRation()==1` on the **opc** compile path; ccec still reports 2.
- **Compile**:
  1. cube / vec objects with `PTO_SYNCALL_MIX_AIC_KERNEL_META(name, 1, 1)`;
  2. register-only objects with `-DSYNCALL_MIX_REGISTER_BUILD`;
  3. `make_mix_register_elf.py` merges nested device ELFs into a compact registration ELF.
- **Launch**: `rtRegisterAllKernel` + `rtKernelLaunchWithHandleV2(handle, tilingKey, 20, ...)`; runtime schedules 20 AIC + 20 AIV under one MIX context.

## Mode Support Matrix

### A2/A3

| Core Type | Hardware Mode | Software Mode |
|-----------|--------------|--------------|
| AIV-only | Supported | Supported |
| AIC-only | Supported | Supported |
| MIX | Supported | Supported |

### A5

| Core Type | Hardware Mode | Software Mode |
|-----------|--------------|--------------|
| AIV-only | Supported | Supported |
| AIC-only | Supported | Not supported |
| MIX | Not supported | Supported |

## Constraints

- Current implementation covers A2/A3 and A5 backends.
- Software mode supports AIV-only, AIC-only, and AIC+AIV mixed kernels.
  - AIC-only mode (A2/A3 only): AIC cores write and read GM slots directly via `copy_cbuf_to_gm` (L1→GM DMA).
  - A2/A3 mixed mode: AIC cores write GM slots via `copy_cbuf_to_gm` (L1→GM DMA); AIV cores write via UB workspace.
  - A5 mixed mode: A5 AIC (`dav-c310-cube`) lacks `copy_cbuf_to_gm`; instead delegates GM writes to AIV subblock 0 of the same block via `intra_block` signaling.
- A5 AIC-only hardware mode is supported: AIC uses `ffts_cross_core_sync` + `wait_flag_dev` for cross-core sync without needing `set_ffts_base_addr`.
- A5 AIC-only software mode is not supported: A5 AIC (`dav-c310-cube`) lacks an independent GM DMA write path (no `copy_cbuf_to_gm`), so GM-polling sync is infeasible.
- A5 hardware MIX mode is unavailable: the runtime API `rtGetC2cCtrlAddr` returns `RT_ERROR_FEATURE_NOT_SUPPORT` (207000) on A5 (`CHIP_DAVID`), preventing FFTS base address retrieval.
- Software mode requires all participating cores to enter the same barrier group in the same order; each core occupies 8 `int32_t` values in `gmWorkspace` for cache-line-isolated sync counting.
- Software mode only provides barrier-arrival semantics. If GM data visibility across the barrier is needed, callers must ensure coherency explicitly.
- The software-mode polling loop has a built-in back-off strategy (inserts `pipe_barrier` beyond a threshold) and timeout protection (default 1,000,000 iteration limit); on timeout the kernel-side breaks out, and CPU-SIM builds trigger an assertion.
- Hardware AIV-only `SYNCALL()` requires `.ascend.meta.<kernel>_mix_aiv` metadata in the kernel ELF so the runtime schedules it as `KERNEL_TYPE_MIX_AIV_1_0`.
- AIC-only kernels must call `__builtin_cce_kernel_type_set(1)` (corresponding to `KERNEL_TYPE_AIC_ONLY`) at the beginning of the function body for correct runtime scheduling.
- `SYNCALL` does not return a `RecordEvent` and is not used as a dependency in `Event<SrcOp, DstOp>`.
- In auto mode, consistent with existing sync instructions, no hardware synchronization is emitted.

## Examples

### Auto

In the **auto** build path, `SYNCALL` is consistent with existing sync strategies and **does not directly emit** cross-core hardware synchronization; it serves as a placeholder or for host/compiler-coordinated graph-level semantics. Typical operator development uses explicit `SYNCALL` in **manual** kernels.

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// In auto mode SYNCALL is a no-op (same as TSYNC etc. under auto)
void example_auto_noop() {
  SYNCALL();  // Does not trigger FFTS
}
```

### Manual — Hardware Mode

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// AIV-only: FFTS barrier across all AIV cores (requires correct kernel meta / ELF)
void example_hard_aiv() {
  SYNCALL();
}

// AIC-only: only available when compiled for AIC (__DAV_CUBE__); verified on A5 hard mode
void example_hard_aic() {
  SYNCALL<SyncCoreType::AICOnly>();
}

// MIX: paired AIC and AIV ELFs, see "Kernel Meta Macros" section above
void example_hard_mix() {
  SYNCALL<SyncCoreType::Mix>();
}
```

### Manual — Software Mode

Software mode requires a **zero-initialized** GM workspace and a correctly-sized UB/L1 Tile. `Mode` must be `SyncAllMode::Soft` (`Hard` ignores the workspace and behaves like the workspace-free `SYNCALL_IMPL`).

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_soft_aiv(__gm__ int32_t *gmPtr) {
  GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(gmPtr);
  Tile<TileType::Vec, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> ub;
  SYNCALL<SyncAllMode::Soft, SyncCoreType::AIVOnly>(gmWs, ub, 0);
}
```

MIX software mode requires both UB and L1 (Mat) Tiles; on A5 the AIC side delegates GM writes via a proxy path — see the "Constraints" section.
