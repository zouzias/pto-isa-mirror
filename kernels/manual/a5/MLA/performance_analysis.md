# MLA Performance Analysis: Why MLA Underperforms MHA/GQA on A5 NPU

## Root Cause: QK matmul inner dimension is larger than head_size

The MLA kernel uses weight-absorbed inference: `q_absorbed @ c_kv^T` instead of `q @ K^T`.
This changes the QK matmul **inner dimension** from `HEAD_SIZE` to `KV_LATENT_DIM`.

| | MHA | MLA (L=256) | MLA (L=64) |
|---|-----|-------------|-------------|
| QK inner dim | 128 | **256** | **64** |
| PV inner dim | 128 | 128 | 128 |

---

## Three compounding factors at kv_latent_dim=256

### 1. Pipeline is Vec (softmax) bound — extra Cube work creates sync stalls

With `CUBE_S0=128, CUBE_S1=128`, `calculateFittingCubeK` returns 128 (L0 buffer limit).

- MHA: `Tile_K=128 → kSegments=1` (one TMATMUL call per QK tile)
- MLA L=256: `Tile_K=256 → kSegments=2` (two TMATMUL calls per QK tile)

Simulator profile data (block_end timestamps) shows **Vec finishes last** in both MHA and MLA:

| Version | Cube (AIC) end | Vec (AIV) end | Vec lags Cube |
|---------|---------------|---------------|---------------|
| Baseline (V from GM) | 21,928 | 22,874 | +946 ticks |
| V recon TMOV | 27,238 | 28,139 | +901 ticks |

The pipeline is **Vec (softmax) bound**: softmax + GU takes more wall-clock time per tile
than QK + PV on Cube. The 4-stage Cube–Vec pipeline (`QK → P → PV → GU`) throughput
is limited by the Vec side.

When QK doubles (kSegments=2), Cube takes longer per tile, which **increases
Cube→Vec sync stalls**: Vec must wait longer for PV data before starting GU.
These stalls extend Vec's total wall-clock time, slowing the pipeline even though
Vec's compute workload (softmax + GU) is unchanged.

### 2. L1 buffer pressure increases 37%

| Buffer | MHA | MLA (L=256) |
|--------|-----|-------------|
| q_mat | 1x(128 x 128 x 2) = 32 KB | 1x(256 x 128 x 2) = 64 KB |
| k / c_kv_mat | 2x(128 x 128 x 2) = 64 KB | 2x(128 x 256 x 2) = 128 KB |
| p_mat | 3x(128 x 128 x 2) = 96 KB | 96 KB |
| v_mat | 2x(128 x 128 x 2) = 64 KB | 64 KB |
| **Total L1** | **256 KB** | **352 KB** |

More L1 pressure means fewer double-buffer slots and less room for pipeline overlap tuning.

### 3. GM bandwidth increases 50% on the K/V side

MHA reads `K + V` per S1 tile: `S1 x 128 + S1 x 128 = 256 elements`.
MLA reads `c_kv + V`: `S1 x 256 + S1 x 128 = 384 elements` (1.5x).
The weight absorption saves storing full K in the KV cache, but the kernel still reads
c_kv (which is 2x larger than K) from GM for every tile.

---

## Why kv_latent_dim=64 performs better (109% of MHA)

QK inner dim=64 → `kSegments=1` with smaller FLOPs.
L1 drops to 208 KB (less than MHA's 256 KB).
GM bandwidth drops to 0.75x of MHA.
The reduced Cube workload shortens Cube→Vec sync stalls, so Vec can proceed
through softmax → GU with fewer wait gaps. Net result: pipeline runs slightly
faster than MHA despite lower per-TMATMUL compute density.

---

## V reconstruction — implemented (TMOV L0C→L1 path)

V reconstruction (`c_kv @ W_uv`) is fused inside the kernel using direct TMOV (L0C→L1)
instead of GM roundtrip (TSTORE→GM→TLOAD). See `mla_performance_opt.md` for detailed
design, bug fixes, and session history.

### Implementation

- V recon TMATMUL produces vReconsAccTile in L0C (float)
- TMOV(vMatTile, vReconsAccTile) moves data directly from L0C to L1 (auto float→half)
- PIPE_MTE1→PIPE_FIX reverse sync with priming prevents binary flag collision
- PIPE_FIX→PIPE_MTE1 forward sync ensures V data ready before PV TEXTRACT
- No GM access needed for V data — eliminates `v.bin` GM read entirely

### Performance results (kv_latent_dim=256, HEAD_SIZE=128, S1=512)

| Version | Total Ticks | vs Baseline | Description |
|---------|------------|-------------|-------------|
| Baseline (V from GM) | 22,891 | — | V loaded directly from GM |
| V recon TMOV L0C→L1 | 28,155 | +23% slower | V = c_kv × W_uv, TMOV L0C→L1 |
| V recon GM roundtrip (old) | 40,009 | +75% slower | V = c_kv × W_uv, TSTORE→GM→TLOAD |

### Why V reconstruction is slower than baseline at kv_latent_dim=256

V reconstruction eliminates the V GM read (saving `S1 × HEAD_SIZE × 2B` per tile) and
reuses c_kv already resident in L1. However, it adds an extra TMATMUL (c_kv × W_uv) with
inner dimension = kv_latent_dim = 256, which increases the Cube workload in compute_pv.

Since the pipeline is **Vec (softmax) bound**, adding Cube work doesn't directly slow
the pipeline — Cube has slack time while Vec processes softmax. Instead, the slowdown
comes from **Cube→Vec sync stalls**: the extra V recon TMATMUL delays PV data delivery
to Vec, extending the wait gap between softmax and GU. Vec's compute workload is unchanged,
but its total wall-clock time (compute + sync waits) increases.

Profile data confirms this mechanism:

| | Cube time increase | Vec time increase | Vec > Cube gap |
|--|--------------------|-------------------|----------------|
| Baseline → V recon | +5,310 ticks | +5,265 ticks | ~900 ticks (constant) |

Cube and Vec increase by roughly the same amount — the extra Cube work propagates
as sync stall delay to Vec, which finishes last and determines pipeline throughput.

**Net effect**: GM bandwidth saved (no V GM read), but Cube→Vec sync stalls worsen
because PV data is delayed by the V recon TMATMUL. Since Vec is the bottleneck,
any increase in Vec's wall-clock time directly slows the pipeline.

### When V reconstruction would provide a performance gain

V reconstruction provides a per-kernel throughput gain only when the V recon TMATMUL is
cheaper than the V GM TLOAD it replaces. This happens when:

- **kv_latent_dim < head_size**: V recon TMATMUL inner dim is smaller, adding less Cube
  overhead. The shorter Cube work means shorter sync stalls for Vec, and the saved GM
  bandwidth (V TLOAD latency) outweighs the extra stall time.
- **GM bandwidth is the bottleneck**: If the pipeline is limited by GM access rather than
  Vec compute, eliminating the V GM read frees up bandwidth for other stages.

At kv_latent_dim=256 > head_size=128, neither condition holds. V reconstruction's benefit
is **system-level** (smaller KV cache → longer context, no V stored in GM), not per-kernel
throughput. For long-sequence inference, the reduced KV cache size may enable longer contexts
that outweigh the 23% per-token throughput loss.

---

## Summary

| Factor | kv_latent_dim=256 vs MHA | kv_latent_dim=64 vs MHA |
|--------|--------------------------|-------------------------|
| QK FLOPs | 2x (inner dim doubled) | 0.5x (inner dim halved) |
| L1 pressure | +37% (352 vs 256 KB) | -19% (208 vs 256 KB) |
| GM bandwidth (K-side) | +50% | -25% |
| Pipeline bottleneck | Vec (softmax) bound, Cube→Vec stalls worsened | Vec bound, stalls reduced |
| **Measured perf** | **83%** | **109%** |

| V reconstruction effect | kv_latent_dim=256 | kv_latent_dim=64 |
|------------------------|--------------------|--------------------|
| Extra Cube TMATMUL | inner dim=256 (same as QK) | inner dim=64 (cheap) |
| GM bandwidth saved | V TLOAD eliminated | V TLOAD eliminated |
| **vs baseline** | **+23% slower** (longer Cube→Vec stalls) | **expected faster** (light TMATMUL + no V GM read) |
| System benefit | smaller KV cache, no V in GM | smaller KV cache, no V in GM |

The core insight: **MLA's KV cache compression is a memory optimization, not a compute optimization.**
The pipeline is Vec (softmax) bound; adding Cube work (larger QK, V recon) worsens Cube→Vec
sync stalls and extends Vec's wall-clock time, slowing the pipeline.
MLA only gains NPU throughput when `kv_latent_dim < head_size`, where QK is cheaper and
the Cube→Vec stall gap shrinks, letting Vec proceed faster.
V reconstruction amplifies this pattern: it helps when `kv_latent_dim < head_size` (light extra
Cube work + no V GM read → shorter stalls) but hurts when `kv_latent_dim > head_size` (heavy
extra Cube work → longer stalls on already-delayed Vec).

### Potential optimization directions

Since the pipeline is Vec (softmax) bound, performance improvements should focus on:

1. **Speed up softmax/GU on Vec**: reduce Vec compute time per tile (fewer Vec ops,
   more efficient manual assembly). Any Vec time reduction directly improves throughput.
2. **Reduce Cube→Vec sync stall latency**: minimize the gap between "P ready on Vec"
   and "PV output delivered back to Vec". Overlapping V recon TMATMUL with softmax
   (e.g., moving V recon to compute_qk so PV starts immediately after P is ready)
   does not reduce total iteration time — the V recon delay shifts from one wait to
   another. Real improvement requires making Cube deliver PV data faster to Vec.
3. **System-level evaluation**: V recon eliminates v.bin GM read and reduces KV cache
   128x. For long-sequence inference, this system benefit may outweigh per-kernel loss.

---

## On-board performance at kv_latent_dim=64 (S0=14336, S1=14336)

### Measured vec cycles

| Version | Vec cycles | vs MHA | Description |
|---------|-----------|--------|-------------|
| MHA (flash_attn) | 34,789,652 | — | Baseline |
| GQA | 34,547,638 | -0.7% | Same per-head compute as MHA |
| MLA (93c6d32c) | 33,838,619 | **-2.8%** | No V reconstruction, cKvMatTNBuffers=2 |
| MLA (7fa3e671, no V recon) | 35,929,978 | **+3.2%** | No V reconstruction, cKvMatTNBuffers=(qkPreload+1)*kTileFactor |
| MLA (362a9c23, toggle, stride=0x20000) | 35,696,381 | **+0.7%** | L1 fix + 93c6d32c sync pattern restored, toggle assign_running_acc_tile |
| MLA (stride=0x10000, toggle) | 36,024,232 | **+3.5%** | stride 0x10000 with toggle — worse than 0x20000 |
| MLA (stride=0x20000, no toggle) | 34,176,416 | **-1.8%** | QK→evtID=0/addr=0x0, PV→evtID=1/addr=0x20000, independent PIPE_FIX→PIPE_M semaphores |
| MLA (stride=0x10000, no toggle) | 34,200,846 | **-1.7%** | QK→evtID=0/addr=0x0, PV→evtID=1/addr=0x10000, stride doesn't matter |
### Why MLA at kv_latent_dim=64 only gains 2.8% over MHA (not the predicted 9%)

The earlier prediction of 109% (9% gain) was based on simulator data with small S1=512.
With the real on-board test at S0=S1=14336 (56 tiles), the gain is only 2.8% because:

1. **Pipeline is Vec (softmax) bound at steady-state**: The softmax+GU workload on Vec
   is identical regardless of kv_latent_dim. Softmax operates on QK output tiles
   (Cube_S1 × Cube_S0) and GU operates on PV output tiles (Cube_S0 × HEAD_SIZE) —
   both shapes are unchanged. Reducing QK inner dim from 128 to 64 only makes Cube
   faster, but Vec can't process tiles any faster. Vec is already nearly saturated.

2. **GM bandwidth savings don't translate to throughput**: MLA reads c_kv(64)+V(128)=192
   elements per S1 row vs MHA's K(128)+V(128)=256 (0.75x). But the pipeline is
   compute-bound on Vec, not GM-bandwidth-bound, so the 25% GM savings is irrelevant.

3. **Reduced Cube→Vec stalls are marginal**: With long S1=14336, the pipeline reaches
   steady-state where Vec is fully saturated. The warmup phase (where Cube→Vec overlap
   matters most) is relatively short. In steady-state, faster Cube delivery only
   reduces the small remaining stall gap — Vec throughput is the limiting factor.

---

## Why commit 7fa3e671 regresses 6.2% vs commit 93c6d32c

Both commits run with ENABLE_V_RECONSTRUCTION=0 (no V reconstruction). The regression
from 33.8M to 35.9M vec cycles is caused by V reconstruction infrastructure being
**always compiled in**, wasting L1 space even when disabled.

### Root cause: unconditional buffer allocations

At commit 7fa3e671, these allocations are NOT conditional on `ENABLE_V_RECONSTRUCTION`:

| Buffer | 93c6d32c | 7fa3e671 | L1 waste |
|--------|----------|----------|----------|
| cKvMatTNBuffers | **2** | **(qkPreload+1)×kTileFactor = 6** | +4 buffers = +64KB |
| w_uvMatTile | **none** | **1** (unused in #else) | +16KB |
| **Total L1** | **208KB** | **288KB** | **+80KB (+38.5%)** |

At kv_latent_dim=64 with CUBE_S1=128, kTileFactor=2, qkPreload=2:
- c_kv buffer: 128×64×2B = 16KB each
- 6 buffers = 96KB vs 2 buffers = 32KB → 64KB wasted
- w_uv buffer: 64×128×2B = 16KB → unused in #else path

This 80KB waste increases L1 pressure from 208KB to 288KB (+38.5%), severely reducing
pipeline overlap efficiency and causing more Cube buffer stalls.

### Additional regression factors

1. **Extra V loading sync**: The #else path at 7fa3e671 adds `set_flag/wait_flag(PIPE_MTE2,
   PIPE_MTE1)` forward sync for V loading that wasn't in 93c6d32c. This adds 2 extra
   flag operations per sub_tile (~2-4 ticks each).

2. **More event IDs**: QK uses 0,1,2 + PV uses 3,4 (5 priming set_flags) vs 93c6d32c's
   QK 0,1 + PV 2,3 (4 priming set_flags). More priming and teardown overhead.

3. **Unconditional wait_flag(PIPE_FIX, PIPE_M, accTileEvtID)**: Was conditional on
   `sub_tile_id==0` at 93c6d32c, now unconditional per sub_tile in #else path.

### Fix: make buffer allocations conditional on ENABLE_V_RECONSTRUCTION

```cpp
#if ENABLE_V_RECONSTRUCTION
constexpr uint32_t cKvMatTNBuffers = (qkPreloadNum + 1) * kTileFactor;
#else
constexpr uint32_t cKvMatTNBuffers = 2;
#endif
```

Similarly, skip w_uvMatTile L1 allocation and vReconsAccTile when V recon is disabled.
This recovers ~80KB of L1 space and should restore most of the 6.2% regression.

### Further optimization: restore 93c6d32c compute_pv sync pattern in #else path

After the L1 fix, vec cycles went from 35.9M to 35.8M — only ~40K cycles recovered.
The remaining ~2M cycle gap (35.8M vs 33.8M) comes from structural sync differences
in the #else compute_pv path vs commit 93c6d32c:

1. **V loading moved from before P to after P**: In 93c6d32c, V TLOAD was done
   *before* `sm2pvSync.wait()`, allowing V data loading to overlap with the wait
   for softmax data. The current #else path moved V loading *after* P loading,
   preventing this overlap and delaying V data availability for PV TMATMUL.

2. **Forward sync for V loading**: The current #else path added `set_flag/wait_flag
   (PIPE_MTE2, PIPE_MTE1)` forward sync for V that wasn't in 93c6d32c. This adds
   2 extra flag operations per sub_tile (~4-8 ticks each).

3. **Unconditional wait_flag(PIPE_FIX, PIPE_M)**: In 93c6d32c, this was conditional
   on `sub_tile_id == 0`. The current code made it unconditional per sub_tile, adding
   an extra sync point for non-first sub_tiles within each tile.

4. **QK_EVENT_ID2 priming/teardown**: Added for V reconstruction's PIPE_M→PIPE_FIX
   sync, but unnecessary in the #else path. Adds 1 extra priming set_flag and 1
   extra teardown wait_flag.

Fix: restore the 93c6d32c compute_pv #else pattern:
- Move V loading back to before `sm2pvSync.wait()` (overlapping V TLOAD with wait)
- Remove forward sync for V loading (same as 93c6d32c — hardware implicit sync)
- Make `wait_flag(PIPE_FIX, PIPE_M)` conditional on `sub_tile_id == 0`
- Remove QK_EVENT_ID2 priming/teardown when V recon is disabled

### Critical fix: independent PIPE_FIX→PIPE_M semaphores for QK and PV

The biggest remaining regression (35.7M → 34.2M, recovering ~1.5M cycles) was caused by
QK and PV **sharing PIPE_FIX→PIPE_M event IDs** instead of having independent semaphores.

#### How it worked in 93c6d32c

`assign_running_acc_tile` was a single shared template with stride 0x10000. QK was
initialized at id=0 (addr=0x0, evtID=0) and PV at id=1 (addr=0x10000, evtID=1).
Because of the shared counter, the preload phase consumed 2 QK calls (toggling the counter),
and in the main loop the counter always settled to the same pattern:

- **QK always used evtID=0** (PIPE_FIX→PIPE_M semaphore 0)
- **PV always used evtID=1** (PIPE_FIX→PIPE_M semaphore 1)

These were **independent counting semaphores** — QK didn't have to wait for PV's
TMOV to complete, and PV didn't have to wait for QK's TMOV. Each type had its own
pipeline rhythm, maximizing overlap.

#### What changed and why it regressed

The V reconstruction refactoring introduced separate `QKAccTag` and `PVAccTag` template
instantiations of `assign_running_acc_tile`, with stride 0x20000 and **a toggle**
(`running_tile_buffer_idx ^= 1`). This meant QK and PV each had their own ping-pong
counter, causing them to **alternate event IDs** across tiles:

- Even tiles: QK→evtID=1, PV→evtID=0
- Odd tiles: QK→evtID=0, PV→evtID=1

Since PIPE_FIX→PIPE_M event IDs form counting semaphores per ID, this alternating
pattern **coupled QK and PV's pipelines**: a PV `set_flag(evtID=0)` could satisfy
a QK `wait_flag(evtID=0)` from a different tile, and vice versa. This created a
serialized dependency chain (QK→PV→QK→PV→...) instead of independent pipelines.

The serialization forced each type to wait for the other's TMOV to complete before
proceeding, preventing Cube-side pipeline overlap. Each tile iteration added
~14K extra cycles from this coupling, totaling ~1.5M across 112 tiles.

#### The fix: remove the toggle

Removing `running_tile_buffer_idx ^= 1` from `assign_running_acc_tile` makes each
type always return its initial id:

- **QKAccTag**: always id=0 → addr=0x0, evtID=0
- **PVAccTag**: always id=1 → addr=0x20000, evtID=1

This restores independent PIPE_FIX→PIPE_M semaphores, matching 93c6d32c's pattern.
Performance improved from 35,696,381 → 34,176,416 (~1.5M cycles recovered).

#### Stride doesn't matter

Both 0x20000 and 0x10000 strides with no-toggle give essentially the same performance
(34,176,416 vs 34,200,846). A5 has 256KB L0C, so both addresses are valid. The
independent semaphore pattern is what matters, not the L0C address.

With the toggle, 0x10000 stride was actually worse (36,024,232) than 0x20000 (35,696,381),
because 0x10000 stride with toggling causes QK and PV to share L0C regions more
frequently, increasing potential banking conflicts.

#### Remaining ~400K gap vs 93c6d32c (34.2M vs 33.8M)

The remaining gap likely comes from **preload phase serialization**. In 93c6d32c,
the shared counter caused preload QK tiles 0 and 1 to ping-pong (addr=0x0 and
addr=0x10000, evtID=0 and evtID=1). Both preload tiles could proceed immediately
(no wait — both evtIDs were primed). With the no-toggle fix, both preload tiles
write to addr=0x0 using evtID=0, so preload tile 1 must wait for preload tile 0's
TMOV to complete. This adds a fixed ~2-tile serialization overhead (~400K cycles).

This is a small fixed cost that's difficult to eliminate without reintroducing
preload-specific ping-pong logic (which would complicate the code for minimal gain).

#### PTO_PREFETCH doesn't help

Adding PTO_PREFETCH for V data (matching 93c6d32c's unconditional V prefetch) actually
made performance slightly worse (36,104,151 vs 35,696,381). SDMA prefetch may interfere
with the kernel's GM accesses during execution, or the V data already benefits from
spatial locality in L2 cache after the first few tiles.
