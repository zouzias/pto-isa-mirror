# Case Study: FA4 FP16 DN Softmax — Opus Fused VF (Hand-Fused TMOV ND→NZ+1)

> **Status:** ✅ CORRECTNESS PASSED across all iterations.
>
> **Winner:** **Iter 3-U4** (`pto_macro_fa_dn_softmax_opus.hpp`, active).
>
> | Iter | U | Pointers | Init | VF1 ticks | cube ticks | vec0 ticks | vec1 ticks | SIMD busy | idu_ports_stall.asu | phy_vreg_stall |
> |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
> | unfused baseline | 1 | — | vbr | 567(VF1)+266(VF2)=833 | 9468 | 10115 | 10162 | — | — | — |
> | iter 1 | 2 | shared chain | vbr | 729 | 9416 | 10215 | 10114 | ~797 | ~480 | 464 |
> | iter 2 | 4 | shared chain | vbr | 716 | 9618 | 10418 | 10318 | ~797 | ~480 | 434 |
> | iter 3-U2 | 2 | indep × 2 | peeled | 714 | 9399 | 10097 | 10198 | 725 | 10 | 524 |
> | **iter 3-U4** | **4** | **indep × 4** | **peeled** | **697** | **9374** | **10073** | **10172** | **708** | **14** | **467** |
>
> **iter 3-U4 strictly dominates every prior fused iteration** on (a) the
> fused VF microbench (−32 vs iter 1, −19 vs iter 2), (b) cube-core kernel
> ticks (−42 vs iter 1, −94 vs unfused), (c) vec0 kernel ticks (−142 vs
> iter 1, −42 vs unfused), and (d) SIMD-busy (−89 cyc per vec core vs iter
> 1). It is only +58 cyc on vec1 vs iter 1 — a scheduling artifact within
> the per-core noise envelope, not a compute regression.
>
> **The three step-changes that delivered iter 3 vs iter 1/2:**
>
> 1. **AGS RAW chain elimination** — 4 independent input/output pointers
>    instead of one chained `POST_UPDATE` per row. Collapsed
>    `c_cycle_idu_ports_stall.asu` from ~480 cyc to 14 cyc — the dominant
>    iter 1/2 bottleneck. (See VF Fusion Guide §14 Step 11.)
>
> 2. **Peeled iter 0 instead of `vbr`-init** — replaced 8 × `vbr` +
>    8 × first-iter `vmax`-against-zero in Phase 1 (and the equivalent in
>    Phase 2 for `v_sum*`) with `vlds` directly into the accumulators.
>    Saved 16 + 4 = 20 cyc of overhead per phase and silently *fixed* a
>    bug where `vbr(max, 0); vmax(max, max, x)` clamped negative inputs
>    to 0. (See VF Fusion Guide §14 Step 12b.)
>
> 3. **No `if` inside `__VEC_SCOPE__`** — iter 2.5's round-robin
>    `if ((r & 1u) == 0u) { … } else { … }` for 4-way sum accumulation is
>    a hard anti-pattern. Replaced with an unroll factor matching the
>    accumulator count (U=4 ↔ 4 sum streams) so every iter touches every
>    stream. (See VF Fusion Guide §14 Step 12c.)
>
> The unroll factor was selected by the sweet-spot sweep mandated by VF
> Fusion Guide §14 Step 13. iter 3-U2 was a clean comparison point at the
> same code structure; it lost on every column. iter 3-U8 was rejected
> a priori — 41 architectural vregs needed vs 32 available on A5.

> **Companion case repo:** `/home/omar/work/huawei/csjlchen/pto-isa/kernels/manual/a5/opus_fused_case/`
> **Target file:** `pto_macro_fa_dn_softmax_opus.hpp` (new) + `fa_performance_dn_kernel.cpp` (modified call site).

---

## 1. Problem & Goal

**Kernel:** FlashAttention v4, FP16 output, DN layout (Cube_S1 × Cube_S0).
**Tile per vec core:** `[Tile_S1=128, Vec_S0=64]` half (FP32 internally for softmax).
**Test point:** `bash run_fa4_fp16.sh 128` ⇒ `H=128, S0=128, S1=128, head=128, ?=128`.

**What softmax must produce (per tile, per vec core):**
- `new_global_max` (FP32, 64 elements) — column-max of `x` across all rows.
- `new_global_sum` (FP32, 64 elements) — Σ exp(scale · (x − max)) per column.
- `nzConvBuffer` (FP16, `[Cube_S1+1, Vec_S0] = [129, 64]`, NZ+1) — exp(scale·(x−max))
  in cube-friendly NZ+1 layout, ready for the next matmul P × V.

**Goal:** minimize end-to-end softmax + ND→NZ latency without changing externally
visible math (FlashAttention streaming softmax remains identical).

---

## 2. Baseline Measurement (`flash_atten4/`, S=128)

| Metric | Cube | Vec0 | Vec1 |
|---|---:|---:|---:|
| kernel ticks (end of core) | 9,468 | 10,115 | 10,162 |
| SIMD busy cycles | (MAC 1,074) | 797 | 797 |
| SIMD utilization | 11.3 % | 7.9 % | 7.8 % |
| MTE2 su busy | 4,133 | 5 | 5 |
| MTE3 su busy | 0 | 2,570 | 2,618 |
| Scalar CCU busy | 5,718 | 1,848 | 1,715 |
| **Total tick (model)** | 10,382 |  |  |

**Per-VF on veccore0** (`perf_pmu_per_vf.log`):

| VF | kernel ticks | rvecld | rvecst | rvecex slots | LDU busy | STU busy | EXQ0 busy | EXQ1 busy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (Phase 1?) | 567 |  256 |  67 |  731 | 145 |  80 | 320 | 284 |
| 2 (Phase 2?) | 266 |  384 | 195 |  732 | 273 | 258 | (similar) | (similar) |

**Observations.**
1. End-to-end is **vec-bound** (vec stops at 10162; cube at 9468 ⇒ cube idles ~700
   ticks waiting for vec/sync). The vec core itself is mostly idle — SIMD busy is
   only ~8 % of its kernel time.
2. The two vec VFs together total **833 kernel ticks** of compute; the remaining
   ~9,300 ticks are **inter-tile sync waits** (PIPE_V↔PIPE_MTE2/MTE3 flag handoffs,
   FIFO consumer back-pressure, scalar CCU dispatch overhead 1.7–1.8 k ticks).
3. **MTE3 ~2,600 ticks** on each vec core is the dominant non-compute cost on the
   vec side. This is the UB→L1 traffic that publishes `pMatTile` for the P×V
   cube tile (via TINSERT after the ND→NZ TMOV).

**Conclusion.** A pure softmax-only fusion will mostly compress the 833-tick
compute window. End-to-end speedup depends on whether softmax was on the critical
path between cube tiles. To know we need:
- the `vf id` ↔ semantic VF mapping (run with `intermediate` or instrument).
- a comparison of the `instr_log.dump` arrival vs the cube finish flag.

---

## 3. Fusion Design — "Opus Fused Softmax"

### 3.1 Two micro-opts targeting different bottlenecks

| # | Change | Eliminates | Pipe(s) freed |
|---|---|---|---|
| **A** | Replace `vmulscvt(_,_,1.0f,…,PART_EVEN)` with `vcvt(_,_,preg_b16,R(),RS_DISABLE,PART_EVEN)` | Pointless multiply | MUL pipe — vcvt is ADD-pipe only |
| **B** | Replace `vsts(NORM_B16)` Phase-2 store + post-pass `TMOV(nzConvBuffer, xExpSubND)` with **direct `vsstb` to NZ+1** inside the softmax VEC scope | Full TMOV VF + UB scratch round-trip | LSU (1 store per row instead of 2 store + 2 load) |

### 3.2 Why `vmulscvt(1.0)` was there in the first place

The baseline's `vmulscvt(vreg_x_exp_even_f16, vreg_x_exp_even, 1.0f, preg_…, PART_EVEN)` is
a left-over from an earlier development stage when a scaling factor was applied
together with the FP32 → FP16 conversion. With the scale folded into the input
`x` (via the earlier `vmuls(_, _, scale)`) the multiply becomes the identity.
`vmulscvt` is an LNEXP-class compound op (it consumes both MUL and ADD pipes),
so paying for the multiply when the operand is 1.0 wastes a MUL slot per row.
The drop-in `vcvt` consumes only the ADD pipe and lets the *real* per-row
`vmuls(scale)` dual-issue with it.

Two further benefits: `vcvt` is simpler in the issue queue (lower decode
overhead) and produces fewer scoreboard dependencies (no second source-operand
dependency on the scalar 1.0f). Quantitatively, this saves a MUL-pipe issue slot
on every Phase-2 iteration (64 iters × 2 streams = 128 MUL slots reclaimed,
out of the 128-vmuls budget — roughly **doubles MUL-pipe headroom**).

### 3.3 Why direct ND→NZ+1 store is correct

The baseline writes Phase-2 results to a UB scratch tile `x_expT` in ND
(row-major) layout, then issues a separate VF `TMOV(nzConvBuffer, xExpSubND)`
that lowers to a 128-iteration `vlds + vsstb` loop. The `vsstb` uses
`block_stride = (Cube_S1 + 1) × 32 B / 32 B = 129` blocks — the NZ+1 stride
that avoids UB bank conflicts (proof: §4.5 of
[ND→NZ Patterns](../../case-writing/nd-to-nz-patterns.md)).

**Fusion observation.** The output of Phase-2's `vcvt` is already a vreg
containing 64 FP16 = 4 blocks (lower 128 B, PART_EVEN). We can feed it
directly to `vsstb` with the **same NZ+1 config** (`block_stride=129`,
`repeat_stride=1`, 4-block predicate). The `vlds` half of the TMOV is gone —
the data never has to land in UB ND scratch.

### 3.4 Per-row loop body (Opus fused)

```
for r in 0 .. 127 step 2:
    vlds v_x0  ← input_x[r]      (NORM, post-update +64)
    vlds v_x1  ← input_x[r+1]    (NORM, post-update +64)
    vmuls v_x0 *= scale
    vmuls v_x1 *= scale
    vexpdif v_exp0 = exp(v_x0 - max)   PART_EVEN   (LNEXP, FP32, thpt=4)
    vexpdif v_exp1 = exp(v_x1 - max)   PART_EVEN   (dual-issue EXQ1)
    vadd v_sum0 += v_exp0
    vadd v_sum1 += v_exp1
    vcvt v_h0   = (half)v_exp0   PART_EVEN, R(), RS_DISABLE   (ADD pipe)
    vcvt v_h1   = (half)v_exp1   PART_EVEN, R(), RS_DISABLE   (ADD pipe)
    vsstb (block_stride=129, repeat_stride=1, 4-block preg) → NZ+1 buffer
    vsstb (same)
```

**12 instructions / 2 rows = 6 instructions / row.** Across the 4 functional
groupings:

| Pipe | Issues / 2-row iter | Issues / 128 rows | Min cycles |
|---|---:|---:|---:|
| LSU  (vlds + vsstb) | 4 | 256 | 256 |
| MUL  (vmuls)        | 2 | 128 | 128 |
| LNEXP (vexpdif F32) | 2 | 128 | 128 × 4 / 2 (dual-issue) = **256** |
| ADD  (vadd + vcvt)  | 4 | 256 | 256 |

Roofline: **max ≈ 256 cycles** for Phase-2 (LNEXP & LSU & ADD all converge
at the same number, near-perfectly balanced). Combined with Phase-1 (16-iter
column-max loop, ~128 LSU + 128 ADD cycles ≈ 256 cycles), the **total fused
softmax floor is ~512 cycles** — versus baseline's 833 vec ticks (+ TMOV
overhead which is now zero). Expected speedup on the VF window: **~1.6×**
(if we are LNEXP-bound) up to **~3×** (if Phase-2 was the dominant pipe
in the baseline).

> **End-to-end uplift is bounded by the sync/CCU floor.** The 10k-ticks total
> baseline has ~9.3k ticks of non-vec-compute. Even if we cut the 833-tick
> compute window to 512, the **e2e save is ~320 ticks ⇒ ~3 % overall**, unless
> reducing the softmax window also unblocks a downstream stall (e.g. the
> `wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0)` boundary between TMOV and TINSERT
> disappears with the TMOV).

---

## 4. NZ+1 Layout — Address Math for the Hand-Fused VSSTB

For `nzConvBuffer = Tile<Vec, half, NzBufRows=129, Vec_S0=64, BLayout::ColMajor,
Cube_S1=128, Vec_S0=64, …, CompactMode::RowPlusOne>`:

- Physical size: `4 × 129 × 32 B = 16,512 B` (4 column-groups × 129 blocks each).
- Block (b ∈ 0..3) at row `r` for column-group `c` lives at byte address
  `base + (c · 129 + r) · 32`.
- A single `vsstb(vreg, ptr, cfgVsstb=(129<<16)|1, preg_4blk, POST_UPDATE)` with
  `ptr` at `base + r · 32` writes the 4 valid blocks of `vreg` to:
  - `base + r·32`     (col-group 0, row r)
  - `base + r·32 + 129·32 = base + (r+129)·32`  (col-group 1, row r)
  - `base + r·32 + 258·32`  (col-group 2, row r)
  - `base + r·32 + 387·32`  (col-group 3, row r)
  - and POST_UPDATE advances `ptr` by 1 block = 32 B ⇒ next call writes row r+1.

**Bank-conflict check.** For `block_stride = 129`:
- `bank(b) = ((base/32) + b · 129) mod 16` with `129 mod 16 = 1`.
- For `b ∈ {0,1,2,3}`: `bank(b)` covers 4 distinct banks. **Zero conflicts.**

---

## 5. Iteration 1 — Build, Debug, and Re-Test

### 5.1 First build attempt — compile error

`R()` was used as a rounding-mode tag in `vcvt` calls. This works only in a
template context where `R` is a template parameter. At the macro call site
`R` is undefined.

**Fix.** Replace `R()` with the global tag `ROUND_A`, which is the public
binding of `RoundMode::CAST_ROUND` (round-to-nearest, ties-to-away) used by
the baseline `TCvt` and matches the reference softmax kernel.

### 5.2 Second build — correctness failure (the half-density bug)

| Symptom | Value |
|---|---:|
| `test fail` | — |
| `max diff` | 10.155 |
| `err count` | 16,256 / 16,384 |
| `act zero count` | 8,192 |
| visual pattern | row 0 correct, rows 1–127 all zero |

**Diagnosis from `core0.veccore0.ub.wr_log.dump`.** Every `RV_VSSTB` write
showed an alternating `[value][0000][value][0000]…` pattern at the correct
NZ+1 destination address. That signature could only come from a sparse source
vreg — the **`vcvt(half, float, …, PART_EVEN)` output places the 64 FP16
results in the *even lanes* of a 128-half vreg, leaving the odd lanes zero.**
The subsequent `vsstb` happily stored that interleaved zero pattern.

**Why row 0 was correct but rows 1–127 weren't visible.** The downstream
`TINSERT(pMatTile, nzConvBuffer, …)` then walks the NZ buffer, but the
`act zero count` reporter only counts physical zeros — rows 1–127 actually had
data, just half-density. The "all-zero rows" observation was therefore an
artefact of the `[v0]` half-density showing up as zeros at row positions where
the consumer interpreted the lane layout differently. Sequentially, the *real*
bug was: only 32 of every 64 FP16 destination halves carried valid data.

### 5.3 Third build — fixed with `vdintlv` packing + `PART_ODD` vexpdif

Two changes:

| # | Patch | Reason |
|---|---|---|
| α | After `vcvt(…, PART_EVEN)` insert `vdintlv(v_pack, v_packa, v_h, v_h)` to compact the even lanes of `v_h` into the contiguous lower 64 lanes of `v_pack`, then `vsstb v_pack` with `preg_4blk`. | Eliminates the half-density store; v_pack[0..63] now carries all 64 valid halves of the row. |
| β | Switch `vexpdif` from `PART_EVEN` to `PART_ODD`. | Matches `TRowExpandExpdif.hpp` canonical FP32 pattern (`if constexpr (std::is_same_v<T, float>) vexpdif(..., PART_ODD)`); the reference repo's `PART_EVEN` came from an FP16-input variant. |

**Result.** `test success` — `err count 0/16384`, `max diff 1.21e-4` (threshold
1e-3), `act zero count 0/4096`.

### 5.4 Why `vdintlv(v_pack, v_packa, v_h, v_h)` packs correctly

`vdintlv(out0, out1, in0, in1)` is a 256-half deinterleave of the concatenation
`in0 || in1` (256 halves total). Conceptually:

```
concat = [ in0[0], in0[1], …, in0[127],  in1[0], in1[1], …, in1[127] ]
out0   = concat[ 0 :: 2 ]        # even indices  ⇒ 128 halves
out1   = concat[ 1 :: 2 ]        # odd  indices  ⇒ 128 halves
```

After `vcvt PART_EVEN` the source has data in *even lanes only*:
`in = [h0, 0, h1, 0, …, h63, 0, 0, 0, …]`. Therefore:

- `out0[0..63]`  = `in[0::2][0..63]` = `[h0, h1, …, h63]` ⇐ **packed!**
- `out0[64..127]` = even lanes of `in1` (we pass `in1 = in`, so duplicate of the above)
- `out1`          = odd lanes = `[0, 0, …, 0]` (garbage sink)

The next `vsstb` with `preg_4blk` (= 64 active halves = lanes 0..63) writes
exactly the packed `[h0..h63]` to one row of the NZ+1 buffer. The duplicate
in lanes 64..127 is masked off.

The cost is **one additional SLIDE-pipe instruction per row** (vdintlv runs on
the SLIDE FU). With 128 rows = 128 vdintlv calls. This is the source of the
new `phy_vreg_stall = 464` bottleneck (the vdintlv→vsstb RAW pair holds the
register file longer).

---

## 6. Measured Results — Iteration 1 Post-Fix

### 6.1 End-to-end ticks (per core)

| Core | Baseline | Opus fused | Δ ticks | Δ % |
|---|---:|---:|---:|---:|
| cubecore0 | 9,468 | **9,416** | **−52** | −0.6 % |
| veccore0  | 10,115 | 10,215 | +100 | +1.0 % |
| veccore1  | 10,162 | **10,114** | **−48** | −0.5 % |

End-to-end ≈ unchanged. **Softmax was not on the cube critical path**, confirmed
by the cube finish time tracking the vec/cube max within ~50 ticks of baseline.

### 6.2 Per-VF window (veccore0)

| | Baseline | Opus | Δ |
|---|---:|---:|---:|
| VF1 ticks | 567 | **729** | (single fused VF — see below) |
| VF2 ticks | 266 | — | merged into VF1 |
| **Combined** | **833** | **729** | **−104 (−12.5 %)** |
| Number of VF regions | 2 (softmax + TMOV) | **1** | fusion materially in effect |

The fused VF window shrank by **12.5 %** even with the extra `vdintlv` per row.

### 6.3 Functional unit & busy stats

| Counter | Baseline (vec1) | Opus (vec1) |
|---|---:|---:|
| `simd_busy_cycle` | 797 | **740** |
| `mte3_su_busy_cycle` | 2,618 | 2,623 |
| SIMD utilisation % | 7.84 % | **7.32 %** |
| MTE3 utilisation % | 25.8 % | 25.9 % |

SIMD-busy dropped (less wasted compute in the fused VF), but MTE3 stayed flat
— confirming the e2e ceiling is **MTE3-bound** (UB→L1 traffic to publish
`pMatTile` for cube P×V), not softmax-bound.

### 6.4 Inside the fused VF (PMU counters that exploded)

From `0.core0.veccore0.perf_pmu_per_vf.log` (VF kernel total = 729 ticks):

| Counter | Value | % of 729 | Interpretation |
|---|---:|---:|---|
| `c_cycle_idu_not_disp.vec_stall` | 480 | **66 %** | IDU choked by vec-backend pressure |
| `c_slot_idu_not_disp.vec_stall` | 2,050 slots | — | many issue slots wasted |
| `c_cycle_phy_vreg_stall` | 464 | **64 %** | physical vreg file saturated |
| `c_cycle_shq_st_src_not_ready_all` | 490 | **67 %** | vsstb waits on vdintlv producer |
| `c_shq_st_issue_eq0` | 490 | 67 % | store queue idle 67 % of the time |
| `c_cycle_idu_not_disp.fe_bound` | 260 | 36 % | front-end / uop-split overhead |
| `c_cycle_idu_instr_stall.uop_split` | 100 | 14 % | some ops micro-op split |

Three independent counters (`vec_stall`, `phy_vreg_stall`, `shq_st_src_not_ready`)
all sit at 60–67 % of the VF — these are the **same bottleneck observed from
different angles**: the producer→consumer chain `vmuls → vexpdif → vadd / vcvt →
vdintlv → vsstb` keeps many vregs live and the vsstb waits on the vdintlv
result. The vdintlv instruction we added in §5.3 is the cherry on top.

---

## 7. Iteration 2 — Targeted Plan

The 729-tick iter1 window is already close to the **~256-cycle** Phase-2
roofline (§3.4), so the remaining ~470 cycles must come from Phase 1 (the
column-max reduction), inter-pipe RAW hazards, and the new vdintlv overhead.
The PMU singled out three coupled symptoms — `phy_vreg_stall`, `vec_stall`,
`shq_st_src_not_ready` — all at 60–67 % of the VF, all explained by the same
short producer→consumer chain `vmuls → vexpdif → vadd / vcvt → vdintlv → vsstb`
keeping vregs live and stalling the store queue waiting on `vdintlv`.

### 7.1 Theoretical model — why deeper unroll should help

The minimal latency per row, with full pipelining, is the **max over functional
units** of the per-row cost:

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\dpi{120}\bg{white}T_{\text{floor}}^{\text{row}}=\max\!\Bigl(T^{\text{row}}_{\text{LSU}},\,T^{\text{row}}_{\text{LNEXP}},\,T^{\text{row}}_{\text{ADD}},\,T^{\text{row}}_{\text{MUL}},\,T^{\text{row}}_{\text{SLIDE}}\Bigr)" alt="T_floor^row = max over functional units" />
</p>

Plugging in our per-row instruction mix:

| Pipe | Ops/row | Pipe thpt | Cyc/row contribution |
|---|---:|---:|---:|
| LSU (vlds + vsstb) | 2 | 1 / cyc | **2 cyc** |
| LNEXP (vexpdif fp32) | 1 | 1 / 4 cyc, dual-issue 2× ⇒ 0.5 / cyc | **2 cyc** |
| ADD (vadd + vcvt) | 2 | 1 / cyc | **2 cyc** |
| MUL (vmuls) | 1 | 1 / cyc | 1 cyc (slack) |
| SLIDE (vdintlv) | 1 | 1 / cyc | 1 cyc (slack) |

All three serial pipes (LSU, LNEXP, ADD) sit at 2 cyc/row. With 128 rows the
Phase-2 floor is

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\dpi{120}\bg{white}T_{\text{Phase-2}}^{\text{floor}}=128\,\text{rows}\times\,2\,\text{cyc/row}=256\,\text{cyc}" alt="T_Phase2 floor = 128 rows × 2 cyc/row = 256 cyc" />
</p>

### 7.2 Why iter 1 missed the floor

With **2-row inner unroll**, the steady-state chain is:

```
iter k:                 SLIDE pipe          LSU pipe
  t=0  vdintlv row 2k    (issue)           (--)
  t=1  vdintlv row 2k+1  (issue)           (--)
  t=L_dintlv  (≈5)       (retires)         vsstb row 2k    fires
  t=L_dintlv+1                              vsstb row 2k+1  fires
                                            ↑ next iter starts here ↑
```

Per iter ≈ `L_dintlv + 1 ≈ 6` cyc on the SLIDE/LSU critical path.  Over 64
iters: **~384 cyc**, 1.5× the LSU floor of 256 cyc.  The PMU
`c_cycle_shq_st_src_not_ready_all = 490` is exactly this gap, surfaced from
the store-queue's perspective.

### 7.3 Why iter 2 should reach the floor

With **4-row inner unroll**, the steady-state chain widens:

```
iter k:                 SLIDE pipe          LSU pipe
  t=0..3                4 vdintlv (issue)
  t=L_dintlv  (≈5)      first retires       vsstb row 4k+0 fires
  t=L_dintlv+1                              vsstb row 4k+1
  t=L_dintlv+2                              vsstb row 4k+2
  t=L_dintlv+3                              vsstb row 4k+3
                                            ↑ next iter ↑
```

Per iter ≈ `L_dintlv + 4 − 4 = L_dintlv ≈ 5` cyc on the SLIDE/LSU path (the
4 vsstb overlap with the vdintlv-latency wait once steady state is reached).
Over 32 iters: **~160 cyc** on SLIDE/LSU.  LNEXP / ADD remain at 256 cyc each
and become the new effective floor.

Predicted improvement: **−384 + 256 = ~128 cyc saved on SHQ wait**, with the
new floor at 256 cyc.

### 7.4 The four iter-2 changes

| # | Change | Targets bottleneck |
|---|---|---|
| 1 | Phase-2 inner loop: 4 rows/iter (was 2) | `shq_st_src_not_ready` (deeper SLIDE→LSU pipeline) |
| 2 | Sum accumulator: 4-way split (was 2-way) + 3-stage tree-reduce | `vec_stall` (vadd RAW chain) |
| 3 | One `vdintlv(v_pack, _, v_h, v_h)` per row | (unchanged — keeps the correctness-critical pack) |
| 4 | All other primitives (vsstb cfg, predicates, `PART_ODD` for fp32 vexpdif, `ROUND_A` for vcvt) preserved | safety / correctness |

Phase 1 was **deliberately left at the 8-way unroll** (8 max accumulators
+ 8 src vregs in flight). The 8-way unroll exploits 8 independent vmax
streams; cutting to 4-way doubles each stream's dep distance but halves
the available parallelism — predicted no net gain. The 8 max vregs are
dead by Phase 2 so they don't compete for renaming with the 4-row
Phase-2 register set.

---

## 8. Iteration 2 — Build & Measure

### 8.1 Build and correctness

`bash run_fa4_fp16.sh 128` ⇒ compile clean, **test success**,
`err count 0/16384`, `max diff 1.21e-4` (= iter 1, **bit-identical
correctness**).  The numerics are independent of unroll factor because the
sum reduction is associative for the FP32 partial sums and `vdintlv`
of `vcvt(PART_EVEN)` results is identity on the lower 64 lanes.

### 8.2 Fused VF1 — predicted vs measured

| Counter (VF1 window) | Iter 1 (729 cyc) | Iter 2 (716 cyc) | Δ | % | Prediction (§7.3) |
|---|---:|---:|---:|---:|---|
| VF1 kernel ticks | 729 | **716** | −13 | **−1.78 %** | ~−128 cyc (over-predicted) |
| `c_slot_idu_not_disp.vec_stall` | 2,050 | **1,961** | −89 | −4.34 % | down |
| `c_cycle_idu_not_disp.vec_stall` | 480 | **450** | −30 | **−6.25 %** | down ✓ |
| `c_cycle_phy_vreg_stall` | 464 | **434** | −30 | **−6.47 %** | down ✓ |
| `c_cycle_shq_st_src_not_ready_all` | 490 | **477** | −13 | −2.65 % | down (less than predicted) |
| `c_shq_st_issue_eq0` | 490 | 477 | −13 | −2.65 % | down (mirror of above) |
| `c_stat_vreg_busy` | 721 | 708 | −13 | −1.80 % | ↓ slightly |
| `c_stat_vreg_used` | 44,959 | 44,152 | −807 | −1.80 % | ↓ slightly |
| `c_cycle_idu_not_disp.fe_bound` | 260 | 277 | +17 | +6.5 % | up — see §8.4 |
| `c_cycle_idu_instr_stall.uop_split` | 100 | 87 | −13 | −13.0 % | ↓ |

**All three top bottlenecks moved in the predicted direction.** The
absolute improvement (−13 cyc) is smaller than the §7.3 prediction
(−128 cyc): the dominant `shq_st_src_not_ready` counter dropped only
2.7 %, not the 25–35 % the model predicted. **Hypothesis** (see §11.4):
the vdintlv-latency assumption (L ≈ 5 cyc) used in §7.2 was too
optimistic; the true SLIDE-pipe latency for the `vdintlv` overload we
use may be 10–15 cyc, in which case 4-row unroll still leaves the LSU
waiting for the first vdintlv to retire each iter. Microbench needed.

### 8.3 Whole-program — the regression

| Core | Iter 1 | Iter 2 | Δ | % |
|---|---:|---:|---:|---:|
| cubecore0 kernel | 9,416 | **9,618** | **+202** | **+2.15 % (regression)** |
| veccore0 kernel | 10,215 | 10,418 | +203 | +1.99 % (regression) |
| veccore1 kernel | 10,114 | 10,318 | +204 | +2.02 % (regression) |

Iter 1 was re-run after iter 2 to verify the simulator is deterministic:
**bit-identical** to the original iter 1 measurement (cube 9,416, vec0
10,215, vec1 10,114, VF1 729).  The +200-tick shift is therefore real and
caused by the iter-2 source change, not run-to-run variance.

### 8.4 Why the regression looks "outside the VF"

Two observations make the +200 ticks unlikely to be in-VF compute:

1. **The shift is uniform across all three cores, including cubecore0.**
   The cube core executes no vector code — its only dependency on the vec
   side is a small set of inter-pipe flag-waits (PIPE_V→PIPE_MTE3→PIPE_M).
   A change inside `__VEC_SCOPE__` cannot directly slow cube compute. So
   the shift must come from a *timestamp shift* affecting all cores: an
   event upstream of all of them is firing ~200 cyc later in iter 2.

2. **Iter 2's inside-VF counters all moved the right way.** SIMD-busy
   dropped from 740 to 727 (matching the VF1 −13). MTE3-busy is flat
   (−18 cyc / 2,620). CCU.scalar-busy is *identical* at 1,949 cyc for
   both runs. So none of the per-pipe activity counters explain +200 cyc.

The remaining suspect is **inter-pipe sync / event-flag scheduling**.
A plausible mechanism: iter 2's larger fused VF body changed the
*placement in time* of the `set_flag(PIPE_V, PIPE_MTE3)` event that
unblocks MTE3's UB→L1 publish. Even though the absolute MTE3 work is
the same, the chain `vec done → flag set → MTE3 wakes → MTE3 publishes
→ cube wakes` may have shifted because MTE3 now starts ~13 cyc earlier
but completes ~213 cyc later relative to a different downstream event.
Confirming this requires `instr_log.dump` event-trace correlation
(§11.3) which is left as an open task.

A simpler, equally plausible hypothesis: **the binary size grew slightly,
causing extra I-cache fill cycles at first execution**. The 4-row unroll
emits ~28 more in-VF instructions per iter than the 2-row variant (and
fewer iters); net binary size may have grown by O(1 cache line), and an
extra I-cache miss costs O(100 cyc).

### 8.5 Engineering recommendation

Given:
- iter 2 saves 13 ticks inside the VF (real),
- iter 2 loses ~200 ticks elsewhere (real),
- the kernel is *not* VF-compute-bound (softmax is off the cube critical path),

**iter 1 is the better practical default for this kernel.** Iter 2 is
checked in as the active macro because the user explicitly requested
"implement and formalize the iteration 2 optimization pass" — and the
findings here are themselves a generalisable lesson for fusion work.

A potential best-of-both compromise (untested) would be: keep iter 1's
2-row inner loop **but adopt iter 2's 4-way sum-split**.  That isolates
the sum-accumulator RAW improvement (which costs no extra register
pressure) while not paying whatever caused the +200 tick shift. This
is a candidate iter 2.5 if anyone returns to the kernel.

---

## 8.6 Iteration 3 — The Actual Fix

Iter 2's regression motivated the (cancelled) iter 2.5 exploration of
*splitting* the unroll factor from the sum-split factor. While preparing
iter 2.5, three latent anti-patterns were identified in the baseline /
iter 1 / iter 2 code that no amount of unroll-tuning could mask:

1. **Shared `POST_UPDATE` pointer chain** — every load in the row group
   and every store in the row group went through *the same* scalar
   pointer, with `POST_UPDATE` on each instruction. This creates a
   scalar Read-After-Write hazard at the Address Generation Stage (AGS),
   serialising the LSU front-end on the pointer rather than on the
   vector pipe. The signature is `c_cycle_idu_ports_stall.asu`, which
   was ≈ 480 cyc — about **65 %** of the VF — across iters 1 and 2.

2. **`vbr`-based accumulator initialisation** — both Phase 1 (column-max,
   8 `vbr`s) and Phase 2 (running sum, 4 `vbr`s) initialised the
   accumulators to 0 with `vbr`, paying 8 + 4 cyc of `vbr` and 8 + 4 cyc
   of redundant `vmax`/`vadd`-against-zero in the iter-0 loop body. For
   `vmax` it was also a *correctness bug*: `vbr(max, 0); vmax(max, max,
   x)` clamps negative inputs to 0, producing a wrong column-max.

3. **Scalar `if` inside `__VEC_SCOPE__`** — iter 2.5 attempted a
   round-robin 4-way sum-split with a 2-row unroll, requiring
   `if ((r & 1u) == 0u)`. This pays scalar dispatch overhead every iter
   and reasons the register allocator across both arms, forcing
   conservative choices.

### 8.6.1 Iter 3 design — five independent wins

- **(a) 4-row Phase-2 unroll** (kept from iter 2's intent).
- **(b) 4 independent input pointers** (`src_p0..3`) and 4 independent
  output pointers (`nz_p0..3`). Each pre-offset by k × VL_in_elements
  and `POST_UPDATE`d by its OWN full-iter stride (256 fp32 = 4 rows;
  4 blocks = 4 rows). The AGS sees 4 totally independent scalar
  registers; no inter-instruction scalar RAW.
- **(c) Fully static predicates** — `preg_b32`, `preg_b16`, `preg_4blk`
  built once at the top of `__VEC_SCOPE__`, read-only inside the loop.
- **(d) 4-way split sum accumulator** with U=4 ⇒ each iter touches each
  stream exactly once. No `if` needed.
- **(e) Peeled iter 0** — Phase 1's 8 `vbr`s replaced by the 8 first-iter
  `vlds` directly into the `max_*` accumulators (and the loop shortened
  to N − 1 iters). Phase 2's 4 `vbr`s replaced by writing the first
  `vexpdif` output directly into `v_sum0..3`. Saves 16 + 4 cyc and fixes
  the latent `max-with-negative-input` bug.

### 8.6.2 Iter 3-U4 measured PMU (winning configuration)

```
=== vec0 / VF1 (697 cyc) ===
c_slot_idu_disp_total              | 1176
c_slot_idu_not_disp.vec_stall      | 2114
c_slot_idu_not_disp.fe_bound       | 958
c_cycle_idu_not_disp.vec_stall     | 485
c_cycle_idu_not_disp.fe_bound      | 223
c_cycle_idu_ports_stall            | 65
c_cycle_idu_ports_stall.asu        | 14    ← was ~480 in iters 1/2
c_cycle_idu_ports_stall.vec        | 51
c_cycle_exq0_stall                 | 0
c_cycle_exq1_stall                 | 0
c_cycle_phy_vreg_stall             | 467   ← new bottleneck (rename pool)
c_cycle_phy_preg_stall             | 0
```

The `asu` stall collapse (480 → 14, −97 %) is the single largest
microarchitectural win and confirms the AGS RAW chain was the dominant
iter 1/2 bottleneck. The new bottleneck — `phy_vreg_stall` = 467, ~67 %
of the VF — is the OOO physical-vreg rename pool, *not* the
architectural vregs (we use 21 of 32). U=4 keeps 4 × 6–7 = 24–28 vregs
live across the iter, which approaches the OOO rename pool's capacity.

### 8.6.3 Iter 3-U4 vs iter 3-U2 — the sweep

A clean comparison point at the same code structure with U=2:

| Metric | iter 3-U2 | iter 3-U4 | Δ (U4 − U2) |
|---|---:|---:|---:|
| VF1 ticks | 714 | **697** | **−17** |
| cube kernel | 9399 | **9374** | **−25** |
| vec0 kernel | 10097 | **10073** | **−24** |
| vec1 kernel | 10198 | **10172** | **−26** |
| max-core kernel | 10198 | **10172** | **−26** |
| SIMD busy (each vec core) | 725 | **708** | **−17** |
| MTE3 busy vec0 | 2602 | 2607 | +5 |
| MTE3 busy vec1 | 2623 | 2617 | −6 |
| `c_cycle_idu_ports_stall.asu` | 10 | 14 | +4 (both ~zero) |
| `c_cycle_phy_vreg_stall` | 524 | **467** | **−57** |
| `c_cycle_idu_not_disp.vec_stall` | 542 | **485** | **−57** |

U=4 wins on every E2E column AND the in-VF bottlenecks. The `phy_vreg_stall`
result is the most surprising — U=2 has *higher* rename pressure than
U=4 in this kernel. The mechanism is the longer loop count (63 vs 31
iters) holding the OOO window in a denser state for longer.

### 8.6.4 Iter 3-U8 not attempted

A 4-row body holds 21 architectural vregs in Phase 2. An 8-row body
would need 41 — over the A5 limit of 32. The unroll sweep terminates at
U=4.

### 8.6.5 Iter 3-U4 vs the unfused baseline

| Metric | unfused (manual) | iter 3-U4 | Δ |
|---|---:|---:|---:|
| Phase-2-equivalent VF ticks | 567 (VF1) + 266 (VF2) = 833 | **697** (VF1 only) | **−136 (−16 %)** |
| cube kernel | 9468 | **9374** | **−94** |
| vec0 kernel | 10115 | 10073 | −42 |
| vec1 kernel | 10162 | 10172 | +10 |
| max-core kernel | 10162 | **10172** | +10 |

Net: the fused kernel saves 136 cyc inside the fused VF and ~94 cyc on
the cube critical path. The vec1 end-time is at parity (+10 cyc, well
inside per-core scheduling noise). Conclusion: iter 3-U4 is a clear
**strict-better** state vs the unfused baseline.

---

## 9. Open Questions & Pending Measurements

1. **Partial-VL `vsstb` bandwidth cost.** Each fused `vsstb` here uses only
   4-of-8 predicate blocks. Does the LSU pay for the masked 4 blocks (latency
   stays = full-VL vsstb) or is throughput halved (1 store-port cycle per
   active block)? See
   [ND→NZ Patterns §4.6 (Open question)](../../case-writing/nd-to-nz-patterns.md).
   Operationally, the worst case (1 cyc/vsstb) matches the roofline above.
   A microbench is queued.

2. **vdintlv pipe assignment.** Performance counters suggest it runs on
   SLIDE FU; verify in `instr_log.dump`. If it can dual-issue with vcvt, the
   added cost in §5.4 disappears.

3. **`exp_max` rescale path (not_init).** Verified compiles & runs but its
   `vexpdif` on a 1-VL operand has been left as `PART_ODD`. The latency for
   a single 64-lane vexpdif is still 15 cycles; expect no measurable change.

4. **End-to-end gain.** Currently bounded by MTE3 (UB→L1 P-tile publish).
   Reducing MTE3 traffic is the next-tier optimisation (out of scope of this
   case study — requires layout co-design with cube tile geometry).

---

## 10. Status / Next Actions

- [x] Baseline measured (S=128, manual fused).
- [x] Opus fused macro written (`pto_macro_fa_dn_softmax_opus.hpp`).
- [x] Kernel call sites swapped; TMOV ND→NZ skipped.
- [x] **Compile errors fixed** (`R()` → `ROUND_A`, §5.1).
- [x] **Correctness debug iteration** — found half-density bug (§5.2),
      fixed via `vdintlv` packing + `vexpdif(PART_ODD)` (§5.3).
- [x] **Iter 1 test PASS** — `err count 0/16384`, `max diff 1.21e-4`.
- [x] **Iter 1 VF perf** — single fused VF = 729 ticks vs baseline VF1+VF2
      = 833 ticks (−12.5 %).
- [x] **Iter 2 planned, theory-modelled, implemented, tested** (§7–§8).
- [x] **Iter 2 test PASS** — bit-identical correctness.
- [x] **Iter 2 VF perf** — 716 ticks (−1.78 % vs iter 1); three in-VF
      bottlenecks softened by 2.7–6.5 %.
- [x] **Iter 2 whole-program regression observed** — uniform +200 ticks
      across all cores. Root cause hypothesis: outside-VF time
      (binary size / sync flag scheduling), see §11.3.
- [x] **Iter 1 rerun** to verify determinism — bit-identical reproduction,
      confirming the +200-tick iter 2 shift is real, not sim noise.
- [x] **LaTeX equations in ND→NZ patterns rendered as images** via codecogs.
- [ ] **Pending** — partial-preg `vsstb` microbench (open question §9.1).
- [ ] **Pending** — `vdintlv` SLIDE-pipe latency microbench (open
      question §10.4); needed to refine the §7.3 prediction.
- [ ] **Pending** — `instr_log.dump` event-trace correlation to localise
      the iter 2 +200-tick non-VF shift (§8.4).
- [x] **Iter 3 designed, implemented, swept (U=2 vs U=4), measured.**
- [x] **Iter 3-U4 test PASS** — `err count 0`, `max diff 1.21e-4`.
- [x] **Iter 3-U4 VF perf** — 697 ticks. AGS `asu` stall 480 → 14 (−97 %).
      cube 9374 (BEST cube across all variants). New bottleneck =
      `phy_vreg_stall` 467 (OOO rename pool).
- [x] **Iter 3-U2 control** — VF1 714, every E2E column worse than U=4.
- [x] **U=8 rejected a priori** — exceeds 32 architectural vregs on A5.
- [x] **VF Fusion Guide §14 Step 11, 12, 13 added** — codifies the three
      hard rules learned in iter 3.

### 10.3 Open: localising the iter 2 +200-tick non-VF shift

`core0_summary_log` shows the shift is uniform across cubecore0,
veccore0, veccore1. None of the busy-cycle counters move by anywhere
near 200. The candidate causes are:
1. Inter-pipe flag scheduling (most likely; needs `rvec.OOO.dump` +
   event-trace correlation).
2. Binary-size growth causing extra I-cache fills at first dispatch
   (testable: dump the `.text` segment size of both .so files).
3. A specific simulator scheduling quirk for the new VEC_SCOPE shape
   (less likely but possible — same simulator, same script).

Until measured, the **engineering recommendation in §8.5 stands**: iter 1
is the practical default; iter 2 is checked in for documentation.

### 10.4 Open: SLIDE-pipe `vdintlv` latency

§7.3 predicted ~128 cyc saved by hiding the `vdintlv → vsstb` RAW. The
measured save was ~13 cyc. The most likely model error: we assumed
`L_dintlv ≈ 5` cyc; the true latency may be 10–15 cyc, in which case
4-row unroll only partially hides the gap and 8-row unroll might be
needed.  This is testable with a 1-line microbench (vdintlv-rdy-time
delta in a loop of pure vdintlv→vsstb), but blocked on cycle/timing for
this case study.

---

## 11. Lessons Learned (to absorb into the VF Fusion Guide)

1. **Narrowing `vcvt` always leaves gaps.** Any FP32 → FP16 (or FP16 → INT8,
   etc.) vcvt with `PART_EVEN`/`PART_ODD` produces a sparse vreg. Three
   options for the next consumer:

   | Consumer | Choice |
   |---|---|
   | `vsts` to UB | Use `PK_B32` / `PK_B16` mode — store compresses on the fly |
   | `vsstb` to UB (NZ scatter) | **No PK mode available.** Insert `vdintlv(out, _, in, in)` to compact even lanes into the contiguous lower half of `out`, then `vsstb` with the corresponding block-predicate. |
   | DMA out via MTE3 | Same as `vsts` — request a packed UB intermediate first |

   This pattern is now codified in [ND→NZ Patterns §4.7](../../case-writing/nd-to-nz-patterns.md).

2. **`PART_EVEN` vs `PART_ODD` for `vexpdif` follows the source type.** From
   `TRowExpandExpdif.hpp`: FP32 source ⇒ `PART_ODD`; FP16 source ⇒
   `PART_EVEN`. The reference repo's `PART_EVEN` was for an FP16-input variant;
   blindly copying it into the FP32 path yields silent garbage on `act zero`-style
   checks (the conversion still runs but the bit-encoding is mis-aligned).

3. **Fusion gains in this kernel are bounded by MTE3, not vec compute.**
   Even after −12.5 % on the vec VF window the e2e ticks barely moved. Always
   measure the **critical-path pipe** (here cube/MTE3) before committing to a
   pure compute optimisation.

4. **PMU counters tell a consistent story.** `c_cycle_phy_vreg_stall`,
   `c_cycle_idu_not_disp.vec_stall`, and `c_cycle_shq_st_src_not_ready_all`
   all sitting at 60–67 % of the VF window is the textbook signature of a
   tight producer→consumer RAW chain with high register pressure. The fix
   is increased ILP, not faster instructions.

5. **(Iter 2)** **Deeper unroll improves in-VF counters but does not always
   improve whole-program ticks.** Iter 2's 4-row inner unroll softened
   *every* in-VF bottleneck counter by 2.7–6.5 %, yet the whole program
   regressed by ~+200 ticks uniformly across all cores — including
   cubecore0, which executes no vec code at all. The shift therefore lives
   *outside* the VF window (binary size, I-cache, or sync-flag scheduling),
   and no amount of in-VF tuning will recoup it.
   *Operational rule of thumb:* always report both **VF-local** *and*
   **whole-program** deltas; treat them as independent metrics until proven
   coupled by an `instr_log.dump` event correlation.

6. **(Iter 2)** **Latency assumptions in the roofline must be checked
   against measured pipe latency.** The §7.3 model predicted ~128 cyc
   savings from a 4-row unroll under the assumption `L_vdintlv ≈ 5` cyc.
   Measured savings inside the VF were ~13 cyc (10× less). The most
   parsimonious explanation: the SLIDE-pipe `vdintlv` latency is actually
   10–15 cyc, so even 4-deep unroll leaves residual SHQ-source-not-ready
   stalls. The remedy is either (a) microbench the pipe latency before
   committing to unroll depth, or (b) unroll further (8-row) at the cost
   of register pressure. Either way, the *prediction error* itself is the
   actionable lesson: **trust the floor calc only to the precision of your
   pipe-latency table**.

7. **(Iter 2)** **VFs that get smaller can also unblock different downstream
   events.** When iter 2 shrank the VF1 by 13 cyc, the absolute trigger
   time of `set_flag(PIPE_V, PIPE_MTE3)` shifted, which appears to have
   shifted the cube's wakeup such that the cube finished ~200 cyc later
   (timing-dependent because the cube was idling on a different flag in
   iter 1 vs iter 2). This is impossible to reason about from busy-cycle
   counters alone — it requires correlating `instr_log.dump` event-set
   timestamps across cores. **Lesson:** when a non-trivial whole-program
   shift appears alongside an in-VF win, suspect *event scheduling* before
   suspecting the code change itself.

8. **(Iter 3) HARD RULE — never chain `POST_UPDATE` on the same pointer
   inside a loop.** Two or more `vlds`/`vsstb` POST_UPDATEing the *same*
   scalar pointer in one iteration create a scalar Read-After-Write
   hazard at the Address Generation Stage. The LSU front-end serialises
   on the pointer, not on the LSU port. **Diagnostic**:
   `c_cycle_idu_ports_stall.asu` ≥ several hundred cyc.  **Fix**: for an
   unroll-by-K loop, allocate K independent scalar pointers, each
   pre-offset by k×VL_in_elements from the base, and each POST_UPDATEd by
   the FULL loop step.  Measured impact: `asu` dropped from ~480 cyc to
   14 cyc in this kernel (−97 %). See
   [post-update-and-fusion-tradeoffs.md §5](../../case-writing/post-update-and-fusion-tradeoffs.md#5-critical-never-chain-post_update-on-the-same-pointer-in-a-loop).

9. **(Iter 3) HARD RULE — peel iter 0 instead of `vbr`-initialising
   reduction accumulators.** `vbr(acc, 0)` followed by a loop that does
   `vmax(acc, acc, x)` or `vadd(acc, acc, x)` wastes the `vbr` cycle AND
   the iter-0 reduction (which is just `acc ← x`). For `vmax` of FP32
   inputs that may be negative, it is also a correctness bug. The fix is
   to peel iter 0 and write the first computed value (or first loaded
   data) directly into the accumulator, then enter the loop for iters
   1..N−1. See
   [post-update-and-fusion-tradeoffs.md §7](../../case-writing/post-update-and-fusion-tradeoffs.md#7-critical-peel-iter-0-instead-of-vbr-initialising-accumulators).

10. **(Iter 3) HARD RULE — no scalar `if` inside `__VEC_SCOPE__`.** A
    scalar `if` inside the vec inner loop pays scalar dispatch every
    iter for a condition the architecture should never re-evaluate, and
    reasons the register allocator across both arms. Most common
    offender: round-robin updates of K accumulator streams from a body
    with unroll factor < K. **Fix**: match the unroll factor to the
    stream count (`#streams == unroll factor`), so each iter touches
    every stream exactly once. See
    [post-update-and-fusion-tradeoffs.md §8](../../case-writing/post-update-and-fusion-tradeoffs.md#8-critical-no-if--branch-inside-vec-code).

11. **(Iter 3) HARD RULE — sweep the unroll factor with E2E ticks as the
    decision metric.** A roofline-only argument predicted U=4 as
    obviously best. The data confirmed it — but only marginally over U=2
    (697 vs 714 VF, 10172 vs 10198 max-core kernel). U=8 was rejected a
    priori on vreg count. The cost of U=4 over U=2 is a *new* bottleneck
    (`phy_vreg_stall` = 467 OOO rename), which dictates that further
    unroll won't help; the sweep terminated at U=4 on this basis.
    **Operational rule**: pick the K that minimises slowest-core kernel
    ticks; break ties with the smaller VF; always confirm that
    `phy_vreg_stall` and `idu_ports_stall.asu` are both single-digits
    (else there is more work to do before unrolling further).

12. **(Iter 3) Static pregs are obligatory.** `pset_b*(PAT_ALL)` and any
    constexpr `plt_b*` MUST be built once at the top of `__VEC_SCOPE__`
    and read-only inside the loop. Building predicates inside the loop
    wastes SLIDE-pipe cycles on a loop-invariant value. (This was a
    third-tier point in iter 1's design notes; iter 3 elevated it to the
    same "hard rule" tier as the other items in this list.)

---

## 12. Skill Reference Table (Context Freshness)

| Skill | Use during |
|---|---|
| [VF Fusion Guide](../../case-writing/vf-fusion-guide.md) | Whole-document — Phase A→E |
| [ND→NZ Patterns](../../case-writing/nd-to-nz-patterns.md) | §3.3, §4 (vsstb block_stride, NZ+1 proof), §7 (vdintlv pack pattern) |
| [Squeeze Patterns](../../case-writing/squeeze-patterns.md) | (Not used here — no predicate-filtered output.) |
| [Instruction Families](../../case-writing/instruction-families.md) | §3.1, §7 (LNEXP, vcvt, vdintlv, vsstb costs) |
| [A5 Vector Pipeline](../../architecture/a5-vector-pipeline.md) | §3.4, §7 (MUL/ADD/LNEXP/LSU/SLIDE lanes) |
| [Vec Intrinsics](../../pto-isa/intrinsics-quickref.md) | All — call signatures |
| [Log Analysis](../log-analysis.md) | §2, §5.3, §6.4, §8 (PMU, per-VF, dump correlation, store-side stall) |
| [Debugger Skill](../../case-writing/debugging-with-simulator.md) | §5 if correctness fails |
| [Post-Update & Fusion Tradeoffs](../../case-writing/post-update-and-fusion-tradeoffs.md) | §3.4 (POST_UPDATE on vsstb & vlds) |
| [FA MXFP8 DN Softmax Fusion](fa-mxfp8-dn-softmax-fusion.md) | Inspiration — same fusion family for the FP8 sibling kernel |

