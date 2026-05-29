# Case study: A5 TopK histogram tiling at 2048

This wiki page documents raising **Phase1/3 `THISTOGRAM` column tiling** from **256** to **2048**: motivation, separate tiling knobs, a subtle correctness bug, simulator debugging, performance gains, and regression checks. Code: [`kernels/manual/a5/topk/`](../../../kernels/manual/a5/topk/) (`draft.cpp`).

---

## 1. Context

| Item | Value |
|------|-------|
| Platform | Ascend A5 (sim `Ascend950PR_9599`) |
| Algorithm | Radix-select TopK on 2-byte keys |
| Shape | N=8192, TopK=512 |
| GM tiles | `kTileCols = 2048` → 4 tiles |
| Check | Host multiset compare on `keys[out[i]]` |

See [`topk_ub`](../../../kernels/manual/a5/topk_ub/) for an all-in-UB reference; this kernel streams from GM.

---

## 2. Three tiling constants (do not conflate)

| Constant | Value | Phases | Role |
|----------|-------|--------|------|
| `kTileCols` | **2048** | Outer GM loop | Columns per GM tile |
| `kHistChunkCols` | **2048** (default) | Phase1, Phase3 | Columns per `TLOAD` + `THISTOGRAM` |
| `kChunkCols` | **256** | Phase5 | `TGATHER` / `TCONCAT_IMPL` slices |

**Why 2048 for histogram but 256 for gather?**

- Histogram: one wide `THISTOGRAM` per GM tile cuts slice count (8→1 per tile) and MTE2 traffic.
- Gather/concat: Phase5 UB layout and VF templates were tuned at **256** columns; do not widen Phase5 without a dedicated validation pass.

---

## 3. Correctness: not “2048 breaks hist”, but broken `TCMPS`

### Symptoms (before fix)

- `kHistChunkCols=2048` **FAIL**; `256` **PASS**
- Wrong Phase4: `lsbWinnerBin=16`, `packedThr=0xF010`
- Fixed: `lsbWinnerBin=90`, `packedThr=0xF05a`

### Ruled out

- `remainK` corruption at `0x25000` (held **12** until Phase4 `TOR`)
- Extra `pipe_barrier(PIPE_ALL)` before Phase5
- `THISTOGRAM` width unsupported (post-rebase **PASS** + ST `1×2048`)

### Root cause

Old `TCmps.hpp` used scalar `T src1Value = *src1` in tile-tile compares. VF lowering compared **`chist` vs index tile**, not **`chist` vs remainK`**.

**Fix (upstream):** load full `src1` tile, e.g. `vlds(src1Reg, src1, 0, BRC_B32)` in `TCmpsTileB32`.

After `git rebase` onto `origin/master`, keep **upstream** `TCmps.hpp`; do not resurrect local scalar/debug variants from stash.

---

## 4. Debugging workflow

- Dumps: `ub.rd_log`, `ub.wr_log`, `rvec_pv` under `CAMODEL_LOG_PATH`
- Scripts in `kernels/manual/a5/topk/scripts/` (phase1/3/4 checks, `compare_phase4_ub_pv.py`)
- Archive PASS-256 vs FAIL-2048 dumps for before/after review
- Ensure CMake **`BEFORE`** repo `include/` so CANN does not shadow fixed headers

---

## 5. Performance (8K, seed `1241200609`)

| Metric | 2048 hist (current) | hist256gm archive |
|--------|--------------------:|------------------:|
| kernel ticks | **52,207** | 87,221 (~**−40%**) |
| MTE2 busy | 36,277 | ~66k |

Phase5 still dominates rvec (~86%). See `perf/phase_perf_8k_seed1241200609.json` and `parse_phase_perf.py --hist-chunk-cols 2048`.

---

## 6. Regression

- TopK: 3 seeds + `--const 0x1234` (see kernel `README_zh.md`)
- ST: `tests/npu/a5/src/st/testcase/thistogram/` includes **1×2048** cases

---

## 7. Takeaways

1. Tune GM tile, hist chunk, and gather chunk **independently**.
2. Validate **tile-tile compare/select** semantics before blaming tiling or barriers.
3. Profile MTE2 and hist slice count when widening hist; Phase5 may be unchanged.
4. Rebase upstream PTO headers; archive old perf JSON for PR evidence.
5. Cover both ST (single op) and manual kernel (full pipeline).

---

## Links

- [`kernels/manual/a5/topk/README.md`](../../../kernels/manual/a5/topk/README.md)
- [`docs/isa/TCMPS.md`](../../isa/TCMPS.md)
- [`docs/coding/opt.md`](../opt.md)
