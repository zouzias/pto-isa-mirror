# A5 TopK — local UB radix variant (`topk_ub`)

This directory is **not** the upstream `kernels/manual/a5/topk` scaffold from `cann/pto-isa`. It holds a **local implementation** (single GM load, full-width `THISTOGRAM` / `TGATHER`, `TCONCAT`, etc.) so you can `git pull` upstream without overwriting your experiment.

- Upstream reference layout: [`../topk/README.md`](../topk/README.md)
- Build/run: same as `topk`, but the executable is **`topk_ub`**:

```bash
cd kernels/manual/a5/topk_ub
bash run.sh -r sim -v Ascend910_9599
```

- Data: `scripts/gen_data.py` writes `input/` and `output/` under **this** directory.

## Low-range ST (keys in 0~10)

To reproduce tie-heavy TopK behavior quickly:

```bash
cd kernels/manual/a5/topk_ub
bash scripts/st_range_0_10.sh            # default seed: 20260511
bash scripts/st_range_0_10.sh 123456789  # custom seed
```

This ST does:

- `python3 scripts/gen_data.py --min-key 0 --max-key 10 --seed <seed>`
- `python3 scripts/radix_topk_golden_stats.py`

`scripts/gen_data.py` now supports:

- `--min-key` (inclusive, uint16 range)
- `--max-key` (inclusive, uint16 range)

## RemainK corner case (`winner == 0`)

For MSB winner selection, `remain_k` must match radix golden:

- `remain_k = (N - TopK) - C[winner-1]`
- define `C[-1] = 0`

In kernel code, `winner-1` is clamped before gather for memory safety, so when `winner == 0` we must explicitly force `cw = 0` before `TSUB`; otherwise it incorrectly uses `C[0]` and may underflow in low-range/tie-heavy inputs.

## `include/` A5 header changes

Local edits under `include/pto/npu/a5/*.hpp` are exported to **`include_hpp_changes.patch`** (from repo root: `git diff include/`). To re-apply after pulling upstream:

```bash
cd /path/to/pto-isa
git apply kernels/manual/a5/topk_ub/include_hpp_changes.patch
```

Regenerate the patch after you change headers: `git diff --no-color include/ > kernels/manual/a5/topk_ub/include_hpp_changes.patch`.
