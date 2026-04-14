# A5 TopK — local UB radix variant (`topk_ub`)

This directory is **not** the upstream `kernels/manual/a5/topk` scaffold from `cann/pto-isa`. It holds a **local implementation** (single GM load, full-width `THISTOGRAM` / `TGATHER`, `TCONCAT`, etc.) so you can `git pull` upstream without overwriting your experiment.

- Upstream reference layout: [`../topk/README.md`](../topk/README.md)
- Build/run: same as `topk`, but the executable is **`topk_ub`**:

```bash
cd kernels/manual/a5/topk_ub
bash run.sh -r sim -v Ascend910_9599
```

- Data: `scripts/gen_data.py` writes `input/` and `output/` under **this** directory.

## `include/` A5 header changes

Local edits under `include/pto/npu/a5/*.hpp` are exported to **`include_hpp_changes.patch`** (from repo root: `git diff include/`). To re-apply after pulling upstream:

```bash
cd /path/to/pto-isa
git apply kernels/manual/a5/topk_ub/include_hpp_changes.patch
```

Regenerate the patch after you change headers: `git diff --no-color include/ > kernels/manual/a5/topk_ub/include_hpp_changes.patch`.
