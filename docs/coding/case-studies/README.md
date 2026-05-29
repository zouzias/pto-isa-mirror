# Operator case studies (Wiki)

End-to-end, reproducible tuning and debugging notes tied to `kernels/manual/` implementations.

## Case list

| Topic | Doc | Code |
|-------|-----|------|
| A5 TopK: histogram tiling 256 → 2048 | [a5-topk-hist-tiling-2048.md](a5-topk-hist-tiling-2048.md) | [`kernels/manual/a5/topk/`](../../../kernels/manual/a5/topk/) |

## Adding a new case

1. Keep a runnable `run.sh`, README, and optional `perf/` archive under `kernels/manual/`.
2. Add `*_zh.md` here (English `*.md` recommended).
3. Link from [`docs/menu_ops_development.md`](../../menu_ops_development.md) and [`docs/coding/opt.md`](../opt.md) §7.
