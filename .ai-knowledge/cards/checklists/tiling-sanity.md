---
name: card-checklist-tiling-sanity
description: |
  触发：在设计或复查 tiling 时进行快速自查。
  Trigger: Run a quick sanity checklist for tiling design and review.
card_type: checklist
version: generic
chip: [950 A5, 910A3]
source: note
last_updated: 2026-04-19
---

# Tiling Sanity Checklist

- tile 大小是否经过 UB 容量核算。
- L1 / L0A / L0B / L0C / FBuffer 预算是否和目标平台匹配。
- A2A3 与 A5 的 `Vec / Acc / Scaling` 容量差异是否已经考虑。
- 主路径与尾块路径是否都存在。
- 多核切分是否覆盖全量且无重叠遗漏。
- queue / buffer 深度是否与流水策略一致。
- 输入输出地址与步长是否满足对齐要求。
- shape、dtype、tile bytes 是否在 host 与 kernel 间一致。

## related
- `cards/tables/chip-memory-specs.md`
