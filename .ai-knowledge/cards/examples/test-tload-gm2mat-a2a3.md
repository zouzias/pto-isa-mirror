---
name: card-example-test-tload-gm2mat-a2a3
description: |
  触发：定位 test-tload-gm2mat-a2a3 testcase、代表源码、主题作用或常见混淆时。
  Trigger: Locate the test-tload-gm2mat-a2a3 testcase, representative files, scope, or common confusion.
card_type: example
version: repo-current
chip: [910A3]
source: example
last_updated: 2026-04-20
---

# PTO test tload_gm2mat (A2A3)

## hit terms
- `tload_gm2mat`
- `TLOAD layout testcase`
- `ND->NZ`
- `DN->ZN`
- `A2A3 TLOAD`

## canonical intent
- 找 `TLOAD` 与 layout/Tile 组合的 testcase
- 观察 A2A3 上 `ND/DN/NZ` 不同路径如何组织

## primary anchors
- `tests/npu/a2a3/src/st/testcase/tload_gm2mat/tload_gm2mat_kernel.cpp`

## common confusion
- 不要把它和 `tload_mx_NZ` 混；后者偏 A5 MX/NZ。
- 不要把它和 `tstore_mat2gm` 混；本例是 load 方向。

## fallback links
- load/store 原语簇 → `cards/primitives/load-store.md`
- testcase 总表 → `cards/tables/pto-test-entry-map.md`
