---
name: card-example-test-tload-mx-nz-a5
description: |
  触发：定位 test-tload-mx-nz-a5 testcase、代表源码、主题作用或常见混淆时。
  Trigger: Locate the test-tload-mx-nz-a5 testcase, representative files, scope, or common confusion.
card_type: example
version: repo-current
chip: [950 A5]
source: example
last_updated: 2026-04-20
---

# PTO test tload_mx_NZ (A5)

## hit terms
- `tload_mx_NZ`
- `MX/NZ`
- `A5 TLOAD`
- `layout testcase`
- `MX load`

## canonical intent
- 找 A5 上 `TLOAD + MX/NZ layout` 的 testcase
- 区分它与 A2A3 普通 load/store layout 路线

## primary anchors
- `tests/npu/a5/src/st/testcase/tload_mx_NZ/tload_mx_NZ_kernel.cpp`

## common confusion
- 不要把它和 `tload_gm2mat` 混；这里偏 A5 MX/NZ。
- 不要把它当量化 testcase；这里核心是 load + layout。

## fallback links
- load/store 原语簇 → `cards/primitives/load-store.md`
- testcase 总表 → `cards/tables/pto-test-entry-map.md`
