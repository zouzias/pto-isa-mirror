---
name: card-example-test-tquant-a5-a2a3
description: |
  触发：定位 test-tquant-a5-a2a3 testcase、代表源码、主题作用或常见混淆时。
  Trigger: Locate the test-tquant-a5-a2a3 testcase, representative files, scope, or common confusion.
card_type: example
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# PTO test tquant (A5 / A2A3)

## hit terms
- `tquant`
- `TQUANT`
- `MXFP8`
- `INT8_ASYM`
- `INT8_SYM`
- `quant testcase`

## canonical intent
- 找 PTO 量化路径 testcase
- 对比 A5 与 A2A3 的量化覆盖面差异

## primary anchors
- `tests/npu/a5/src/st/testcase/tquant/main.cpp`
- `tests/npu/a5/src/st/testcase/tquant/tquant_kernel.cpp`
- `tests/npu/a2a3/src/st/testcase/tquant/tquant_kernel.cpp`

## common confusion
- 不要把它和 `tload_mx_NZ` 混；后者偏 layout/load，`tquant` 偏量化语义。
- 不要把 A5 与 A2A3 的量化能力默认等同；A5 覆盖更丰富。

## fallback links
- quant path → `cards/tables/pto-primitives-map.md`
- testcase 总表 → `cards/tables/pto-test-entry-map.md`
