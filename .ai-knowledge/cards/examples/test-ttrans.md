---
name: card-example-test-ttrans
description: |
  触发：定位 test-ttrans testcase、代表源码、主题作用或常见混淆时。
  Trigger: Locate the test-ttrans testcase, representative files, scope, or common confusion.
card_type: example
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# PTO test ttrans

## hit terms
- `ttrans`
- `TTRANS testcase`
- `transpose testcase`
- `golden.bin`
- `gen_data.py`

## canonical intent
- 找 `TTRANS` 的基础 testcase
- 找带 host+golden 闭环的基础转置入口

## primary anchors
- `tests/npu/a5/src/st/testcase/ttrans/main.cpp`

## common confusion
- 不要把它和 `ttrans_conv` 混；后者偏卷积布局变换。
- 不要把它当 matmul 或 load/store 路径；它是基础转置入口。

## fallback links
- testcase 总表 → `cards/tables/pto-test-entry-map.md`
- 卷积布局变换 → `cards/examples/test-ttrans-conv-a5.md`
