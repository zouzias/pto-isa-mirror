---
name: card-example-test-ttrans-conv-a5
description: |
  触发：定位 test-ttrans-conv-a5 testcase、代表源码、主题作用或常见混淆时。
  Trigger: Locate the test-ttrans-conv-a5 testcase, representative files, scope, or common confusion.
card_type: example
version: repo-current
chip: [950 A5]
source: example
last_updated: 2026-04-20
---

# PTO test ttrans_conv (A5)

## hit terms
- `ttrans_conv`
- `ConvTile`
- `TTRANS`
- `NCHW`
- `NC1HWC0`
- `FRACTAL_Z`

## canonical intent
- 找卷积布局变换相关的 PTO testcase
- 找 `ConvTile + TTRANS` 的直接样例

## primary anchors
- `tests/npu/a5/src/st/testcase/ttrans_conv/main.cpp`
- `tests/npu/a5/src/st/testcase/ttrans_conv/ttrans_conv_kernel.cpp`

## common confusion
- 不要把它和基础 `ttrans` 混；这里偏卷积布局与 `ConvTile`。
- 不要把它路由到普通 load/store；这里核心是布局变换。

## fallback links
- 基础转置 → `cards/examples/test-ttrans.md`
- testcase 总表 → `cards/tables/pto-test-entry-map.md`
