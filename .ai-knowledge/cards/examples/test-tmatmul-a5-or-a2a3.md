---
name: card-example-test-tmatmul-a5-or-a2a3
description: |
  触发：定位 test-tmatmul-a5-or-a2a3 testcase、代表源码、主题作用或常见混淆时。
  Trigger: Locate the test-tmatmul-a5-or-a2a3 testcase, representative files, scope, or common confusion.
card_type: example
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# PTO test tmatmul

## hit terms
- `tmatmul`
- `TMATMUL`
- `TMATMUL_ACC`
- `matmul testcase`
- `cube testcase`

## canonical intent
- 找 PTO 矩阵主算子的 testcase
- 找 `TMATMUL` 家族的直接运行入口

## primary anchors
- `tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp`

## common confusion
- 不要把它和 gemm pipeline kernel 样例混；这里是 testcase 入口，不是 kernel pattern 总览。
- 不要把它当 A5 MX 特化例子；MX 特化应回 A5 pattern / MX kernel。

## fallback links
- matmul family → `cards/primitives/matmul-family.md`
- testcase 总表 → `cards/tables/pto-test-entry-map.md`
