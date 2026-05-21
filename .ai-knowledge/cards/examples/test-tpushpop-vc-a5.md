---
name: card-example-test-tpushpop-vc-a5
description: |
  触发：定位 test-tpushpop-vc-a5 testcase、代表源码、主题作用或常见混淆时。
  Trigger: Locate the test-tpushpop-vc-a5 testcase, representative files, scope, or common confusion.
card_type: example
version: repo-current
chip: [950 A5]
source: example
last_updated: 2026-04-20
---

# PTO test tpushpop_vc (A5)

## hit terms
- `tpushpop_vc`
- `Vec -> Cube`
- `TDEQUANT`
- `TPUSH`
- `TPOP`
- `TMATMUL_ACC`

## canonical intent
- 找 Vec -> Cube 的 FIFO testcase
- 找 dequant 后再喂给 cube/matmul 的真实例子

## primary anchors
- `tests/npu/a5/src/st/testcase/tpushpop_vc/tpushpop_vc_kernel.cpp`

## common confusion
- 不要把它和 `tpushpop_cv` 混；这里是 Vec 预处理后喂给 Cube。
- 不要把它当纯量化 testcase；这里还有 FIFO 与 matmul feed。
- 不要把它当最小案例；它是高组合度样例。

## fallback links
- quant path → `cards/primitives/vec-elementwise-and-reduce.md`
- matmul family → `cards/primitives/matmul-family.md`
- testcase 总表 → `cards/tables/pto-test-entry-map.md`
