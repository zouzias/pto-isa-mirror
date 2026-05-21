---
name: card-example-test-tpushpop-cv
description: |
  触发：定位 test-tpushpop-cv testcase、代表源码、主题作用或常见混淆时。
  Trigger: Locate the test-tpushpop-cv testcase, representative files, scope, or common confusion.
card_type: example
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# PTO test tpushpop_cv

## hit terms
- `tpushpop_cv`
- `TPUSH`
- `TPOP`
- `TFREE`
- `Cube -> Vec`
- `TPipe`
- `FIFO`

## canonical intent
- 找 Cube -> Vec 的 FIFO/TPipe testcase
- 找 `TPUSH/TPOP/TFREE` 在真实 testcase 中的落点

## primary anchors
- `tests/npu/a5/src/st/testcase/tpushpop_cv/tpushpop_cv_kernel.cpp`
- `tests/npu/a2a3/src/st/testcase/tpushpop_cv/tpushpop_cv_kernel.cpp`

## common confusion
- 不要把它和 `tpushpop_vc` 混；后者是 Vec -> Cube。
- 不要把它当最小同步示例；最小同步应回 `tadd`。
- 不要把它当 comm 原语示例；这里核心是 FIFO/cube-vec 协同。

## fallback links
- manual binding / sync → `cards/primitives/manual-binding-and-sync.md`
- testcase 总表 → `cards/tables/pto-test-entry-map.md`
