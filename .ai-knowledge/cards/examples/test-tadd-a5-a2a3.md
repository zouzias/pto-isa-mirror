---
name: card-example-test-tadd-a5-a2a3
description: |
  触发：定位 test-tadd-a5-a2a3 testcase、代表源码、主题作用或常见混淆时。
  Trigger: Locate the test-tadd-a5-a2a3 testcase, representative files, scope, or common confusion.
card_type: example
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# PTO test tadd (A5 / A2A3)

## hit terms
- `tadd`
- `TADD testcase`
- `minimal PTO example`
- `TLOAD + TADD + TSTORE`
- `A5 tadd`
- `A2A3 tadd`

## canonical intent
- 找 PTO 最小读算写 testcase
- 对比 A5 与 A2A3 的同步风格

## primary anchors
- `tests/npu/a5/src/st/testcase/tadd/main.cpp`
- `tests/npu/a5/src/st/testcase/tadd/tadd_kernel.cpp`
- `tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp`

## common confusion
- 不要把它当 layout 或 matmul 入口；它是最小 elementwise 入口。
- 不要把它和 `tpushpop_*` 混；后者是 FIFO/cube-vec 协同样例。

## fallback links
- 同步与资源绑定问题 → `cards/primitives/manual-binding-and-sync.md`
- testcase 总表 → `cards/tables/pto-test-entry-map.md`
