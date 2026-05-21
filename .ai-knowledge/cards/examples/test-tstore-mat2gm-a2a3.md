---
name: card-example-test-tstore-mat2gm-a2a3
description: |
  触发：定位 test-tstore-mat2gm-a2a3 testcase、代表源码、主题作用或常见混淆时。
  Trigger: Locate the test-tstore-mat2gm-a2a3 testcase, representative files, scope, or common confusion.
card_type: example
version: repo-current
chip: [910A3]
source: example
last_updated: 2026-04-20
---

# PTO test tstore_mat2gm (A2A3)

## hit terms
- `tstore_mat2gm`
- `TSTORE layout testcase`
- `TLOAD -> TSTORE`
- `A2A3 TSTORE`

## canonical intent
- 找 `TSTORE` 与 layout 回写路径的 testcase
- 观察事件链式 `TLOAD -> TSTORE` 写法

## primary anchors
- `tests/npu/a2a3/src/st/testcase/tstore_mat2gm/tstore_mat2gm_kernel.cpp`

## common confusion
- 不要把它和 `tload_gm2mat` 混；本例是 store 方向。
- 不要把它当同步模式主入口；同步问题优先回 manual-binding-and-sync card。

## fallback links
- load/store 原语簇 → `cards/primitives/load-store.md`
- 手动绑定/同步 → `cards/primitives/manual-binding-and-sync.md`
- testcase 总表 → `cards/tables/pto-test-entry-map.md`
