---
name: card-example-test-tgather-a5-comm
description: |
  触发：定位 test-tgather-a5-comm testcase、代表源码、主题作用或常见混淆时。
  Trigger: Locate the test-tgather-a5-comm testcase, representative files, scope, or common confusion.
card_type: example
version: repo-current
chip: [950 A5]
source: example
last_updated: 2026-04-20
---

# PTO test tgather (A5 comm)

## hit terms
- `tgather`
- `TGATHER`
- `comm testcase`
- `A5 comm`

## canonical intent
- 找 PTO 通信原语 `TGATHER` 的 testcase 入口
- 确认 `TGATHER` 在 comm 测试目录中的落点

## primary anchors
- `tests/npu/a5/comm/st/testcase/tgather/tgather_kernel.cpp`

## common confusion
- 不要把它和 TopK/vector 路线里的 gather 混；这里是 comm 目录下的 `TGATHER` testcase。
- 不要把它当 compute/comm decouple 模式总览；这里只是 testcase 入口。

## fallback links
- comm primitives → `cards/primitives/comm-primitives.md`
- testcase 总表 → `cards/tables/pto-test-entry-map.md`
