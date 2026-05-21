---
name: card-primitive-manual-binding-and-sync
description: |
  触发：查找 PTO manual binding and sync 相关 PTO 原语簇、入口文件、联动页面或常见混淆时。
  Trigger: Lookup the PTO manual binding and sync PTO primitive family, entry files, linked pages, or common confusion.
card_type: primitive
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# PTO manual binding and sync

## hit terms
- `TASSIGN`
- `TSYNC`
- `event`
- `set_flag / wait_flag`
- `fifo`
- `TPUSH / TPOP / TFREE`
- `manual binding`

## canonical intent
- 找 PTO 里的手动资源绑定与同步主线
- 区分事件链式写法与显式 flag/wait 写法
- 找 FIFO/TPipe 的直接落点

## primary anchors
- `include/pto/common/event.hpp`
- `include/pto/common/fifo.hpp`
- `tests/npu/a5/src/st/testcase/tadd/tadd_kernel.cpp`
- `tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp`
- `tests/npu/a5/src/st/testcase/tpushpop_cv/tpushpop_cv_kernel.cpp`

## common confusion
- 不要把 `TASSIGN` 和普通原语并列理解；它更像资源绑定/configuration。
- 不要把 `TSYNC`、事件、flag/wait、FIFO 混成一种同步风格；它们是不同抽象层。
- 不要把 FIFO 问题直接路由到最小 `tadd` 示例。

## fallback links
- 最小同步示例 → `cards/examples/test-tadd-a5-a2a3.md`
- FIFO 示例 → `cards/examples/test-tpushpop-cv.md`
- ISA 映射 → `wiki/topics/pto-isa-mapping.md`
