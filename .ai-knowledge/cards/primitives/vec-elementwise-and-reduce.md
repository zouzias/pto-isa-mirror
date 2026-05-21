---
name: card-primitive-vec-elementwise-and-reduce
description: |
  触发：查找 PTO vec elementwise and reduce 相关 PTO 原语簇、入口文件、联动页面或常见混淆时。
  Trigger: Lookup the PTO vec elementwise and reduce PTO primitive family, entry files, linked pages, or common confusion.
card_type: primitive
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# PTO vec elementwise and reduce

## hit terms
- `TADD`
- `TROWMAX`
- `TROWSUM`
- `TEXP`
- `TCVT`
- `TSORT32`
- `TMRGSORT`
- `TGATHER`
- `softmax`
- `topk`

## canonical intent
- 找 PTO 的 vector-only 主线
- 区分逐元素/归约路径与排序/gather 路径
- 把单条 vec 原语路由到 TopK 或 Softmax 相关模式

## primary anchors
- `include/pto/common/pto_instr.hpp`
- `kernels/manual/a2a3/topk/topk_kernel.cpp`
- `kernels/manual/common/flash_atten/pto_macro_fa_softmax.hpp`
- `tests/npu/a5/src/st/testcase/tadd/tadd_kernel.cpp`
- `tests/npu/a5/comm/st/testcase/tgather/tgather_kernel.cpp`

## common confusion
- 不要把 `TGATHER` 自动归到通信；在某些上下文里它也命中 vector 数据处理主线。
- 不要把 vector-only 路线和 cube/matmul 路线混成一个模式。
- 不要把 `TADD` 这类最小原语直接当成 TopK/Softmax 模式本身。

## fallback links
- TopK 模式 → `cards/patterns/topk-vector-pipeline.md`
- FlashAttention stages → `cards/patterns/flashattention-stages.md`
- testcase 总表 → `cards/tables/pto-test-entry-map.md`
