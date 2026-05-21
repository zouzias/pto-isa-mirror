---
name: card-primitive-matmul-family
description: |
  触发：查找 PTO matmul family 相关 PTO 原语簇、入口文件、联动页面或常见混淆时。
  Trigger: Lookup the PTO matmul family PTO primitive family, entry files, linked pages, or common confusion.
card_type: primitive
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# PTO matmul family

## hit terms
- `TMATMUL`
- `TMATMUL_ACC`
- `TMATMUL_MX`
- `TEXTRACT`
- `TIMG2COL`
- `matmul family`
- `MX matmul`
- `flashattention matmul`

## canonical intent
- 找 PTO 的 cube/matmul 主线
- 把 `TMATMUL*`、`TEXTRACT`、`TIMG2COL` 路由到 gemm/conv/flashattention/A5 MX 相关模式

## primary anchors
- `include/pto/common/pto_instr.hpp`
- `kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp`
- `kernels/manual/a2a3/conv2d_forward/conv2d_forward_kernel.cpp`
- `kernels/manual/a5/matmul_mxfp4_performance/mxmatmul_performance_kernel.cpp`
- `tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp`

## capacity linkage
- matmul 路线最直接受 `Mat / L1`、`Left / L0A`、`Right / L0B`、`Acc / L0C` 的容量约束。
- A5 与 A2A3 的关键差异点之一是 `Acc / L0C`：A5 更大，因此某些 accumulator/buffering 方案更容易成立。
- A5 的 `ScaleLeft / ScaleRight` 也让 MX 路线有了额外预算空间。

## common confusion
- 不要把 `TEXTRACT`/`TIMG2COL` 当独立主题；它们通常是 matmul 数据前处理的一部分。
- 不要把 `TMATMUL_MX` 当成通用 matmul；它更偏 A5 特化。
- 不要把 matmul 路线与 vector-only 路线混用。

## fallback links
- gemm 模式 → `cards/patterns/gemm-pipeline.md`
- A5 特化 → `cards/patterns/a5-mx-simt-sync.md`
- testcase → `cards/examples/test-tmatmul-a5-or-a2a3.md`
- 容量规格 → `cards/tables/chip-memory-specs.md`
