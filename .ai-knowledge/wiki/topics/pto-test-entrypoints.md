---
page_type: topic
title: pto-test-entrypoints
status: draft
version: [pto-isa repository current layout]
chip: [A2, A3, 910A3, 910B, A5, 950]
sources:
  - origin: code:tests/README_zh.md
    query: "tests overview"
    source_type: example
    reliability: high
    ingested_at: 2026-04-19
  - origin: code:tests/npu/a5/src/st/testcase/tadd/main.cpp
    query: "pto minimal testcase"
    source_type: example
    reliability: high
    ingested_at: 2026-04-19
  - origin: code:tests/npu/a5/src/st/testcase/tpushpop_cv/tpushpop_cv_kernel.cpp
    query: "pto fifo testcase"
    source_type: example
    reliability: high
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - pto-overview.md
  - ../../cards/tables/pto-test-entry-map.md
---

# PTO Test Entrypoints

## canonical scope

本页是 PTO testcase 问题的 canonical router，用来把“tests 里该看哪个例子”归一化到有限的 testcase 簇上。本页不展开教程式说明，只负责把查询词稳定路由到合适的 testcase。

## hit terms

- `tadd`
- `tpushpop`
- `tquant`
- `ttrans`
- `ttrans_conv`
- `tload_gm2mat`
- `tstore_mat2gm`
- `tmatmul`
- `tgather`
- `golden.bin`
- `gen_data.py`

## normalized routing

- 最小读算写 testcase → `tadd`
- FIFO / cube-vec / vec-cube 协同 → `tpushpop_cv` / `tpushpop_vc`
- 量化路径 → `tquant`
- 基础转置 → `ttrans`
- 卷积布局变换 → `ttrans_conv`
- load/store layout → `tload_gm2mat` / `tstore_mat2gm` / `tload_mx_NZ`
- 矩阵主算子 → `tmatmul`
- 通信原语 → `tgather`

## primary anchors

- `tests/README_zh.md`
- `cards/tables/pto-test-entry-map.md`
- `tests/npu/**/testcase/*`

## common confusion

- 不要把 `ttrans` 和 `ttrans_conv` 混成同一入口；前者偏基础转置，后者偏卷积布局。
- 不要把 `tload_gm2mat` 和 `tload_mx_NZ` 混成同一类 layout 问题；前者偏 A2A3 ND/DN/NZ，后者偏 A5 MX/NZ。
- 不要把 testcase 目录当 pattern 页；模式问题应转到 `pto-kernel-patterns`。

## escalation path

- 需要 testcase 总表 → `cards/tables/pto-test-entry-map.md`
- 需要对应原语簇 → `cards/tables/pto-primitives-map.md`
- 需要 host+golden 闭环细节 → 回 `main.cpp` / `gen_data.py`
- 需要 kernel 组合细节 → 回 `*_kernel.cpp`

## current status

当前 PTO tests 已经足够支撑最小示例、layout、量化、矩阵主算子、通信原语这五类查询。后续新增 testcase 内容应优先挂到现有 testcase 簇，而不是平铺目录。

## Sources

- `tests/README_zh.md`
- `tests/npu/a5/src/st/testcase/tadd/main.cpp`
- `tests/npu/a5/src/st/testcase/tpushpop_cv/tpushpop_cv_kernel.cpp`

## Related

- `wiki/topics/pto-overview.md`
- `cards/tables/pto-test-entry-map.md`
