---
name: card-table-pto-test-entry-map
description: |
  触发：查找 PTO Test Entry Map 对应的映射表、规格表或入口对照时。
  Trigger: Lookup the PTO Test Entry Map mapping table, spec table, or entry cross-reference.
card_type: table
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# PTO Test Entry Map

| Test entry | Platform | Theme | Primary PTO focus | Recommended level |
|---|---|---|---|---|
| `tests/npu/a5/src/st/testcase/tadd/*` | A5 | 最小读算写示例 | `TLOAD + TADD + TSTORE` | entry |
| `tests/npu/a2a3/src/st/testcase/tadd/*` | A2/A3 | 最小读算写示例 | 事件链式 PTO 同步 | entry |
| `tests/npu/a5/src/st/testcase/tpushpop_cv/*` | A5 | Cube -> Vec FIFO 协同 | `TPUSH/TPOP/TFREE` | intermediate |
| `tests/npu/a5/src/st/testcase/tpushpop_vc/*` | A5 | Vec -> Cube FIFO 协同 | dequant + cube feed | advanced |
| `tests/npu/a5/src/st/testcase/tquant/*` | A5 | 量化路径 | `TQUANT`, layout/format | intermediate |
| `tests/npu/a2a3/src/st/testcase/tquant/*` | A2/A3 | 量化路径 | INT8 asym/sym | intermediate |
| `tests/npu/a5/src/st/testcase/ttrans/*` | A5 | 基础转置 testcase | `TTRANS` host+golden 闭环 | intermediate |
| `tests/npu/a5/src/st/testcase/ttrans_conv/*` | A5 | 卷积布局变换 | `TTRANS`, `ConvTile` | intermediate |
| `tests/npu/a5/src/st/testcase/tload_mx_NZ/*` | A5 | MX/NZ 载入 | layout + load/store | intermediate |
| `tests/npu/a2a3/src/st/testcase/tload_gm2mat/*` | A2/A3 | GM->Mat load | `TLOAD` + layout | intermediate |
| `tests/npu/a2a3/src/st/testcase/tstore_mat2gm/*` | A2/A3 | Mat->GM store | `TSTORE` + layout | intermediate |
| `tests/npu/a5/src/st/testcase/tmatmul/*` | A5 | 矩阵主算子 | `TMATMUL` family | advanced |
| `tests/npu/a5/comm/st/testcase/tgather/tgather_kernel.cpp` | A5 comm | 通信原语 | `TGATHER` | advanced |

## usage
- 入门先从 `tadd` 开始。
- 看 FIFO/cube-vec 协同优先 `tpushpop_cv` / `tpushpop_vc`。
- 看量化与格式路径优先 `tquant` 与 `tload_mx_NZ`。
- 看基础转置优先 `ttrans`，看卷积布局变换再进 `ttrans_conv`。
- 看 load/store layout 细节优先 `tload_gm2mat` / `tstore_mat2gm`。
- 看矩阵主算子与通信原语再进入 `tmatmul` / `tgather`。

## related
- `wiki/topics/pto-test-entrypoints.md`
- `cards/tables/pto-primitives-map.md`
