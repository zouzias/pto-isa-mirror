---
name: card-primitive-load-store
description: |
  触发：查找 PTO load / store 相关 PTO 原语簇、入口文件、联动页面或常见混淆时。
  Trigger: Lookup the PTO load / store PTO primitive family, entry files, linked pages, or common confusion.
card_type: primitive
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# PTO load / store

## hit terms
- `TLOAD`
- `TSTORE`
- `GM->Mat`
- `Mat->GM`
- `ND/NZ/DN`
- `load store layout`
- `tload_gm2mat`
- `tstore_mat2gm`
- `tload_mx_NZ`

## canonical intent
- 找 PTO 的数据搬运主线
- 找 `TLOAD/TSTORE` 与 `GlobalTensor/Tile/Layout` 的组合落点
- 找最适合验证 load/store layout 的 testcase

## primary anchors
- `include/pto/common/pto_instr.hpp`
- `tests/npu/a2a3/src/st/testcase/tload_gm2mat/tload_gm2mat_kernel.cpp`
- `tests/npu/a2a3/src/st/testcase/tstore_mat2gm/tstore_mat2gm_kernel.cpp`
- `tests/npu/a5/src/st/testcase/tload_mx_NZ/tload_mx_NZ_kernel.cpp`

## capacity linkage
- `TLOAD/TSTORE` 的 layout 路径最终仍受 `Vec / UB`、`Mat / L1` 等容量与对齐约束影响。
- `TASSIGN_zh.md` 给出的 `TileType -> 内存空间 -> 容量` 是这类路径的直接硬边界。
- 当 `validRow/validCol`、boxed/fractal 或 ping-pong 地址规划看起来“不自然”时，先回容量表再判断是否只是布局问题。

## common confusion
- 不要把 `TLOAD/TSTORE` 当纯 memcpy；它们通常和 layout、Tile 形状、validRow/validCol 一起决定语义。
- 不要把 A2A3 的 `tload_gm2mat` 和 A5 的 `tload_mx_NZ` 混成同一入口。
- 不要把 load/store 问题直接路由到 matmul 或 comm 页面。

## fallback links
- 原语总映射 → `cards/tables/pto-primitives-map.md`
- testcase 总表 → `cards/tables/pto-test-entry-map.md`
- include 分层 → `wiki/concepts/pto-layering.md`
- 容量规格 → `cards/tables/chip-memory-specs.md`
