---
name: card-primitive-comm-primitives
description: |
  触发：查找 PTO comm primitives 相关 PTO 原语簇、入口文件、联动页面或常见混淆时。
  Trigger: Lookup the PTO comm primitives PTO primitive family, entry files, linked pages, or common confusion.
card_type: primitive
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# PTO comm primitives

## hit terms
- `TPUT`
- `TGET`
- `TNOTIFY`
- `TWAIT`
- `TTEST`
- `TGATHER`
- `TSCATTER`
- `TBROADCAST`
- `TREDUCE`
- `ParallelGroup`
- `Signal`

## canonical intent
- 找 PTO 通信原语的统一入口
- 把通信原语问题路由到 `include/pto/comm`、comm testcase 或 compute/comm 模式

## primary anchors
- `include/pto/comm/pto_comm_inst.hpp`
- `include/pto/comm/comm_types.hpp`
- `kernels/manual/a5/allgather_gemm/allgather_gemm_comm_kernel.cpp`
- `kernels/manual/a2a3/gemm_ar/comm_kernel.cpp`
- `tests/npu/a5/comm/st/testcase/tgather/tgather_kernel.cpp`

## common confusion
- 不要把 `TGATHER` 在所有上下文里都解释成通信；具体要看它命中 comm 目录还是 vector 模式。
- 不要把 comm 原语问题路由到普通 vec/cube 原语页。
- 不要把 compute/comm 解耦问题只看 testcase；通常还需要回 comm kernel。

## fallback links
- comm 模式 → `cards/patterns/compute-comm-decouple.md`
- testcase → `cards/examples/test-tgather-a5-comm.md`
- ISA 映射 → `wiki/topics/pto-isa-mapping.md`
