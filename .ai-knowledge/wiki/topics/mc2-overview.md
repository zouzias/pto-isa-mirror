---
page_type: topic
title: mc2-overview
status: draft
version: [generic, ops-transformer mc2 mainline]
chip: [950, 910A3, 910B]
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/README.md
    query: "ops-transformer mc2 moe_distribute_dispatch README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/op_api/aclnn_moe_distribute_dispatch.cpp
    query: "ops-transformer mc2 dispatch aclnn entry"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/op_host/moe_distribute_dispatch_def.cpp
    query: "ops-transformer mc2 dispatch op host def"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch.cpp
    query: "ops-transformer mc2 dispatch op kernel"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/common/op_api/mc2_aclnn_util.cpp
    query: "ops-transformer mc2 common aclnn util"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_all_reduce/README.md
    query: "ops-transformer mc2 matmul all reduce README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/all_gather_matmul_v2/README.md
    query: "ops-transformer mc2 all gather matmul v2 README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.cpp
    query: "ops-transformer mc2 matmul reduce scatter v2 aclnn entry"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_reduce_scatter_v2/op_host/matmul_reduce_scatter_v2_def.cpp
    query: "ops-transformer mc2 matmul reduce scatter v2 op host"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_v2.cpp
    query: "ops-transformer mc2 matmul reduce scatter v2 op kernel"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_all_reduce_add_rms_norm/README.md
    query: "ops-transformer mc2 matmul all reduce add rms norm README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/allto_all_matmul/README.md
    query: "ops-transformer mc2 alltoall matmul README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/grouped_mat_mul_all_reduce/README.md
    query: "ops-transformer mc2 grouped matmul all reduce README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_update_expert/README.md
    query: "ops-transformer mc2 moe update expert README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - ../concepts/mc2-layering.md
  - ../compare/mc2-vs-shmem.md
  - mc2-moe-dispatch-combine-latest.md
  - ../compare/mc2-dispatch-combine-closure.md
  - ../compare/mc2-collective-positioning.md
  - ../compare/mc2-collective-triangle.md
  - mc2-grouped-quant-alltoallv-latest.md
  - ../concepts/pipeline-sync.md
---

# MC2 Overview

## canonical scope

本页是 MC2 主主题的 canonical router，用来把 MC2 相关问题归一化到几个稳定子问题：
- 主线分层怎么读
- Moe distribute 默认看哪版
- dispatch/combine 为什么必须按闭环理解
- matmul/collective 家族默认看哪类最新版入口
- grouped / quant / alltoallv 家族如何收敛
- pipeline / lifecycle 问题该落到哪里

本页不替代代码事实，也不再承担“大而全综述”职责。

## hit terms

- `mc2`
- `ops-transformer mc2`
- `moe distribute`
- `dispatch combine`
- `allgather matmul`
- `alltoall matmul`
- `reducescatter`
- `grouped quant alltoallv`
- `mc2 layering`
- `mc2 vs shmem`

## normalized routing

- 问“MC2 总体怎么分层看” → `wiki/concepts/mc2-layering.md`
- 问“MC2 与 SHMEM 的实现路线差异” → `wiki/compare/mc2-vs-shmem.md`
- 问“Moe distribute 现在默认看哪版” → `wiki/topics/mc2-moe-dispatch-combine-latest.md`
- 问“dispatch/combine 为什么必须一起看” → `wiki/compare/mc2-dispatch-combine-closure.md`
- 问“AllGather / AlltoAll / ReduceScatter 的语义位置” → `wiki/compare/mc2-collective-positioning.md` / `wiki/compare/mc2-collective-triangle.md`
- 问“grouped / quant / alltoallv 默认看什么入口” → `wiki/topics/mc2-grouped-quant-alltoallv-latest.md`
- 问“流水、buffer 生命周期、同步语义” → `wiki/concepts/pipeline-sync.md`
- 问 repo 当前实现细节 → 直接回代码，不用本页代替代码事实

## primary anchors

- `ops-transformer/mc2/moe_distribute_dispatch/README.md`
- `ops-transformer/mc2/moe_distribute_dispatch/op_api/aclnn_moe_distribute_dispatch.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch/op_host/moe_distribute_dispatch_def.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch.cpp`
- `ops-transformer/mc2/common/op_api/mc2_aclnn_util.cpp`

## canonical answer units

- **MC2 不是单层实现**：至少要同时持有 README / op_api / op_host / op_kernel / common 五层心智模型
- **Moe distribute 家族默认入口**：`DispatchV4 + CombineV4`
- **dispatch/combine 的理解方式**：闭环协议，不是松散前后处理
- **matmul + collective 家族**：默认按最新版入口收敛，不再平铺基础版与旧版
- **grouped / quant / alltoallv 家族**：默认按主语义簇收敛，不把命名组合当一级入口
- **repo 当前事实**：最终以代码搜索和代码锚点为准

## common confusion

- 不要把 MC2 当成单一算子；它是多家族产品面。
- 不要把 wiki 当 repo 当前现实；当前实现仍必须回代码核实。
- 不要把旧版本页重新拉回主入口；旧版本只保留背景价值。
- 不要把 collective、grouped/quant、dispatch/combine 混成同一维度问题。

## escalation path

- 分层模型 → `wiki/concepts/mc2-layering.md`
- Moe distribute 默认入口 → `wiki/topics/mc2-moe-dispatch-combine-latest.md`
- dispatch/combine 闭环 → `wiki/compare/mc2-dispatch-combine-closure.md`
- AllGather/AlltoAll/ReduceScatter → `wiki/compare/mc2-collective-positioning.md` / `wiki/compare/mc2-collective-triangle.md`
- grouped/quant/alltoallv → `wiki/topics/mc2-grouped-quant-alltoallv-latest.md`
- pipeline/lifecycle → `wiki/concepts/pipeline-sync.md`
- repo 当前实现 → 回代码搜索

## current status

当前知识层已经把 MC2 主线压回“总入口页 + 家族路由页 + 对比页”的结构。后续若继续增补，应优先补 canonical answer unit 或新路由节点，而不是恢复大综述叙事。

## Sources

- `ops-transformer/mc2/moe_distribute_dispatch/README.md`
- `ops-transformer/mc2/moe_distribute_dispatch/op_api/aclnn_moe_distribute_dispatch.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch/op_host/moe_distribute_dispatch_def.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch.cpp`
- `ops-transformer/mc2/common/op_api/mc2_aclnn_util.cpp`
- `ops-transformer/mc2/matmul_all_reduce/README.md`
- `ops-transformer/mc2/all_gather_matmul_v2/README.md`
- `ops-transformer/mc2/matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.cpp`
- `ops-transformer/mc2/matmul_reduce_scatter_v2/op_host/matmul_reduce_scatter_v2_def.cpp`
- `ops-transformer/mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_v2.cpp`
- `ops-transformer/mc2/matmul_all_reduce_add_rms_norm/README.md`
- `ops-transformer/mc2/allto_all_matmul/README.md`
- `ops-transformer/mc2/grouped_mat_mul_all_reduce/README.md`
- `ops-transformer/mc2/moe_update_expert/README.md`

## Related

- `wiki/concepts/mc2-layering.md`
- `wiki/compare/mc2-vs-shmem.md`
- `wiki/topics/mc2-moe-dispatch-combine-latest.md`
- `wiki/compare/mc2-dispatch-combine-closure.md`
- `wiki/compare/mc2-collective-positioning.md`
- `wiki/compare/mc2-collective-triangle.md`
- `wiki/topics/mc2-grouped-quant-alltoallv-latest.md`
- `wiki/concepts/pipeline-sync.md`
