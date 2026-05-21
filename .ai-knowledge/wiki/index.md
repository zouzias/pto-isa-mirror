# Wiki Index

## Entity / API
- [DataCopy](api/DataCopy.md) — UB↔GM 数据搬运的综合页，含约束、陷阱与来源入口 (stable)

## Entity / Chip
- [910A3](chips/910A3.md) — A3 / 910A3 在 PTO TileType 可见容量上的轻量规格入口 (draft)
- [950A5](chips/950A5.md) — A5 / 950 在 PTO TileType 可见容量上的轻量规格入口 (draft)

## Concept
- [pipeline-sync](concepts/pipeline-sync.md) — Queue、EnQue/DeQue 与同步语义的统一理解 (stable)
- [mc2-layering](concepts/mc2-layering.md) — MC2 主线代码的 README/op_api/op_host/op_kernel/common 分层模型 (draft)
- [pto-layering](concepts/pto-layering.md) — PTO include 体系的统一入口、公共 API 与后端分派分层 (draft)

## Compare
- [mc2-vs-shmem](compare/mc2-vs-shmem.md) — MC2 与手写 SHMEM 的比较维度与适用场景路由 (draft)
- [mc2-dispatch-combine-closure](compare/mc2-dispatch-combine-closure.md) — Dispatch/Combine 闭环协议问题的路由页 (draft)
- [mc2-dispatch-combine-platform-splits](compare/mc2-dispatch-combine-platform-splits.md) — dispatch_v2/combine_v2 在 A2/A3/950 的平台分流轴 (draft)
- [mc2-dispatch-combine-tiling-decisions](compare/mc2-dispatch-combine-tiling-decisions.md) — base/helper/tiling 三层 host 决策链 (draft)
- [mc2-a5-design-to-implementation](compare/mc2-a5-design-to-implementation.md) — A5 设计术语到 arch35 实现落点映射 (draft)
- [mc2-collective-positioning](compare/mc2-collective-positioning.md) — AllGatherMatmulV2 与 AlltoAllMatmul 的语义定位路由 (draft)
- [mc2-collective-triangle](compare/mc2-collective-triangle.md) — AllGather / AlltoAll / ReduceScatter 的三角定位路由 (draft)
- [mc2-moe-v2-v3-evolution](compare/mc2-moe-v2-v3-evolution.md) — Moe distribute V2/V3 到最新版入口的历史背景页 (draft)

## Topic
- [mc2-overview](topics/mc2-overview.md) — MC2 主主题的 canonical router (draft)
- [mc2-moe-dispatch-combine-latest](topics/mc2-moe-dispatch-combine-latest.md) — Moe distribute 默认主入口 `DispatchV4 + CombineV4` (draft)
- [mc2-grouped-quant-alltoallv-latest](topics/mc2-grouped-quant-alltoallv-latest.md) — grouped / quant / alltoallv 家族主语义簇路由 (draft)
- [pto-overview](topics/pto-overview.md) — PTO 原语、kernel 模式与 testcase 入口的总览页 (stable)
- [pto-isa-mapping](topics/pto-isa-mapping.md) — PTO 原语从 manifest/category 到实现状态、模式和样例的映射页 (draft)
- [pto-kernel-patterns](topics/pto-kernel-patterns.md) — kernels/manual 中稳定 PTO 使用模式的主题页 (draft)
- [pto-test-entrypoints](topics/pto-test-entrypoints.md) — tests/npu 下 PTO testcase 路由页 (draft)
