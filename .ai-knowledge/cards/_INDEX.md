# Cards Index

## api
- [datacopy](api/datacopy.md) — DataCopy 签名、对齐、容量边界与常见坑。
- [cast](api/cast.md) — Cast 支持类型、常见误区与精度注意项。

## error-codes
- [aclnn-161xxx](error-codes/aclnn-161xxx.md) — 参数类错误的典型症状、根因与修复方向。

## checklists
- [tiling-sanity](checklists/tiling-sanity.md) — tiling 设计前后的快速自查项。

## tables
- [chip-specs](tables/chip-specs.md) — 芯片能力占位表与待补齐字段。
- [chip-memory-specs](tables/chip-memory-specs.md) — A2A3 / A5 在 PTO TileType 可见内存空间上的容量速查表。
- [cann-version-matrix](tables/cann-version-matrix.md) — CANN 版本差异占位表与更新规则。
- [pto-primitives-map](tables/pto-primitives-map.md) — PTO 原语族到 docs/include/kernels/tests 的最小映射表。
- [pto-kernel-pattern-map](tables/pto-kernel-pattern-map.md) — PTO kernel 模式到代表源码的映射表。
- [pto-test-entry-map](tables/pto-test-entry-map.md) — PTO testcase 到主题与推荐用途的映射表。

## primitives
- [load-store](primitives/load-store.md) — PTO 数据搬运与 layout/load-store 主线。
- [manual-binding-and-sync](primitives/manual-binding-and-sync.md) — PTO 里的 TASSIGN、事件、flag/wait 与 FIFO 心智模型。
- [vec-elementwise-and-reduce](primitives/vec-elementwise-and-reduce.md) — PTO vector 逐元素、归约、排序与 gather 主线。
- [matmul-family](primitives/matmul-family.md) — PTO 矩阵主算子、提取前处理与 A5 MX 扩展。
- [comm-primitives](primitives/comm-primitives.md) — PTO 通信原语与 comm 类型入口。

## patterns
- [gemm-pipeline](patterns/gemm-pipeline.md) — PTO 标准 cube/gemm 流水骨架。
- [topk-vector-pipeline](patterns/topk-vector-pipeline.md) — TopK 类排序/merge/gather 的 vector 流水。
- [flashattention-stages](patterns/flashattention-stages.md) — FlashAttention 的多阶段 PTO 协同模式。
- [compute-comm-decouple](patterns/compute-comm-decouple.md) — compute / comm 解耦与流式协同模式。
- [a5-mx-simt-sync](patterns/a5-mx-simt-sync.md) — A5 上 MX、SIMT 与同步策略的特化模式。

## examples
- [test-tadd-a5-a2a3](examples/test-tadd-a5-a2a3.md) — PTO 最小可运行示例与 A5/A2A3 同步风格对照。
- [test-tpushpop-cv](examples/test-tpushpop-cv.md) — Cube -> Vec FIFO 协同样例。
- [test-tpushpop-vc-a5](examples/test-tpushpop-vc-a5.md) — Vec -> Cube FIFO 协同与 dequant 进阶样例。
- [test-tquant-a5-a2a3](examples/test-tquant-a5-a2a3.md) — A5/A2A3 量化路径样例。
- [test-ttrans](examples/test-ttrans.md) — PTO 基础转置 testcase 的 host+golden 闭环入口。
- [test-ttrans-conv-a5](examples/test-ttrans-conv-a5.md) — 卷积布局变换 PTO 样例。
- [test-tload-mx-nz-a5](examples/test-tload-mx-nz-a5.md) — A5 上 MX/NZ 载入与 layout 路径样例。
- [test-tload-gm2mat-a2a3](examples/test-tload-gm2mat-a2a3.md) — A2A3 上 GM->Mat 的 TLOAD layout 示例。
- [test-tstore-mat2gm-a2a3](examples/test-tstore-mat2gm-a2a3.md) — A2A3 上 Mat->GM 的 TSTORE layout 示例。
- [test-tmatmul-a5-or-a2a3](examples/test-tmatmul-a5-or-a2a3.md) — PTO 矩阵主算子的测试闭环入口。
- [test-tgather-a5-comm](examples/test-tgather-a5-comm.md) — PTO 通信原语 TGATHER 的测试入口。
