# Writeback Index

## templates
- [bug-note](templates/bug-note.md) — 问题、根因、修复、反例模板。
- [recipe](templates/recipe.md) — 可复用做法模板。
- [decision-record](templates/decision-record.md) — 决策原因模板。
- [pitfall](templates/pitfall.md) — 易错点模板。

## bug-notes
- [datacopy-alignment-first](bug-notes/2026-04-19-datacopy-alignment-first.md) — DataCopy 异常先查对齐与 buffer 预算。
- [mc2-shmem-read-path](bug-notes/2026-04-19-mc2-shmem-read-path.md) — 读 MC2 shmem 代码至少要覆盖 README、host tiling、kernel entry 三层。
- [mc2-mainline-read-path](bug-notes/2026-04-19-mc2-mainline-read-path.md) — 读 MC2 主线代码至少要覆盖 README、op_api、op_host、op_kernel 四层。
- [mc2-is-multi-family-product-surface](bug-notes/2026-04-19-mc2-is-multi-family-product-surface.md) — MC2 主线应先按家族理解，而不是把所有算子混成一类。
- [mc2-readme-before-kernel-details](bug-notes/2026-04-19-mc2-readme-before-kernel-details.md) — 先看 README 中的平台/通信域/废弃约束，再决定是否深入 kernel。
- [mc2-versioning-and-flow-glue-must-be-read-together](bug-notes/2026-04-19-mc2-versioning-and-flow-glue-must-be-read-together.md) — 读 MC2 主流程时，版本演进和流程配套层必须一起看。
- [mc2-v2-v3-are-interface-refactors](bug-notes/2026-04-19-mc2-v2-v3-are-interface-refactors.md) — V2/V3 主要是接口重构，不是算法家族分裂。
- [mc2-v2-and-postprocess-expand-whole-stack](bug-notes/2026-04-19-mc2-v2-and-postprocess-expand-whole-stack.md) — V2 加上 postprocess 会扩展整栈，不是单点修改。
- [mc2-has-flow-glue-operators](bug-notes/2026-04-19-mc2-has-flow-glue-operators.md) — MC2 存在 setup/teardown/route 配套流程算子，阅读时不能遗漏。
- [mc2-complexity-center-varies-by-family](bug-notes/2026-04-19-mc2-complexity-center-varies-by-family.md) — 不同 MC2 家族的复杂度中心不同，阅读顺序要随之切换。
- [mc2-shape-algebra-before-kernel](bug-notes/2026-04-19-mc2-shape-algebra-before-kernel.md) — 读 MC2 算子前先在 host 层把 shape 代数理顺，再决定是否深入 kernel。
- [mc2-thin-entry-flow-ops](bug-notes/2026-04-19-mc2-thin-entry-flow-ops.md) — 薄入口流程算子应先看流水职责，不要默认深挖 kernel。
- [mc2-collective-is-not-just-prefix](bug-notes/2026-04-19-mc2-collective-is-not-just-prefix.md) — 不同 collective 前缀对应不同的通信结果语义，不要混读。
- [mc2-reducescatter-is-output-slicing](bug-notes/2026-04-19-mc2-reducescatter-is-output-slicing.md) — ReduceScatter 更像结果回切，不是输入补全或布局重排。
- [mc2-dispatch-combine-closure](bug-notes/2026-04-19-mc2-dispatch-combine-closure.md) — Dispatch/Combine 应按闭环协议理解，不要拆成独立算子。
- [mc2-platform-split-centers](bug-notes/2026-04-19-mc2-platform-split-centers.md) — dispatch_v2/combine_v2 在 A2/A3/950 要看不同分流轴。
- [mc2-base-helper-tiling-chain](bug-notes/2026-04-19-mc2-base-helper-tiling-chain.md) — host 决策链要按 base→helper→tiling 读取。
- [mc2-a5-design-mapping](bug-notes/2026-04-19-mc2-a5-design-mapping.md) — A5 设计文档里的关键词都能在 arch35 实现里找到落点。
- [mc2-latest-entry-is-v4](bug-notes/2026-04-19-mc2-latest-entry-is-v4.md) — 主知识层默认应收敛到 DispatchV4/CombineV4。

## recipes
- [bootstrap-structure-first](recipes/2026-04-19-bootstrap-structure-first.md) — 先写稳定结构，再逐步补事实细节。
- [read-mc2-by-operator-families](recipes/2026-04-19-read-mc2-by-operator-families.md) — 先按算子家族建立 MC2 功能地图，再细读单算子四层。
- [pto-route-primitives-patterns-tests](recipes/2026-04-19-pto-route-primitives-patterns-tests.md) — 先建 PTO 总骨架，再按原语/模式/示例逐步纳管。

## decision-records
- [minimum-viable-layers-first](decision-records/2026-04-19-minimum-viable-layers-first.md) — 先补最小可用层，再补高精度内容。

## pitfalls
- [lifecycle-before-precision](pitfalls/2026-04-19-lifecycle-before-precision.md) — 先查生命周期，不先把异常归因成精度问题。
