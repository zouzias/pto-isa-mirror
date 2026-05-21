---
type: bug-note
date: 2026-04-19
env:
  source_repo: /home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2
  focus: dispatch_v2/combine_v2 tiling
  chip: [950, 910A3, 910B]
tags: [mc2, dispatch-v2, combine-v2, tiling, base-helper-tiling]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_tiling_helper.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_tiling_helper.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_v2_tiling.cpp
    reliability: high
---

# dispatch_v2/combine_v2 的 host 决策链要按 base → helper → tiling 来读

## 问题

读 v2 的 dispatch/combine 时，很容易直接跳进 kernel，以为真正的选择都在模板实例化里。

## 根因

很多关键决策在 kernel 之前就做完了：
- base 层先按平台改写 groupTp / performanceInfo，并设置 HCCL server type。
- helper 层先把 shape/dtype/format 和 quantMode 对应关系卡死。
- tiling 主逻辑才在这些合法输入上决定 fullmesh / layered / shared expert / quant 等策略。

## 正确做法

- 先看 base：平台有没有先裁路径。
- 再看 helper：输入世界是否合法，特别是 scales/dynamicScales/assistInfo 的维度规则。
- 最后看 tiling：合法输入最终被送到哪条组织方式。

## 反例

- 不看 host/base/helper，直接在 kernel 模板里找“为什么没走这条路径”。
- 把 quantMode 引起的 shape 变化误当成 kernel 内部细节。

## 参考来源

见 frontmatter `sources`。
