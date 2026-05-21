---
type: bug-note
date: 2026-04-19
env:
  source_repo: /home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2
  focus: MC2 operator reading
  chip: [Ascend 950, 910A3/910B]
tags: [mc2, moe, flow-glue, thin-entry, read-path]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_update_expert/README.md
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_update_expert/op_api/aclnn_moe_update_expert.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_update_expert/op_host/moe_update_expert_def.cpp
    reliability: high
---

# MC2 中存在“薄入口流程算子”，不要默认所有算子都要深挖 kernel

## 问题

读 MC2 时很容易形成一个固定动作：README → op_api → op_host → op_kernel 一路深挖。但这会把所有算子都当成“通算融合主复杂度入口”，阅读成本会迅速失控。

## 根因

MC2 里有一类算子主要承担流程配套职责，而不是主通算复杂度本体。`MoeUpdateExpert` 就是代表：README 已经说明它负责逻辑专家到物理实例的映射、剪枝与与 Dispatch/Combine 的固定配套顺序；而 op_api 基本只做非空检查后转给 inner，op_host 也主要声明 IO/属性和基础平台配置。

## 正确做法

- 先判断算子属于“主复杂度算子”还是“流程配套算子”。
- 若是 `MoeUpdateExpert` 这类薄入口流程算子，优先记录它在整条流水中的职责、调用顺序、关键属性一致性约束。
- 只有当问题确实落到 inner 实现、性能瓶颈或具体行为异常时，再继续深挖 inner/op_kernel。

## 反例

- 看到名字属于 MC2，就默认必须一路深读到 kernel。
- 把流程配套算子和 `MatmulReduceScatterV2` 这类主复杂度算子用同一阅读粒度处理。

## 参考来源

见 frontmatter `sources`。
