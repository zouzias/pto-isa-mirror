---
type: bug-note
date: 2026-04-19
env:
  source_repo: /home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2
  focus: dispatch_v2/combine_v2 platform split
  chip: [950, 910A3, 910B]
tags: [mc2, dispatch-v2, combine-v2, a2, a3, 950]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_def.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_host/moe_distribute_combine_v2_def.cpp
    reliability: high
---

# dispatch_v2/combine_v2 不同平台要看不同分流轴

## 问题

读 `dispatch_v2` / `combine_v2` 时，容易把 A2、A3、950 都理解成“同一套逻辑 + 少量 ifdef 差异”。

## 根因

kernel 层的分流中心其实完全不同：
- A2 重点在 `CommMode`、`LayeredMode`、`DataplaneMode(AICPU/AIV/MTE)`。
- A3 重点在 `FullMesh` / `Hierarchy` / `HasTp`。
- 950 重点在 `ArchTag == A5` 的独立模板路径和 arch35 专线。

## 正确做法

- 遇到 A2 问题，先顺着 dataplane / layered 路径读。
- 遇到 A3 问题，先判断 full-mesh 还是 hierarchy。
- 遇到 950 问题，先把它当独立 A5 路线，不要套用 A3 心智模型。

## 反例

- 一上来先比较 dtype 列表，以为这就是平台差异核心。
- 把 A2 的 layered/AICPU/AIV 问题按 A3 的 full-mesh 思路分析。

## 参考来源

见 frontmatter `sources`。
