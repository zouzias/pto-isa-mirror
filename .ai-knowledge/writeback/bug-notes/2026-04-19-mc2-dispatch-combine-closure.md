---
type: bug-note
date: 2026-04-19
env:
  source_repo: /home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2
  focus: MC2 dispatch/combine
  chip: [950, 910A3, 910B, A2]
tags: [mc2, moe, dispatch, combine, closure]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine/README.md
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine/op_api/aclnn_moe_distribute_combine.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/README.md
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/README.md
    reliability: high
---

# Dispatch/Combine 要按“闭环协议”来读，不要按两个独立算子来读

## 问题

直觉上容易把 `MoeDistributeDispatch` 理解成“把 token 发出去”，把 `MoeDistributeCombine` 理解成“最后收回来做聚合”。这样会把很多中间张量误读成业务张量。

## 根因

README 已经明确：`Combine` 是“按 Dispatch 收集数据的路径原路返还”。因此 `expandIdx`、`epRecvCounts`、`tpRecvCounts`、`expandScales` 这些张量，本质上不是业务可解释结果，而是 dispatch→combine 闭环中的协议字段。

V2 更进一步：把 `expandIdx` 替换为 `assistInfoForCombine`，并把共享专家输入、通信算法等边界显式接口化，说明闭环协议被前移到接口层了。

## 正确做法

- 读 dispatch/combine 时，把它们当成一条发散-返还闭环。
- 遇到中间计数/辅助张量时，先问“这是给 combine 回放路径用的，还是给业务逻辑用的”。
- 对 V2，优先理解它如何把闭环协议显式化，再去看新增参数细节。

## 反例

- 单独研究 `expandIdx` 或 `epRecvCounts` 的元素值，并把它们当成稳定业务语义。
- 只看 dispatch 或只看 combine，就试图还原整个 MoE 路由过程。

## 参考来源

见 frontmatter `sources`。
