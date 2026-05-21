---
type: bug-note
date: 2026-04-19
env:
  source_repo: /home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2
  focus: latest dispatch/combine interface
  chip: [950PR, 950DT, 910A3, 910B]
tags: [mc2, dispatch-v4, combine-v4, latest]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV4.md
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV4.md
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v4.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_api/aclnn_moe_distribute_combine_v4.cpp
    reliability: high
---

# 最新版主入口应统一到 DispatchV4 / CombineV4，但实现内核仍复用 V2 体系

## 问题

如果继续把 V2/V3/V4 平铺展开，很容易让知识层入口越来越散，用户也不清楚默认该看哪一版。

## 根因

当前对外最新版已经是 `DispatchV4 + CombineV4`，但 op_api 里能看到它们仍然复用 V2/base/inner 体系。也就是说，接口版本在往前演进，但实现家族没有完全重写。

## 正确做法

- 在主知识层里，默认把 `DispatchV4 + CombineV4` 作为最新版入口。
- 把 V2/V3 仅保留为“为什么现在接口长这样”的演进背景。
- 遇到实现细节时，明确说明：最新版接口之下仍复用 V2 inner/base 路线。

## 反例

- 一上来就让用户在 V2/V3/V4 之间自己挑入口。
- 把“接口最新版”误说成“实现完全换代”。

## 参考来源

见 frontmatter `sources`。
