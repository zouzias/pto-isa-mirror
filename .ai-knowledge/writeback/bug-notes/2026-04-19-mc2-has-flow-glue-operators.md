---
type: bug-note
date: 2026-04-19
env:
  cann: ops-transformer mc2 mainline
  chip: 950 / 910A3 / 910B
  mode: setup teardown and route-level readthrough
tags: [mc2, setup, teardown, routing, product-path]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_setup/op_api/aclnn_moe_distribute_combine_setup.cpp
    query: "mc2 moe distribute combine setup aclnn"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_teardown/op_api/aclnn_moe_distribute_dispatch_teardown.cpp
    query: "mc2 moe distribute dispatch teardown aclnn"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_teardown/op_kernel/moe_distribute_dispatch_teardown.cpp
    query: "mc2 moe distribute dispatch teardown kernel"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/attention_to_ffn/README.md
    query: "mc2 attention to ffn README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/ffn_to_attention/README.md
    query: "mc2 ffn to attention README"
    reliability: medium
---

# MC2 里有一类算子是流程配套件，不是主融合算子本体

## 问题

阅读 MC2 时，容易把所有目录都当成“核心融合算子”，从而对 setup/teardown/route 类目录的定位产生偏差。

## 根因

像 `MoeDistributeCombineSetup`、`MoeDistributeDispatchTeardown`、`AttentionToFFN`、`FFNToAttention` 这类算子，更多承担流程拼装、节点间路由、命令信息准备/回收等职责。它们的重要性不在于单独公式多复杂，而在于它们把主流程前后打通。

## 正确做法

看到 `setup` / `teardown` / 节点路由类算子时，先从“它在整条 MC2 流程里补的是哪一段”来理解，再去看平台限制、输出 size 计算、量化路径是否仍然保留。

## 反例

把这类算子也当成和 `matmul_all_reduce` 同一层级的“主融合核心”，只盯 kernel 公式，不看它在整条流程中的连接作用。
