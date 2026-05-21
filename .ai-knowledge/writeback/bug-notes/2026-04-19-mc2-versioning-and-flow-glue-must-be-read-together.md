---
type: bug-note
date: 2026-04-19
env:
  cann: ops-transformer mc2 mainline
  chip: 950 / 910A3 / 910B
  mode: versioning and flow-glue synthesis
tags: [mc2, versioning, flow-glue, interface-boundary]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/README.md
    query: "mc2 moe distribute dispatch v2 README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v3/README.md
    query: "mc2 moe distribute dispatch v3 README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/README.md
    query: "mc2 moe distribute combine v2 README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v3/README.md
    query: "mc2 moe distribute combine v3 README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_setup/op_api/aclnn_moe_distribute_combine_setup.cpp
    query: "mc2 moe distribute combine setup aclnn"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_teardown/op_api/aclnn_moe_distribute_dispatch_teardown.cpp
    query: "mc2 moe distribute dispatch teardown aclnn"
    reliability: medium
---

# MC2 的版本演进和流程配套层必须一起看，不能只看主算子名

## 问题

如果只围绕 `dispatch/combine` 主算子名阅读，很容易忽略两个事实：一是 V2/V3 正在重构接口边界，二是 setup/teardown/route 这类流程配套层也在决定真实使用方式。

## 根因

主线 `mc2/` 代码表明，版本演进和流程配套不是边角信息，而是主线的一部分：V2/V3 在改辅助信息、上下文传递和配套关系；setup/teardown 在补齐命令信息准备和回收；route 算子在打通 Attention/FFN 节点。

## 正确做法

分析 MC2 某条流程时，至少同时回答三件事：
1. 这条路径属于哪个版本边界（V1/V2/V3）；
2. 它的对端算子是谁；
3. 它是否依赖 setup/teardown/route 这类流程配套层。

## 反例

只盯 `MoeDistributeDispatch` 或 `MoeDistributeCombine` 的主 kernel，把版本边界和流程配套层都当作可忽略细节。
