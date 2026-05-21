---
type: bug-note
date: 2026-04-19
env:
  cann: ops-transformer mc2 mainline
  chip: 950 / 910A3 / 910B
  mode: moe_distribute_dispatch mainline readthrough
tags: [mc2, aclnn, op_host, op_kernel]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/README.md
    query: "ops-transformer mc2 mainline dispatch README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/op_api/aclnn_moe_distribute_dispatch.cpp
    query: "ops-transformer mc2 mainline dispatch aclnn"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/op_host/moe_distribute_dispatch_def.cpp
    query: "ops-transformer mc2 mainline dispatch host def"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch.cpp
    query: "ops-transformer mc2 mainline dispatch kernel"
    reliability: medium
---

# MC2 主线代码阅读不能只看 kernel，要按 README→op_api→op_host→op_kernel 读

## 问题

读 `ops-transformer/mc2` 主线代码时，容易直接跳进 `op_kernel`，把 MC2 理解成“模板分支很多的 kernel 集合”。

## 根因

主线 MC2 的真实结构是分层的：README 定义功能和参数，op_api 定义 aclnn 入口与平台特殊处理，op_host 定义输入输出/属性/AICore 配置和 HcclGroup，op_kernel 才负责模板与架构分派。漏读前几层，会误判入口约束、平台分流和调试路径。

## 正确做法

阅读主线 MC2 算子时，至少按 README → op_api → op_host → op_kernel 四层各读一份代表文件，再去总结功能、限制和分支复杂度。

## 反例

只读 `op_kernel/moe_distribute_dispatch.cpp`，就直接总结 MC2 的接口、平台支持和调试模型。
