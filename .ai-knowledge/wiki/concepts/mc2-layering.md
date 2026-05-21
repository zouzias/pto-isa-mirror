---
page_type: concept
title: mc2-layering
status: draft
version: [ops-transformer mc2 mainline]
chip: [950, 910A3, 910B]
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/README.md
    query: "mc2 dispatch README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/op_api/aclnn_moe_distribute_dispatch.cpp
    query: "mc2 dispatch aclnn"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/op_host/moe_distribute_dispatch_def.cpp
    query: "mc2 dispatch op host"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch.cpp
    query: "mc2 dispatch kernel"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - ../topics/mc2-overview.md
  - pipeline-sync.md
---

# MC2 Layering

## 定位

本页单独总结 `ops-transformer/mc2` 主线代码的分层结构，避免这部分内容长期堆在 `mc2-overview` 里。

## 核心心智模型

MC2 不是“一个 kernel 目录”，而是一条完整的产品化实现链。读 MC2 时，应该默认把一个算子拆成以下几层：

1. README / docs：定义功能、产品支持、公式、参数与约束。
2. op_api：定义 aclnn 入口、前置校验、平台差异和 optional 输入处理。
3. op_host：定义注册、输入输出矩阵、属性矩阵、AICore 配置、HcclGroup。
4. op_kernel：定义模板分发、tiling key 路径、最终 device 执行。
5. common：提供公共 tiling、平台信息、日志和 aclnn 辅助函数。

## 为什么这套分层重要

- 如果只看 README，只能知道“算子宣称做什么”，不知道入口责任和平台分支。
- 如果只看 op_kernel，会误把很多入口约束、shape 代数和版本边界当成“外部噪声”。
- 如果只看 op_api，又会低估 device 侧模板分支的复杂度。

## 各层的典型复杂度中心

- README 层：平台支持、通信域限制、dtype 支持矩阵、是否废弃。
- op_api 层：shape 检查、dtype 组合、group 长度、optional 输入、空 tensor 路径。
- op_host 层：输入输出/属性矩阵、不同平台的配置分叉、HcclGroup 声明。
- op_kernel 层：架构分派、tiling key、量化路径、通信模式与执行模板。

## 使用方式

面对一个新 MC2 算子时，先用这页固定阅读顺序，再去看具体算子页或 writeback。

## Sources

- `ops-transformer/mc2/moe_distribute_dispatch/README.md`
- `ops-transformer/mc2/moe_distribute_dispatch/op_api/aclnn_moe_distribute_dispatch.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch/op_host/moe_distribute_dispatch_def.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch.cpp`
