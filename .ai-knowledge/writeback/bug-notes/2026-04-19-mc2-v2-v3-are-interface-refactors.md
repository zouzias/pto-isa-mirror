---
type: bug-note
date: 2026-04-19
env:
  cann: ops-transformer mc2 mainline
  chip: 950 / 910A3 / 910B
  mode: moe distribute version evolution readthrough
tags: [mc2, moe, versioning, context]
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
---

# MC2 的 V2/V3 更像接口边界重构，而不是只加新功能

## 问题

看到 `DispatchV2/V3`、`CombineV2/V3` 时，容易把它理解成“在原算子上继续加量化和参数”。

## 根因

主线 README 表明，这些版本不仅扩功能，还在重定义接口边界：辅助信息更细、成对配套关系更强、通信域上下文被显式引入、某些原始字符串型 group 入参被弱化或替代。

## 正确做法

分析 V2/V3 时，先比较三件事：
1. 与哪个版本的对端算子配套；
2. 辅助信息结构有没有重定义；
3. 通信域信息是靠 group 名字传，还是靠 context/ccl_buffer_size 这类上下文对象传。

## 反例

只把 V2/V3 当成“多支持几个 quantMode”，忽略接口边界和调用责任已经发生迁移。
