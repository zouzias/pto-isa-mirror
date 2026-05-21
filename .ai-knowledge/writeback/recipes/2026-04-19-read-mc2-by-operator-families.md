---
type: recipe
date: 2026-04-19
env:
  cann: ops-transformer mc2 mainline
  chip: 950 / 910A3 / 910B
tags: [mc2, readthrough, operator-families]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_all_reduce/README.md
    query: "mc2 matmul all reduce README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/all_gather_matmul_v2/README.md
    query: "mc2 all gather matmul v2 README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_reduce_scatter_v2/README.md
    query: "mc2 matmul reduce scatter v2 README"
    reliability: medium
---

# 先按算子家族读 MC2，再下沉到单算子四层

## 适用场景

需要系统性读 `ops-transformer/mc2` 整体，而不是只补一个算子页面时。

## 做法

先按算子家族分组：MoE distribute、Matmul+通信融合、Grouped/Quant/AlltoAllV、路由转换/屏障。先读每组 1~3 个代表 README 建立“功能地图”，再对关键算子补 README → op_api → op_host → op_kernel 四层细读。

## 边界

如果一开始就逐文件细读整个 `mc2/`，很容易在大量 op_api/op_host/op_kernel 之间迷失，且不容易形成稳定知识层结构。
