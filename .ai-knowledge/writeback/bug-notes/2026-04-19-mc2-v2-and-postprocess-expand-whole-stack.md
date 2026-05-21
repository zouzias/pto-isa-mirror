---
type: bug-note
date: 2026-04-19
env:
  cann: ops-transformer mc2 mainline
  chip: 950 / 910A3 / 910B
  mode: deeper op-layer mc2 readthrough batch 3
tags: [mc2, v2, deprecation, op_host]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.cpp
    query: "mc2 matmul reduce scatter v2 aclnn"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_reduce_scatter_v2/op_host/matmul_reduce_scatter_v2_def.cpp
    query: "mc2 matmul reduce scatter v2 op host"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_all_reduce_add_rms_norm/op_api/aclnn_matmul_all_reduce_add_rms_norm.cpp
    query: "mc2 matmul all reduce add rms norm aclnn"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_all_reduce_add_rms_norm/op_host/matmul_all_reduce_add_rms_norm_def.cpp
    query: "mc2 matmul all reduce add rms norm op host"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_update_expert/op_api/aclnn_moe_update_expert.cpp
    query: "mc2 moe update expert aclnn"
    reliability: medium
---

# MC2 的 V2 和复合后处理算子，复杂度常在整条链同时抬升

## 问题

看到 V2、AddRmsNorm、Setup/Teardown 这类名字时，容易以为只是“在原算子上加一点点功能”。

## 根因

主线代码表明，很多这类算子不是局部增量，而是整条链同时变化：README 的能力矩阵更复杂，op_api 的 case 区分更细，op_host 的 dtype/attr 组合更多，kernel 的 tiling key 路径也更多。反过来，有些复合算子虽然技术复杂，却已经在 op_api 层提示迁移或废弃。

## 正确做法

遇到 V2 或复合后处理算子时，不要只盯 README 或 kernel，要同时判断三件事：
1. 它是不是整条链都在扩张；
2. 它是不是已经进入产品迁移期；
3. 它的复杂度中心更偏 op_api/op_host 还是 kernel。

## 反例

看到名字里有 `V2` 或 `AddRmsNorm`，就默认只是旧算子的轻量包装。
