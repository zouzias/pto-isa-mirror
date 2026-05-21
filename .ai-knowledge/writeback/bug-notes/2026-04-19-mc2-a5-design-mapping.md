---
type: bug-note
date: 2026-04-19
env:
  source_repo: /home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2
  focus: A5 design mapping
  chip: [950PR, 950DT]
tags: [mc2, a5, arch35, design, double-buffer, cc u]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/docs/MoeDistributeDispatch-Combine算子设计介绍.md
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_arch35.h
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_arch35.h
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/arch35/moe_distribute_dispatch_tiling_arch35.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_host/op_tiling/arch35/moe_distribute_combine_tiling_arch35.cpp
    reliability: high
---

# A5 上设计文档里的关键术语都能在 arch35 实现里找到具体落点

## 问题

只看设计文档时，容易把“设备侧自治”“双缓冲”“批量通信”这些词当成架构口号；只看 arch35 代码时，又容易只看到一堆常量和 HCCL 调用。

## 根因

设计文档和实现文件之间缺少显式映射。实际上：
- 文档里的双缓冲，对应 arch35 里的 `BUFFER_NUM = 2`、status 区和前后半 window 切换。
- 文档里的设备侧自治，对应 base 层选择 `CCU` server type 与 kernel 中的 `Hccl<HCCL_SERVER_TYPE_CCU>`。
- 文档里的按 rank 批量发送，对应 `perRankDataSize_`、send/recv offset 以及 `AlltoAllvWrite`。
- 文档里的规模约束，对应 arch35 tiling 常量和校验逻辑。

## 正确做法

读 A5/arch35 时，把设计术语和代码落点配对着看：
- 设计说“同步”，找 status/flag 切换。
- 设计说“批量发送”，找 window 分区和 AlltoAllvWrite。
- 设计说“范围限制”，找 arch35 tiling 常量。

## 反例

- 把设计文档看成纯理论说明，不回到 arch35 常量和 kernel 流程验证。
- 把 arch35 常量看成孤立 magic numbers，而不回溯到设计约束来源。

## 参考来源

见 frontmatter `sources`。
