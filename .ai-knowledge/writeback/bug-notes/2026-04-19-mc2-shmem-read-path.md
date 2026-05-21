---
type: bug-note
date: 2026-04-19
env:
  cann: 8.5.0 experimental mc2 shmem example
  chip: 910A3
  mode: ops-transformer mc2 dispatch/combine shmem
tags: [mc2, shmem, tiling, integration]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/experimental/mc2/moe_distribute_dispatch_shmem/README.md
    query: "ops-transformer mc2 shmem README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/experimental/mc2/moe_distribute_dispatch_shmem/op_host/op_tiling/moe_distribute_dispatch_shmem_tiling.cpp
    query: "ops-transformer mc2 shmem tiling"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/experimental/mc2/moe_distribute_dispatch_shmem/op_kernel/moe_distribute_dispatch_shmem.cpp
    query: "ops-transformer mc2 shmem kernel entry"
    reliability: medium
---

# MC2 shmem 路径不是单 kernel 问题，而是 host/kernel/上层一起接入

## 问题

初看 MC2 shmem 代码时容易把关注点只放在 kernel 主体，误以为“读一个 kernel 文件就知道这条路径怎么落地”。

## 根因

在 `ops-transformer` 当前实现里，MC2 shmem 路径至少同时涉及 README 中的构建与上层适配、host 侧 tiling 参数检查与 tiling key 生成、kernel 侧模板分支分发，以及 `shmem_space` 这类新增入参的端到端传递。任何一层漏看，都会对整体路径理解失真。

## 正确做法

读 MC2 相关实现时，至少按“README / host tiling / kernel entry”三层各读一份代表文件；先建立接入链，再深入具体 kernel 内部细节。

## 反例

只读 kernel 模板分支，不看 host tiling 和上层 README，就直接总结 MC2 的接入成本、分支复杂度或调试模型。
