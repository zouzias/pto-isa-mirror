---
name: card-table-chip-memory-specs
description: |
  触发：查找 Chip memory specs 对应的映射表、规格表或入口对照时。
  Trigger: Lookup the Chip memory specs mapping table, spec table, or entry cross-reference.
card_type: table
version: repo-current
chip: [950 A5, 910A3]
source: official
last_updated: 2026-04-20
---

# Chip memory specs

## purpose
- 提供 A2A3 / A5 等平台在 PTO `TileType -> 内存空间 -> 容量 -> 对齐` 维度上的硬规格速查。
- 当前内容优先基于 `docs/isa/TASSIGN_zh.md` 的编译时边界检查表，不推测未给出的芯片资源。

## source of truth
- `docs/isa/TASSIGN_zh.md`
- `include/pto/common/buffer_limits.hpp`（若后续需要核源码）

## TileType to memory-space capacity

| TileType | Memory space | A2A3 | A5 | Alignment |
|---|---|---:|---:|---:|
| Vec | UB | 192 KB | 256 KB | 32 B |
| Mat | L1 | 512 KB | 512 KB | 32 B |
| Left | L0A | 64 KB | 64 KB | 32 B |
| Right | L0B | 64 KB | 64 KB | 32 B |
| Acc | L0C | 128 KB | 256 KB | 32 B |
| Bias | Bias | 1 KB | 4 KB | 32 B |
| Scaling | FBuffer | 2 KB | 4 KB | 32 B |
| ScaleLeft | L0A | N/A | 4 KB | 32 B |
| ScaleRight | L0B | N/A | 4 KB | 32 B |

## direct design implications
- **A5 的 UB 大于 A2A3**：`Vec` tile、双缓冲和更大中间块在 A5 上更容易容纳。
- **A5 的 L0C 大于 A2A3**：Accumulator 预算更宽，某些矩阵/累加路径在 A5 上更容易放开。
- **A5 有额外的 ScaleLeft / ScaleRight**：这与 MX / scaling 路线直接相关，A2A3 不应默认套用。
- **L1 在 A2A3 / A5 相同**：`Mat` 主存储层这里没有因为平台切换自动翻倍。

## common confusion
- 这张表是 **PTO 编程可见容量表**，不是完整 marketing 硬件规格表。
- 这张表回答的是 `TileType` 能绑定到多大的片上空间，不等于芯片所有缓存层次的完整清单。
- 目前这张表不回答 AICore/VectorCore/CubeCore 数量，也不回答完整 L2/ICache 规格。

## related
- `cards/primitives/manual-binding-and-sync.md`
- `cards/primitives/load-store.md`
- `wiki/chips/910A3.md`
- `wiki/chips/950A5.md`
