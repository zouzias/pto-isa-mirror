---
name: patterns-tiling-2d-split
description: |
  触发：需要按二维网格切分数据，规划 tile 行列、核间分工与尾块处理时。
  Trigger: Design 2D tiling layouts, tile grids, per-core work split, and tail handling.
status: draft
---

# Patterns Tiling 2D Split

## 触发判断

当问题是“二维数据怎么切 tile”“多核怎么按行列分工”“尾块怎么处理”时读我。

## 核心方法

1. 先定义全局网格，再定义单核负责的 tile 区间。
2. tile 设计同时受 UB 容量、数据对齐、核数约束。
3. 单独设计尾块路径，不和主路径混写。

## 可执行清单

- 明确 tileRows / tileCols / tilesPerCore
- 区分主块与尾块
- 给出每核起止索引公式
- 检查每 tile 的 bytes 是否落在 UB 预算内

## 反模式

- 只写主路径，不写尾块。
- 直接按经验拍 tile 大小，不核算 UB。
- 把调度与计算逻辑缠在一起。

## 相关 skill

- `ascendc-tiling-design`
