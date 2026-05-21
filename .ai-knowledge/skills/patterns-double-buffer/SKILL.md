---
name: patterns-double-buffer
description: |
  触发：设计双缓冲 / ping-pong pipeline，需要重叠数据搬运与计算时。
  Trigger: Build double-buffer or ping-pong pipelines to overlap data movement and compute.
status: draft
---

# Patterns Double Buffer

## 触发判断

当 kernel 需要隐藏搬运延迟、Queue 深度需要设为 2、以及 CopyIn/Compute/CopyOut 交错时读我。

## 核心方法

1. 输入输出队列都以 depth=2 初始化。
2. 主循环按 CopyIn → Compute → CopyOut 组织。
3. EnQue/DeQue/FreeTensor 必须成对。
4. tile 大小由 tiling 决定，不写死在 pattern 里。

## 可执行清单

- `pipe.InitBuffer(..., 2, <tile_bytes>)`
- `CopyIn(i)` 负责搬运并入队
- `Compute(i)` 负责出队、计算、释放输入、回写输出队列
- `CopyOut(i)` 负责出队并写回 GM

## 反模式

- Queue depth 写 1，退化成单缓冲。
- 忘记 FreeTensor。
- EnQue/DeQue 不成对。

## 相关 skill

- `ascendc-tiling-design`
- `ascendc-api-best-practices`
