---
type: pitfall
date: 2026-04-19
env:
  cann: 9.0 beta2
  chip: Atlas 900 A3
tags: [pipeline, queue, lifecycle]
reliability: verified
sources:
  - origin: writeback:session
    query: "pipeline sync issues often look like precision bugs"
    reliability: medium
---

# 把生命周期问题误判成精度问题

## 易错点

结果异常、偶发错误或边界块出错时，容易第一时间怀疑 Cast、算术精度或数据类型转换。

## 为什么容易错

因为表面症状出现在数值层，但真实问题常是 queue 生命周期、buffer 覆盖、EnQue/DeQue/FreeTensor 顺序错误。

## 正确做法

先回到 `pipeline-sync` 模型检查 buffer 所有权与阶段边界，再决定是否继续查精度链路。
