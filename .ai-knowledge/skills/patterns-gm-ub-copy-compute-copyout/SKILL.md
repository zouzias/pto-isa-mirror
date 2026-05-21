---
name: patterns-gm-ub-copy-compute-copyout
description: |
  触发：需要搭建最基础的 GM→UB→计算→GM kernel 骨架时。
  Trigger: Build the basic GM to UB to compute to GM kernel skeleton.
status: draft
---

# Patterns GM UB Copy Compute Copyout

## 触发判断

当任务是最基本的数据搬运+计算 kernel，且还不需要复杂同步或通信时读我。

## 核心方法

1. 明确输入 GM、输出 GM、UB 临时 buffer。
2. 把流程拆成 CopyIn、Compute、CopyOut 三段。
3. 先保证地址、容量、对齐正确，再谈重叠优化。

## 可执行清单

- 初始化 GlobalTensor / LocalTensor
- `DataCopy` 到 UB
- 在 UB 上执行基础向量 API
- 结果回写 GM

## 反模式

- 把 GM 地址直接当 UB 算。
- 忽略对齐与 calCount 上限。
- 初版就混入复杂优化，导致问题不可定位。

## 相关 skill

- `ascendc-api-best-practices`
- `patterns-double-buffer`
