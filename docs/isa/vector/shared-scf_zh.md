# 向量路径中的共享结构化控制流

PTO 向量程序依赖结构化控制流而不是隐式 launch 魔法。这里记录的是 `scf` 如何围绕 `pto.v*` 区域表达循环、分支和 loop-carried state。

## 用途

- 重复向量块上的计数循环
- 向量尾部处理
- 条件执行
- 跨迭代携带谓词、offset 或对齐状态

## 约束

!!! warning "约束"
    - 控制流应保持结构化
    - 影响向量合法性或可见状态的 loop-carried 值必须显式出现
    - `vecscope` 周围的控制结构不能被模糊成未说明的 runtime 行为

## 相关页面

- [向量流水同步](./pipeline-sync_zh.md)
- [共享标量算术](./shared-arith_zh.md)
