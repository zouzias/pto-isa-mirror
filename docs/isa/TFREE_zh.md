# TFREE

## 指令示意图

![TFREE tile operation](../figures/isa/TFREE.svg)

## 简介

将当前占用的 pipe 或 FIFO 槽位释放回生产者。

## 数学语义

语义随指令而变化。 除非另有说明，行为都按目标 valid region 定义。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```text
%dst = pto.tfree ...
```

### AS Level 2（DPS）

```text
pto.tfree ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`.

## 约束

数据类型、layout、location 和 shape 的进一步限制以对应 backend 的合法性检查为准。

## 示例

具体的 Auto / Manual 使用方式见 `docs/isa/` 下的相关指令页。
