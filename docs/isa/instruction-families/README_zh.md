# 指令族


本章描述 PTO ISA 的指令族（Instruction Set）——共享约束和行为的指令分组。每个族定义了该族所有指令共同遵循的规则。

## 本章内容


- [指令族总览](README_zh.md) — 完整导航地图和族规范模板
- [Tile 指令族](tile-families_zh.md) — Tile 指令集下的 8 个指令族（逐元素、归约、布局等）
- [Vector 指令族](../vector/README_zh.md) — Vector 指令集下的 9 个指令族
- [标量与控制指令族](scalar-and-control-families_zh.md) — 标量、控制和配置的 6 个指令族
- [通信指令族](communication-families_zh.md) — 跨 NPU collective、点到点交换和通知
- [系统调度指令族](system-scheduling-families_zh.md) — TPipe/TMPipe 生产者-消费者协议和资源生命周期

## 指令集与指令族的关系


- **指令集（Instruction Set）** 按功能角色分类指令（Tile / Vector / Scalar / Communication / System Scheduling）
- **族（Instruction Set）** 共享约束、行为模式和规范语言；同一族的指令共享家族概览页中的共同约束

## 每个族必须定义的内容


1. **Mechanism** — 族的用途说明
2. **Shared Operand Model** — 共同的操作数模型和交互方式
3. **Common 副作用** — 所有族内操作共享的副作用
4. **Shared Constraints** — 适用于全族的合法性规则
5. **Cases That Are Not Allowed** — 全族禁止的条件
6. **Target-Profile Narrowing** — A2/A3 和 A5 的差异
7. **Operation List** — 指向各 per-op 页面的链接

## 章节定位


本章属于手册第 7 章（指令集）的一部分。族文档是 per-op 页面的上一层抽象，同一族的指令共享家族概览页中的共同约束。


# Instruction Set Contracts
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Overview
本节提供该主题的总览说明与阅读入口，与英文章节语义保持一致。

## Why These Instruction Sets Exist
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Tile ISA (`pto.t*`)
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Vector ISA (`pto.v*`)
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Scalar/Control ISA (`pto.*`)
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Communication ISA
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### System Scheduling ISA
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## What An Instruction Set Contract Must State
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Navigation Map
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Normative Language
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## See Also
本节给出相关章节和上下游链接。
