# pto.tfree


pto.tfree 属于系统调度（System Scheduling）指令集，用于释放由 	alloc 申请的资源或槽位。

## 摘要


- 与 	alloc 成对使用
- 用于生命周期回收，避免资源泄漏
- 主要用于 NPU 端系统级资源管理路径

## 使用建议


- 建议与分配点保持明确配对
- 在异常或提前返回路径中确保可达

## 说明


详细语义、目标约束与示例请参考英文页：
[TFREE.md](./TFREE.md)。

# TFREE
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Tile Operation Diagram
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Introduction
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Math Interpretation
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Assembly Syntax
本节列出语法形态（SSA / DPS / Assembly），用于与英文页逐项对照。

### IR Level 1 (SSA)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### IR Level 2 (DPS)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## C++ Intrinsic
本节给出 C++ 内建接口入口与参数语义说明。

## Constraints
本节列出类型、布局、shape、valid-region 与 profile 相关约束。

## Examples
本节提供 Auto/Manual 及 AS 形式示例，便于中英文对照复现。
