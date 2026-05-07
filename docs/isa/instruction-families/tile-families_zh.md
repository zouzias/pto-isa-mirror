# Tile 指令族


Tile 指令族定义 `pto.t*` 各组操作共享的机制、操作数模型、约束和 profile 缩窄关系，然后再进入具体 per-op 页面。

## 指令族概览


| 指令族 | 说明 |
| --- | --- |
| 同步与配置 | `tassign`、`tsync`、`talias`、`sethf32mode`、`settf32mode`、`setfmatrix` 等 |
| 逐元素 Tile-Tile | `tadd`、`tmul`、`tpow`、`tcmp`、`tcvt` 等 |
| Tile-标量与立即数 | `tadds`、`taxpy`、`tmuls`、`tpows`、`tmins` 等 |
| 归约与扩展 | `trowsum`、`tcolmax`、`trowexpand` 等 |
| 内存与数据搬运 | `tload`、`tstore`、`mgather`、`mscatter` |
| 矩阵与矩阵-向量 | `tgemv`、`tmatmul` 及其变体 |
| 布局与重排 | `tmov`、`ttrans`、`tconcat`、`tpack`、`textract`、`tinsert` 等 |
| 不规则与复杂 | `tmrgsort`、`tquant`、`tdequant`、`trandom`、`thistogram`、`tprint` 等 |

## 共享操作数模型


- tile / tile buffer
- 标量修饰符与立即数
- GM 视图
- 事件链与等待事件

## 共享约束


- valid region 交互必须明确
- layout 与 role 限制必须明确
- target-profile 缩窄必须单独标明
- 不允许的组合必须在族级说明中出现

## 域外语义


对于逐元素类族，目标 tile 的 valid region 决定迭代域。源 tile 在域外坐标上的读取如果没有单独定义，应按 implementation-defined 或 unsupported 处理，而不是默认视为有意义数据。

## 不允许的情形


!!! danger "不允许的情形"
    - 把目标 profile 的实现便利当成 PTO 通用规则
    - 把未定义的域外行为写成稳定语义
    - 依赖隐式广播、隐式 reshape 或隐式同步

## 相关页面


- [Tile 指令集](tile-families_zh.md)
- [Tile 参考入口](../tile/README_zh.md)


# Tile Instruction Set
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Instruction Overview
本节提供该主题的总览说明与阅读入口，与英文章节语义保持一致。

### Data Flow
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Instruction Classes
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Inputs
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Expected Outputs
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Side Effects
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Shared Constraints
本节列出关键约束与不允许组合，确保使用方式可验证。

## Valid Region Model
本节说明抽象模型与关键对象关系，用于统一术语与行为理解。

## Constraints
本节列出关键约束与不允许组合，确保使用方式可验证。

## Cases That Are Not Allowed
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Saturating Variants
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Type Support by Profile
本节定义类型/布局/格式规则与合法性要求。

## Syntax
本节给出语法与操作数约定，便于与英文规范逐项对照。

### Assembly Form (PTO-AS)
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### SSA Form (AS Level 1)
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### DPS Form (AS Level 2)
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## C++ Intrinsic
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Navigation
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## See Also
本节给出相关章节和上下游链接。
