# pto.settf32mode


`pto.settf32mode` 属于[同步与配置](../../sync-and-config_zh.md)指令集。

## 概要


配置 FP32 矩阵乘和卷积路径使用的 TF32 变换模式。该指令设置 CCE（Cube Compute Engine）TF32 模式寄存器。

在 A2/A3 上，TF32 不受支持，`pto.settf32mode` 为 no-op；`enable` 标志会被忽略，不建立模式状态。在 A5 上，当 `enable = true` 时，后端配置 CCE 对后续 FP32 矩阵乘和卷积路径使用 TF32 尾数截断；`mode` 字段选择舍入行为。CPU 模拟器上该指令为功能性 no-op，并在软件中保留模式状态。

该指令不直接产生 tensor 算术结果。它更新后续指令会读取的目标模式状态。

## 语法


文本拼写由 PTO ISA 的语法与操作数页面定义。

```text
pto.settf32mode {enable = true, mode = ...}
```

### IR Level 1 (SSA)


```text
pto.settf32mode {enable = true, mode = ...}
```

### IR Level 2 (DPS)


```text
pto.settf32mode ins({enable = true, mode = ...}) outs()
```

## C++ 内建接口


声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <bool isEnable, RoundMode tf32TransMode = RoundMode::CAST_ROUND, typename... WaitEvents>
PTO_INST RecordEvent SETTF32MODE(WaitEvents &... events);
```

## 输入


| 操作数 | 类型 | 说明 |
|--------|------|------|
| `enable` | `bool` | 启用（`true`）或禁用（`false`）TF32 变换模式 |
| `mode` | `RoundMode` | TF32 舍入模式；`CAST_ROUND` 表示 round-to-nearest-even，其他模式保留 |

## 期望输出


该形式主要由排序或配置效果定义，不产生新的 payload tile。

## 副作用


- **A2/A3**：no-op。TF32 不受支持，不影响架构状态。
- **A5**：配置 CCE TF32 模式寄存器，影响后续 FP32 矩阵或卷积计算精度。
- **CPU 模拟器**：更新软件模拟模式状态，不影响 IEEE 754 舍入。

## 约束


!!! warning "约束"
    - 在 A2/A3 上调用 `pto.settf32mode` 合法但无效果。
    - 在 A5 上，`enable` 必须为 `true` 或 `false`，`mode` 必须是受支持的 `RoundMode`。
    - 该指令具有控制状态副作用，应当相对于依赖它的计算指令正确排序。

## 示例


```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_enable_tf32() {
  SETTF32MODE<true, RoundMode::CAST_ROUND>();
}
```

## 另请参阅


- 指令集总览：[同步与配置](../../sync-and-config_zh.md)
- 前一个操作：[pto.sethf32mode](./sethf32mode_zh.md)
- 后一个操作：[pto.setfmatrix](./setfmatrix_zh.md)

## Summary
本节给出该指令/主题的核心语义与使用定位，和英文章节保持一致。

## Syntax
本节列出语法形态（SSA / DPS / Assembly），用于与英文页逐项对照。

## C++ Intrinsic
本节给出 C++ 内建接口入口与参数语义说明。

## Inputs
本节定义输入操作数角色、数据来源与有效区域要求。

## Expected Outputs
本节定义输出结果及其在有效区域内的语义保证。

## Side Effects
本节说明除结果写回外是否存在额外可观察副作用。

## Constraints
本节列出类型、布局、shape、valid-region 与 profile 相关约束。

## Cases That Are Not Allowed
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Examples
本节提供 Auto/Manual 及 AS 形式示例，便于中英文对照复现。

## See Also
本节给出上下游指令与相关章节链接。
