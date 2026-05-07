# pto.setfmatrix


`pto.setfmatrix` 属于[同步与配置](../../sync-and-config_zh.md)指令集。

## 概要


配置 FMATRIX（fast matrix）引擎模式和地址。该指令写入 FMATRIX 控制寄存器，并将 tile 绑定到指定 FMATRIX 槽，供后续矩阵乘或卷积操作使用。

该指令不直接产生 tensor 算术结果。它更新后续矩阵类操作会读取的 tile 到硬件资源绑定。

## 语法


文本拼写由 PTO ISA 的语法与操作数页面定义。

```text
pto.setfmatrix %tile : !pto.tile<...>
```

### IR Level 1 (SSA)


```text
pto.setfmatrix %tile : !pto.tile<...>
```

### IR Level 2 (DPS)


```text
pto.setfmatrix ins(%tile : !pto.tile_buf<...>) outs()
```

## C++ 内建接口


声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent SETFMATRIX(TileData &tile, WaitEvents &... events);
```

## 输入


| 操作数 | 类型 | 说明 |
|--------|------|------|
| `tile` | `TileData` | 要绑定到 FMATRIX 槽的 tile |

## 期望输出


该形式主要由排序或配置效果定义，不产生新的 payload tile。

## 副作用


- **A2/A3 与 A5**：写入 FMATRIX 控制寄存器，将 tile 绑定到矩阵计算单元。
- **CPU 模拟器**：功能性 no-op，不影响架构状态。

## 约束


!!! warning "约束"
    - tile 必须是目标 profile 要求的有效矩阵 tile 类型。
    - 在 A5 上，tile 形状必须与 FMATRIX 槽尺寸兼容。
    - 应在依赖它的矩阵乘操作之前执行。

## 另请参阅


- 指令集总览：[同步与配置](../../sync-and-config_zh.md)
- 下一个操作：[pto.set_img2col_rpt](./set-img2col-rpt_zh.md)

## Summary
本节给出该指令/主题的核心语义与使用定位，和英文章节保持一致。

## Mechanism
本节说明执行机制与关键语义规则，细节与边界条件以英文版为准。

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
