# pto.set_img2col_padding


`pto.set_img2col_padding` 属于[同步与配置](../../sync-and-config_zh.md)指令集。

## 概要


从 IMG2COL 配置 tile 设置 IMG2COL padding 元数据。在 A2/A3 与 A5 上，该指令写入 FMATRIX padding 寄存器，控制卷积窗口滑动前在图像 patch 周围补零的形态。CPU 模拟器上该指令为功能性 no-op。

该指令不直接产生 tensor 算术结果。它更新后续数据搬运操作会读取的 IMG2COL padding 控制状态。

## 语法


文本拼写由 PTO ISA 的语法与操作数页面定义。

```text
pto.set_img2col_padding %cfg
```

### AS Level 1 (SSA)


```text
pto.set_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

### AS Level 2 (DPS)


```text
pto.set_img2col_padding ins(%cfg : !pto.fmatrix_config) outs()
```

## C++ 内建接口


声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename ConvTileData, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_PADDING(ConvTileData &src, WaitEvents &... events);
```

## 输入


| 操作数 | 说明 |
|--------|------|
| `src` | 包含 padding 元数据的 IMG2COL 配置 tile |

## 期望输出


该形式主要由排序或配置效果定义，不产生新的 payload tile。

## 副作用


- **A2/A3 与 A5**：更新 FMATRIX padding-control 寄存器，由同一执行流中的后续 `pto.timg2col` DMA 操作读取。
- **CPU 模拟器**：不影响架构状态。

## 约束


!!! warning "约束"
    - 该指令仅适用于暴露 IMG2COL 配置状态的后端。
    - `src` 必须是后端实现接受的有效 IMG2COL 配置 tile 类型。
    - 应在同一执行流中依赖它的 `pto.timg2col` 操作之前使用。

## 另请参阅


- 指令集总览：[同步与配置](../../sync-and-config_zh.md)
- 前一个操作：[pto.set_img2col_rpt](./set-img2col-rpt_zh.md)
- 下一个操作：[pto.subview](./subview.md)

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

### Auto Mode
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

# Auto mode: compiler/runtime-managed placement and scheduling.
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Manual Mode
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

# Manual mode: bind resources explicitly before issuing the instruction.
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## See Also
本节给出上下游指令与相关章节链接。
