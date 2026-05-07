# TIMG2COL


## 指令示意图


![TIMG2COL tile operation](../../../../figures/isa/TIMG2COL.svg)

## 简介


`TIMG2COL` 把输入特征图 Tile 重排成卷积友好的列矩阵形式，是 PTO 里连接卷积样式输入布局与矩阵乘法路径的关键桥梁。

这条指令不只是简单提取窗口。它同时综合了：

- 输入特征图几何信息
- kernel 大小
- stride / dilation
- padding
- channel 打包方式
- 当前在逻辑 im2col 矩阵中的起始位置 `posM / posK`

## 数学语义


把卷积输入展开成矩阵时，可以把输出矩阵看成按 `(m, k)` 编址：

- `m` 选择输出空间位置
- `k` 选择卷积核内的通道与空间偏移

`TIMG2COL` 会把源特征图中与 `(posM, posK)` 对应的卷积窗口元素，写到目标 Left Tile 中。若窗口越过输入边界，则写入 pad value。

CPU 模拟器里的显式计算逻辑是：

- 先根据 `stride / dilation / filter / pad` 推出输出位置 `(outRow, outCol)`
- 再根据 `kIndex` 推出 `(channelIndex, kernelH, kernelW)`
- 若映射回输入后的 `(inputH, inputW)` 越界，则写 `padValue`
- 否则读取源特征图相应元素

## 汇编语法


PTO-AS 形式：参见 [汇编写法与操作数](../../../syntax-and-operands/assembly-model_zh.md)。

### AS Level 1（SSA）


```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL,
          typename... WaitEvents>
PTO_INST RecordEvent TIMG2COL(TileData &dst, ConvTileData &src, uint16_t posM = 0, uint16_t posK = 0,
                              WaitEvents&... events);
```

## 约束


!!! warning "约束"
    ### 通用约束

    - `src` 必须是卷积配置/特征图 Tile，位置类型为 `TileType::Mat`。
    - 输入布局必须是 `NC1HWC0` 或 `NDC1HWC0`。
    - `dst` 必须是 `TileType::Left`。
    - `src` 与 `dst` 的元素类型必须一致。
    - `posM / posK` 不是像素坐标，而是逻辑 im2col 矩阵中的起始偏移。

    ### A2/A3 实现

    - 支持的数据类型是：
      `int8_t`、`half`、`bfloat16_t`、`float`。
    - A2/A3 的 `Left` 目标约束是：
      - `dst.SFractal == SLayout::RowMajor`
      - `dst.isRowMajor == true`
    - 当 `FmatrixMode` 为 `FMATRIX_A_AUTO` 或 `FMATRIX_B_AUTO` 时，A2/A3 会自动根据 `src` 的：
      - `fmapH / fmapW`
      - `padList`
      来设置 FMATRIX。
    - A2/A3 的 `TIMG2COL` auto 路径**不会**顺手设置 repeat 和 padding 寄存器；如果后续路径依赖这些状态，应显式使用对应的 `pto.set_img2col_rpt` 或 `pto.set_img2col_padding`。

    ### A5 实现

    - 支持的数据类型更宽，除 `int8_t/half/bfloat16_t/float` 外，还覆盖若干 `uint*` / `int*` 类型。
    - A5 的 `Left` 目标约束是：
      - `dst.SFractal == SLayout::RowMajor`
      - `dst.isRowMajor == false`
    - 当 `FmatrixMode` 为 `FMATRIX_A_AUTO` 或 `FMATRIX_B_AUTO` 时，A5 会自动根据 `src` 的：
      - `fmapH / fmapW`
      - `padList`
      - `repeatStride / repeatTime / repeatMode / dstStride / dstMposition`
      - `padValue`
      一并设置 FMATRIX、repeat 和 padding 状态。

    ### CPU 模拟器

    - CPU 使用显式公式直接完成 im2col 展开。
    - CPU 目前沿用与 A5 相同的 `Left` 目标布局约束：
      - `dst.SFractal == SLayout::RowMajor`
      - `dst.isRowMajor == false`

    这意味着 `TIMG2COL` 的 Left Tile 细节并不是所有目标完全一致，文档里必须按目标区分。

## 示例


```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename LeftTile, typename ConvTile>
void example(LeftTile& dst, ConvTile& src) {
  TIMG2COL(dst, src, /*posM=*/0, /*posK=*/0);
}
```

## 相关页面


- [pto.setfmatrix](../sync-and-config/setfmatrix.md)
- [pto.set_img2col_rpt](../sync-and-config/set-img2col-rpt.md)
- [pto.set_img2col_padding](../sync-and-config/set-img2col-padding.md)

# pto.timg2col
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Summary
本节给出该指令/主题的核心语义与使用定位，和英文章节保持一致。

## Mechanism
本节说明执行机制与关键语义规则，细节与边界条件以英文版为准。

## Syntax
本节列出语法形态（SSA / DPS / Assembly），用于与英文页逐项对照。

### AS Level 1 (SSA)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### AS Level 2 (DPS)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### IR Level 1 (SSA)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### IR Level 2 (DPS)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

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

## Exceptions
本节描述非法输入、不支持组合与验证失败行为。

## Target-Profile Restrictions
本节给出 A2/A3、A5 及 CPU-SIM 的差异化限制与行为说明。

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

# Optional for tile operands:
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

# pto.tassign %arg0, @tile(0x1000)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

# pto.tassign %arg1, @tile(0x2000)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### PTO Assembly Form
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

# AS Level 2 (DPS)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Related Ops / Instruction Set Links
本节给出上下游指令与相关章节链接。
