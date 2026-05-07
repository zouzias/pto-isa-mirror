# TFILLPAD


## 指令示意图


![TFILLPAD tile operation](../../../../figures/isa/TFILLPAD.svg)

## 简介


`TFILLPAD` 复制源 Tile，并把源 valid region 之外的部分用一个**编译期确定的 pad 值**补满。它最常见的用途，是把“运行时有效矩形”扩成“可安全继续参与后续计算的完整静态 Tile”。

如果后续操作不想显式处理边界，就需要有人把边界外的位置先变成确定值。`TFILLPAD` 做的正是这件事。

## 数学语义


设：

- `VR = src.GetValidRow()`
- `VC = src.GetValidCol()`

对 `dst` 的每个元素 `(i, j)`：

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src}_{i,j} & \text{当 } i < VR \text{ 且 } j < VC \\
\mathrm{pad} & \text{否则}
\end{cases}
$$

其中 `pad` 来自 `TileDataDst::PadVal`。常见取值有：

- `PadValue::Zero`
- `PadValue::Min`
- `PadValue::Max`
- 通过 `PadValueCustom(...)` 指定的自定义常量

对浮点类型，`Min/Max` 往往会映射到 `-inf/+inf` 一类“适合做极值归约”的值；对整数类型则映射到对应类型的最小值 / 最大值。

## 汇编语法


PTO-AS 形式：参见 [汇编写法与操作数](../../../syntax-and-operands/assembly-model_zh.md)。

同步形式：

```text
%dst = tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）


```text
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, PadValue PadVal = PadValue::Zero, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(TileData &dst, TileData &src, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

相关的同族接口还有：

- `TFILLPAD_INPLACE(dst, src)`：原位填充
- `TFILLPAD_EXPAND(dst, src)`：允许 `dst` 比 `src` 更大

## 约束


!!! warning "约束"
    ### 通用约束

    - Vec Tile 版本要求 `TileDataDst::PadVal != PadValue::Null`。
    - `src` 和 `dst` 的元素大小必须一致，并且当前实现只接受 `1`、`2` 或 `4` 字节元素。
    - 如果 `dst.GetValidRow() == 0` 或 `dst.GetValidCol() == 0`，当前 backend 会直接返回，不执行填充。

    ### 形状约束

    - `TFILLPAD(dst, src)`：`dst.Rows/Cols` 必须与 `src.Rows/Cols` 相同。
    - `TFILLPAD_INPLACE(dst, src)`：`dst.Rows/Cols` 也必须与 `src.Rows/Cols` 相同。
    - `TFILLPAD_EXPAND(dst, src)`：`dst.Rows >= src.Rows` 且 `dst.Cols >= src.Cols`。

    ### Mat Tile 特化

    - 单类型重载 `TFILLPAD(TileData &dst, TileData &src)` 还支持一条 Mat Tile 特化路径。
    - 这条路径当前只支持：
      - NZ 形态的 Mat Tile（非 row-major，`SLayout::RowMajor`）
      - `TileData::PadVal` 为 `PadValue::Zero` 或 `PadValue::Null`
    - 这条 Mat 特化更像“把矩阵 Tile 的未覆盖区域置成可接受的默认值”，而不是通用的 Vec copy+pad。

## 示例


### Vec Tile


```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_vec() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16,
                    BLayout::RowMajor, 16, 16, SLayout::NoneBox,
                    TileConfig::fractalABSize, PadValue::Min>;

  SrcT src;
  DstT dst;
  TFILLPAD(dst, src);
}
```

### Mat Tile


```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_mat() {
  using TileMatData = Tile<TileType::Mat, float, 16, 256,
                           BLayout::ColMajor, 1, 224,
                           SLayout::RowMajor, 512>;

  TileMatData matTile;
  TFILLPAD(matTile, matTile);
}
```

## 相关页面


- [布局与重排指令集](../../layout-and-rearrangement_zh.md)
- [布局参考](../../../state-and-types/layout_zh.md)
- [Tiles 与有效区域](../../../programming-model/tiles-and-valid-regions_zh.md)

# pto.tfillpad
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
