# SETFMATRIX

## 指令示意图

![SETFMATRIX tile operation](../figures/isa/SETFMATRIX.svg)

## 简介

为类 IMG2COL 操作设置 FMATRIX 寄存器。

## 数学语义

该指令不直接产生张量算术结果。它将 ConvTile 的 fmap 宽/高与 padding 列表打包写入 **FMATRIX 硬件寄存器**，供后续 `TIMG2COL` 等类 IMG2COL 操作使用。寄存器布局（实现定义）：低 16 位为 `fmapW`，次 16 位为 `fmapH`，从第 32 位起每 8 位存放一个 padding 值（共 4 个，取自 `src.GetPadListArray()[0..3]`）。

仅在 `FmatrixMode` 为 `FMATRIX_A_MANUAL` / `FMATRIX_B_MANUAL` 时生效（分别调用 `set_fmatrix` / `set_fmatrix_b`）；`FMATRIX_A_AUTO` / `FMATRIX_B_AUTO` 下为空操作。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SETFMATRIX(ConvTileData &src, WaitEvents &... events);
```

## 约束

- 该指令属于后端相关能力，仅在支持 FMATRIX 寄存器的后端可用。
- `src` 必须是能提供 `GetFmapW()` / `GetFmapH()` / `GetPadListArray()` 的 ConvTile 类型。
- `FmatrixMode` 取 `FMATRIX_A_MANUAL` / `FMATRIX_B_MANUAL` 时才写入寄存器；`FMATRIX_A_AUTO` / `FMATRIX_B_AUTO` 为空操作。
- 在同一执行流中，应先设置 FMATRIX，再执行依赖的 `TIMG2COL` 指令。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_setfmatrix() {
  // ConvTile<Loc, Element, BufferSize, Layout, ConvTileShape<...>>
  using CfgTile = ConvTile<TileType::Mat, half, 128, Layout::NC1HWC0,
                           ConvTileShape<1, 1, 16, 16, 16>>;
  CfgTile cfg;

  SETFMATRIX(cfg);                                            // 默认 FmatrixMode = FMATRIX_A_MANUAL
  SETFMATRIX<CfgTile, SetFmatrixMode::FMATRIX_B_MANUAL>(cfg); // 显式指定 B 侧
}
```
