# TPRINT

## 简介

调试指令：打印/导出 Tile 内容（实现定义）。

## 计算流程图

![TPRINT 计算流程图](figures/TPRINT.svg)

## 汇编语法

PTO-AS 形式：参见 `docs/grammar/PTO-AS.md`。

```text
tprint %src : !pto.tile<...> | !pto.global<...>
```

## C++ Intrinsic（内建接口）
在 `include/pto/common/pto_instr.hpp` 中声明：
```cpp
template <typename T, typename... WaitEvents>
PTO_INST RecordEvent TPRINT(T &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TPRINT, src);
  return {};
}
```

### Supported Types for T
- **Tile**：必须是具有受支持元素类型的向量 Tile (`TileType::Vec`)。
- **GlobalTensor**：必须使用布局 `ND`、`DN` 或 `NZ`，并且具有受支持的元素类型。

## 约束

- **支持的元素类型**：
  - 浮点：`float`、`half`
  - 有符号整数：`int8_t`、`int16_t`、`int32_t`
  - 无符号整数：`uint8_t`、`uint16_t`、`uint32_t`
- **对于 Tile**：`TileData::Loc == TileType::Vec`（仅向量 Tile 可打印）。
- **对于 GlobalTensor**：布局必须是 `Layout::ND`、`Layout::DN` 或 `Layout::NZ` 之一。

## Behavior
- **强制编译标志**：

  在 A2/A3/A5 设备上，`TPRINT` 使用 `cce::printf` 通过设备到主机调试通道发出输出。 **您必须启用 CCE 选项 `-D_DEBUG --cce-enable-print`**。

- **缓冲区限制：**

  `cce::printf` 的内部打印缓冲区的大小有限。如果输出超出此缓冲区，可能会出现警告消息，例如 `"Warning: out of bound! try best to print"`，并且**仅打印部分数据**。

- **同步**：

  打印前自动插入 `pipe_barrier(PIPE_ALL)`，以确保之前所有操作完成且数据一致。

- **格式化**：

  - 浮点值：打印为 `%6.2f`
  - 整数值：打印为 `%6d`
  - 对于 `GlobalTensor`，由于数据大小和缓冲区限制，仅打印其逻辑形状（由 `Shape` 定义）内的元素。
  - 对于 `Tile`，指定部分有效性时，无效区域（超出 `validRows`/`validCols` ）仍会打印，但会用 `|` 分隔符标记。

## 示例

### Print a Tile

```cpp
#include <pto/pto-inst.hpp>

PTO_INTERNAL void DebugTile(__gm__ float *src) {
  using ValidSrcShape = TileShape2D<float, 16, 16>;
  using NDSrcShape = BaseShape2D<float, 32, 32>;
  using GlobalDataSrc = GlobalTensor<float, ValidSrcShape, NDSrcShape>;
  GlobalDataSrc srcGlobal(src);

  using srcTileData = Tile<TileType::Vec, float, 16, 16>;
  srcTileData srcTile;
  TASSIGN(srcTile, 0x0);

  TLOAD(srcTile, srcGlobal);
  TPRINT(srcTile);
}
```

### Print a GlobalTensor

```cpp
#include <pto/pto-inst.hpp>

PTO_INTERNAL void DebugGlobalTensor(__gm__ float *src) {
  using ValidSrcShape = TileShape2D<float, 16, 16>;
  using NDSrcShape = BaseShape2D<float, 32, 32>;
  using GlobalDataSrc = GlobalTensor<float, ValidSrcShape, NDSrcShape>;
  GlobalDataSrc srcGlobal(src);

  TPRINT(srcGlobal);
}
```