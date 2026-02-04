# PTO ISA 约定

此页面定义 `docs/isa/` 中的每指令 ISA 参考页以及 `include/pto/common/pto_instr.hpp` 中相应的 C++ Intrinsic（内建接口）使用的共享约定。

## 符号

- **Tile**：固定大小的片上 Tile 对象（例如，`pto::Tile<...>`）。许多指令对 Tile 进行操作并使用 Tile 的有效区域（`GetValidRow()`、`GetValidCol()`）。
- **GM（全局存储器）**：通过 `pto::GlobalTensor<...>` 访问片外存储器。
- **标量/立即数**：`*S` / `*C` 变体使用的主机端标量值或编码立即数。

有关这些术语背后的详细 C++ 编程模型，请参阅：

- Tile：`docs/coding/Tile.md`
- 全局张量：`docs/coding/GlobalTensor.md`
- 标量和枚举：`docs/coding/Scalar.md`

## 形状和布局

- **行主序与列主序**：除非另有说明，CPU 模拟器内核采用行主序 Tile。支持多种布局的指令将明确说明支持的布局。
- **有效区域**：切片的运行时计算区域，表示为 `(valid_row, valid_col)` 并通过 `GetValidRow()` / `GetValidCol()` 查询。

### 有效区域语义

对于指令页面，当我们说“对于有效区域中的每个元素 `(i, j)`”时，我们的意思是：

- `valid_row = dst.GetValidRow()` 和 `valid_col = dst.GetValidCol()` 除非指令显式定义不同的域（例如，某些操作可能使用源 Tile 的有效区域）。
- 数学解释仅针对 `0 <= i < valid_row` 和 `0 <= j < valid_col` 的索引定义 `dst[i, j]`。
- 有效区域之外的元素**未指定**，除非指令明确说明否则（不要假设它们被清零或保留）。

对于多操作数指令（例如，`src0`、`src1`），文档假定输入 Tile 与迭代域兼容，除非约束部分规定了更严格的要求。

## 类型

- 指令页面列出了支持的数据类型（例如，`fp16`、`fp32`、`int8`、`int16`、`int32`、`uint8`、 `uint16`、`uint32`）。 CPU 模拟器支持可能是一个子集，并记录在 `include/README.md` 中。

## 事件和同步

- 指令可能需要在内存和向量管道之间进行排序。当示例显示事件（例如，`set_flag(...)` / `wait_flag(...)`）时，它们指示目标后端所需的排序约束。
- `TSYNC` 用于指令序列需要时的显式同步。

请参阅 `docs/coding/Event.md` 了解 PTO Tile Lib 使用的事件模型。