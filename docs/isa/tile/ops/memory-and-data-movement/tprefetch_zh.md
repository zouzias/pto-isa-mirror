# pto.tprefetch


`pto.tprefetch` 属于[内存与数据搬运](../../memory-and-data-movement_zh.md)指令集。

## 概述


把后续可能会访问的一段 `GlobalTensor` 数据提前搬进 tile 本地缓冲，用作预取或预热。

## 机制


若把 `src` 的当前视图看成二维切片，则：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j} $$

和 `TLOAD` 不同，这条指令的重点不是复杂布局转换，而是把“稍后会用到的数据”尽早拉近。它在当前仓库实现里并不是“可以完全忽略的 hint bit”，而是真会填充 `dst`。

## 语法


同步形式：

```text
%dst = tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### AS Level 1（SSA）


```text
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


```cpp
template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData &dst, GlobalData &src);
```

与大多数 PTO C++ 接口不同，这条封装不会自动执行 `TSYNC(events...)`。

## 输入


- `src`：源 `GlobalTensor`
- `dst`：目标 tile 缓冲

## 预期输出


- `dst`：被预取进本地路径的数据

## 副作用


这条指令可能会从 GM 读取。某些 target 可能会把它当提示，也可能会直接完成缓冲填充。

## 约束


!!! warning "约束"
    - 可移植代码应把它用于“提前搬运即将访问的数据”，而不要把它当 `TLOAD` 的语义等价替代品。
    - 在当前仓库实现里，预取范围仍然受 `dst` 大小和 `src` 切片大小影响。

## Target-Profile 限制


### CPU


- CPU 会直接按 `dst.GetValidRow()` / `dst.GetValidCol()` 做逐元素拷贝

### A2A3 / A5


- 若单个切片能放进 `dst`，则一次性预取
- 若放不下，则按 `dst` 容量分块预取

## 异常与非法情形


!!! danger "异常与非法情形"
    - 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能


当前手册未单列 `tprefetch` 的公开周期表。它的真正收益更依赖后续访存局部性，而不是单条指令本身的独立算术成本。

## 示例


```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T, typename GT>
void example(Tile<TileType::Vec, T, 16, 16>& tileBuf, GT& globalView) {
  TPREFETCH(tileBuf, globalView);
}
```

## 相关页面


- [TLOAD](./tload_zh.md)
- [内存与数据搬运](../../memory-and-data-movement_zh.md)

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
