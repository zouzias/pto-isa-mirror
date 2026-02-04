# TFILLPAD_EXPAND

## 简介

`TFILLPAD` 的 expand 变体：允许 `dst` 大于 `src`（实现定义）。

## 计算流程图

![TFILLPAD_EXPAND 计算流程图](figures/TFILLPAD_EXPAND.svg)

## See also

- TFILLPAD 概述和约束：`docs/isa/TFILLPAD.md`。

## C++ Intrinsic（内建接口）

在 `include/pto/common/pto_instr.hpp` 中声明：

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_EXPAND(DstTileData &dst, SrcTileData &src,
                            WaitEvents&... events);
```