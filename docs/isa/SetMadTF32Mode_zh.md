# SetMadTF32Mode

## 指令示意图

![SetMadTF32Mode tile operation](../figures/isa/SetMadTF32Mode.svg)

## 简介

设置 TF32 变换模式（实现定义），在进行TMatmul操作前，对FP32的操作数进行精度模式设置，在不同的平台上实现不同的精度模式：
  - 在A2/A3平台实现HF32（e8m11）。
  - 在A5平台实现TF32（e8m10）。

## 数学语义

该指令本身不执行直接的张量运算，而是更新后续指令所使用的目标模式状态。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

Schematic form:

```text
SetMadTF32Mode {mode = ...}
```

### AS Level 1（SSA）

```text
pto.SetMadTF32Mode {mode = ...}
```

### AS Level 2（DPS）

```text
pto.SetMadTF32Mode ins({mode = ...}) outs()
```

## C++ 内建接口

声明于 `include/pto/common/pto_tile.hpp`：

```cpp
PTO_INTERNAL void SetMadTF32Mode(RoundMode tf32TransMode = RoundMode::CAST_ROUND)
```

## 约束

- 仅当相应的后端能力宏启用时可用。

- 具体的模式值和硬件行为由目标平台定义。

- 该指令具有控制状态副作用，应相对于依赖的计算指令进行适当排序。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_enable_tf32() {
  using LeftTile = TileLeft<U, M, K, M, K>;
  LeftTile aTile;
  aTile.SetMadTF32Mode(RoundMode::CAST_ROUND);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
pto.SetMadTF32Mode {mode = ...}
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tile.SetMadTF32Mode {mode = ...}
```

### PTO 汇编形式

```text
pto.tile.SetMadTF32Mode {mode = ...}
# IR Level 2 (DPS)
pto.tile.SetMadTF32Mode ins({mode = ...}) outs()
```

