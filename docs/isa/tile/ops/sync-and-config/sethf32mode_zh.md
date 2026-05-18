# pto.sethf32mode

`pto.sethf32mode` 属于[同步与配置](../../sync-and-config_zh.md)指令集。

## 概要

配置受支持矩阵乘或卷积路径使用的 HF32 变换模式。

该指令控制后端定义的 HF32 变换行为。它属于 tile 同步与配置控制面，可见效果是建立排序或状态，而不是产生算术载荷。

该指令不直接产生 tensor 算术结果。它更新后续指令会读取的目标模式状态。

## 语法

文本拼写由 PTO ISA 的语法与操作数页面定义。

示意形式：

```text
pto.sethf32mode {enable = true, mode = ...}
```

### IR Level 1 (SSA)

```text
pto.sethf32mode {enable = true, mode = ...}
```

### IR Level 2 (DPS)

```text
pto.sethf32mode ins({enable = true, mode = ...}) outs()
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <bool isEnable, RoundMode hf32TransMode = RoundMode::CAST_ROUND, typename... WaitEvents>
PTO_INST RecordEvent SETHF32MODE(WaitEvents &... events);
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `enable` | `bool` | 启用（`true`）或禁用（`false`）HF32 变换模式 |
| `mode` | `RoundMode` | HF32 舍入模式；支持的取值由目标 profile 定义 |

## 期望输出

该形式主要由排序或配置效果定义，不产生新的 payload tile。

## 副作用

- **A2/A3 与 A5**：当目标 profile 支持时，配置实现定义的 HF32 变换模式状态。
- **CPU 模拟器**：在建模时更新软件模拟模式状态；否则为功能性 no-op。

## 约束

!!! warning "约束"
    - 精确的模式取值与后续硬件行为由目标定义。
    - 该指令具有控制状态副作用，应当相对于依赖它的计算指令正确排序。
    - 程序不得把该操作视为 tile payload 变换；它只修改模式状态。

## 不允许的情形

!!! danger "不允许的情形"
    - 在不支持 HF32 变换模式的目标 profile 上依赖 HF32 行为。
    - 使用超出所选目标 profile 支持集合的模式值。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_enable_hf32() {
  SETHF32MODE<true, RoundMode::CAST_ROUND>();
}
```

## 另请参阅

- 指令集总览：[同步与配置](../../sync-and-config_zh.md)
- 前一个操作：[pto.tassign](./tassign_zh.md)
- 相关模式操作：[pto.settf32mode](./settf32mode_zh.md)
- 后一个操作：[pto.setfmatrix](./setfmatrix_zh.md)
