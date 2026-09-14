# PTO-ISA Costmodel 用户指南

本文说明如何在 PTO-ISA 中使用 A2/A3 和 A5 的 host 侧 costmodel。

costmodel 是 host 侧编译路径。用户代码仍然包含 PTO-ISA 头文件，并继续调用 `TADD`、`TLOAD`、`TSTORE`、`SYNCALL` 等 PTO API，但底层实现会被切换到 host mock。costmodel 会记录 PTO 指令并返回预测 cycles，不会真正启动 NPU kernel。

## 支持目标

| 目标 | `ARCH` 参数 | `__NPU_ARCH__` | cycle 来源 |
| --- | --- | --- | --- |
| A2/A3 | `A2A3` | `2201` | lightweight costmodel 和 perf_sim fallback |
| A5 | `A5` | `3101` | memory、sync 和 cube 沿用现有模型；VF 公式模型正在逐项接入 |

## CMake 接入

在构建 host 测试或工具的 target 上 include costmodel helper，然后启用对应架构：

```cmake
include(<pto-isa-root>/include/pto/costmodel/cmake/pto_costmodel.cmake)

add_executable(my_costmodel_test main.cpp)
pto_enable_costmodel(my_costmodel_test ARCH A2A3)
```

A5 使用方式：

```cmake
include(<pto-isa-root>/include/pto/costmodel/cmake/pto_costmodel.cmake)

add_executable(my_a5_costmodel_test main.cpp)
pto_enable_costmodel(my_a5_costmodel_test ARCH A5)
```

`pto_enable_costmodel()` 会给目标添加 PTO include 路径，并定义：

```text
__COSTMODEL
PTO_COMM_NOT_SUPPORTED
__NPU_ARCH__=<目标架构>
```

A5 仅使用 PTO-ISA 自带的头文件，不需要额外的模拟器源码、LLVM pass 或链接库。

## A2/A3 行为

A2/A3 下，PTO wrapper 会进入 costmodel 路径：

```text
pto/pto-inst.hpp
  -> costmodel/runtime_stub.hpp
  -> costmodel/pto_instr.hpp
  -> a2a3/cce_costmodel/*
```

A2/A3 costmodel 优先使用 lightweight 公式模型。对于 lightweight 未覆盖的指令，perf_sim 会根据 opcode、tile shape、dtype 和 pipe stage 使用 fallback 估算 cycles。

## A5 行为

A5 下，PTO wrapper 会进入 A5 costmodel 路径：

```text
pto/pto-inst.hpp
  -> costmodel/runtime_stub.hpp
  -> costmodel/pto_instr.hpp
  -> a5/cce_costmodel/*
```

A5 设备实现中的 intrinsic 在 host 编译时由轻量 mock 提供声明和空实现。VF 周期将由 tileop 的公式模型直接预测，不再捕获 VF 指令流或构造中间表示。

第一阶段尚未接入具体 VF 公式。此时 VF 指令会记录为 `Unsupported`，cycles 保持为 0，并提供诊断信息；不会使用通用 fallback 伪造一个可用预测值。memory、sync 和 cube 仍按各自现有模型处理。

## 获取 cycle 结果

costmodel 会通过 perf_sim 记录 PTO 指令。通常用户只需要从测试或集成侧已有的 costmodel/perf_sim 结果路径读取 cycle 数值。

对于 A5，读取 cycle 前应先检查记录中的 `costmodel_status`。只有 `Supported` 状态下的 cycle 才是有效预测；`Unsupported` 的诊断信息会指出尚未覆盖的 VF 指令。

## 注意事项

- costmodel 路径用于 host 侧估算和验证，不会执行真实 NPU kernel。
- A5 host mock 只解决普通 C++ 编译器无法识别设备 intrinsic 的问题，不执行设备侧运算。
- 用户 kernel 代码保持正常 PTO API 写法，通过 CMake 选择 costmodel 路径，不需要为了 costmodel 改写 kernel API。
