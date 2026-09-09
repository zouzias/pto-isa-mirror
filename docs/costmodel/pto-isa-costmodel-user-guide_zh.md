# PTO-ISA Costmodel 用户指南

本文说明如何在 PTO-ISA 中使用 A2/A3 和 A5 的 host 侧 costmodel。

costmodel 是 host 侧编译路径。用户代码仍然包含 PTO-ISA 头文件，并继续调用 `TADD`、`TLOAD`、`TSTORE`、`SYNCALL` 等 PTO API，但底层实现会被切换到 host mock。costmodel 会记录 PTO 指令并返回预测 cycles，不会真正启动 NPU kernel。

## 支持目标

| 目标 | `ARCH` 参数 | `__NPU_ARCH__` | cycle 来源 |
| --- | --- | --- | --- |
| A2/A3 | `A2A3` | `2201` | lightweight costmodel 和 perf_sim fallback |
| A5 | `A5` | `3101` | VF scope 由 A5 VfSim 预测，其它指令由 perf_sim/fallback 处理 |

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

对于 A5，它还会启用 VfSim 构建支持，并链接需要的 VfSim 组件。

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
  -> pkg_inc/pto/costmodel/vfsim/*
```

A5 的 VF scope 内指令会被捕获并组装成 `VfInfo`，再交给 VfSim 预测 cycles。预测结果会作为当前 PTO 指令 cycles 的一部分记录下来。

memory、sync、cube 以及 VfSim 不支持或信息不足的 VF 形态，会由 host mock 和 fallback 估算处理。fallback 的目标是保证详细 VfSim 结果不可用时，仍然能够返回可用的 cycle 数值。

## 获取 cycle 结果

costmodel 会通过 perf_sim 记录 PTO 指令。通常用户只需要从测试或集成侧已有的 costmodel/perf_sim 结果路径读取 cycle 数值。

对于 A5，公开使用方式以 cycle 结果为主。fallback 可能在内部发生，但打包使用时不要求用户依赖 fallback 日志或诊断信息。

## 注意事项

- costmodel 路径用于 host 侧估算和验证，不会执行真实 NPU kernel。
- A5 costmodel 需要 `pkg_inc/pto/costmodel/vfsim/cmake` 下的 VfSim CMake helper。
- 用户 kernel 代码保持正常 PTO API 写法，通过 CMake 选择 costmodel 路径，不需要为了 costmodel 改写 kernel API。
