# PTO Costmodel 方案说明

## 概述

本文档描述 `include/pto/costmodel` 的设计方案——一个面向 PTO-ISA 的主机侧 costmodel 后端。

它的目标是：

- 在标准主机工具链上编译 PTO NPU 代码路径
- 记录 PTO 与 CCE 的执行轨迹
- 基于轨迹估算执行代价，并将结果绑定到目标 Tile 上
- 将 costmodel 专用逻辑与主 PTO common 路径隔离

该后端面向分析与评估场景，不是数值模拟器。

## 目的

`include/pto/costmodel` 旨在为 PTO 开发提供一条轻量级分析路径，主要解决四类问题：

- **主机编译**：让原本依赖设备限定符、ACL 运行时和 CCE intrinsic 的 PTO 代码能够在 host 上编译。
- **轨迹可见性**：明确顶层 PTO API 实际执行了什么，以及它们向下发出了哪些 CCE 调用。
- **代价评估**：把轨迹结果转换为面向架构的 cycle 估算，并绑定到对应的 PTO 指令目标 Tile 上。
- **集成清晰**：保持主 PTO include 流程不变，同时把 costmodel 逻辑集中在独立目录下。

## 实现方式

整体设计是在标准 PTO 入口之后插入一层主机侧兼容与追踪后端。

### 总体流程

```text
                    用户代码 include pto/pto-inst.hpp
                                  |
                     +------------+------------+
                     |                         |
                     | 未开启 __COSTMODEL      | 开启 __COSTMODEL
                     |                         |
                     v                         v
                  正常 PTO 路径        pto/costmodel/runtime_stub.hpp
                                                |
                           +--------------------+--------------------+
                           |                    |                    |
                           v                    v                    v
                common/qualifiers.hpp   common/aclrt_stub.hpp  common/runtime_util.hpp
                - 设备限定符置空        - ACL runtime 伪实现    - flag/helper 伪实现
                - host 安全宏           - malloc/memcpy/free   - trap/min/max 等
                                                |
                                                v
                                  common/arch_select.hpp
                                                |
                            +-------------------+-------------------+
                            |                                       |
                            v                                       v
                    a2a3/cce_stub.hpp                        a5/cce_stub.hpp
                    - CCE 伪 stub                            - CCE 伪 stub
                    - RecordCceCall(...)                    - RecordCceCall(...)
                            |                                       |
                            +-------------------+-------------------+
                                                |
                                                v
                                    costmodel/pto_instr.hpp
                              - MAP_INSTR_IMPL 宏包装 PTO API
                              - PtoInstrScope 记录 PTO 指令名
                              - CaptureCycleIntoTile 评估并存储 cycle
                                                |
                                                v
                                         trace.hpp
                               - PTO 记录
                               - CCE 调用记录
                               - 参数记录
                                                |
                                                v
                            evaluator/trace_evaluator.hpp + arch_config.hpp
                               - 按架构查规则
                               - cycle 估算
                                                |
                                                v
                                    tile.SetLastCycle(cycles)
                               - 评估结果写入目标 Tile
                               - tile.GetCycle() 可查询
```

### 集成入口

对用户而言，入口仍然是 `pto/pto-inst.hpp`。

当开启 `__COSTMODEL` 时，`pto-inst.hpp` 会把构建路径切换到：

- `runtime_stub.hpp` —— host 运行时兼容层
- `pto_instr.hpp` —— PTO 指令的 costmodel 包装

这样可以保持外部编程模型不变，使 costmodel 成为一种后端选择，而不是一套新的用户接口。

### Host 运行时兼容层

主机运行时兼容层由以下文件提供：

- `common/qualifiers.hpp` —— 将 `__gm__`、`__ubuf__`、`AICORE` 等设备限定符置为空
- `common/aclrt_stub.hpp` —— 提供 malloc/free/memcpy 形式的 ACL runtime 伪实现
- `common/runtime_util.hpp` —— 提供 flag/wait helper 的 host shim

核心思想不是重写 PTO 逻辑，而是补齐它在 host 分析模式下所需的外围运行环境。

### 轨迹采集

轨迹采集集中在 `trace.hpp` 中。它记录：

- 顶层 PTO 指令名
- 发出的 CCE 调用
- 关键参数（Tile 类型、数据类型、形状等）
- 必要时未归属到 PTO 分组的原始 CCE 调用

#### PtoInstrScope —— PTO 指令边界标记

`PtoInstrScope` 是一个 RAII 类，在构造时将 PTO 指令名压入活跃栈，在析构时弹出：

```cpp
// 在 MAP_INSTR_IMPL 宏中展开
::pto::mocker::PtoInstrScope _scope("TADD");  // 构造：记录 PTO 指令开始
TADD_IMPL(dst, src0, src1);                    // 执行：CCE stub 记录底层调用
::pto::mocker::CaptureCycleIntoTile(dst);      // 析构前：评估并存储 cycle
```

因此轨迹会按 PTO 指令聚合：

```text
host 上执行一个 PTO 调用
   |
   v
PtoInstrScope("TADD")
   |
   v
原始 PTO 实现继续执行
   |
   v
CCE 伪 stub 记录如下调用：
- copy_gm_to_ubuf_align_*
- vadd / vmul / vsub
   |
   v
CaptureCycleIntoTile(dst) 评估 trace 并将 cycle 写入 dst tile
   |
   v
轨迹最终归入 TADD, dst.GetCycle() 返回该指令的 cycle 估算
```

### 逐 Tile 的 Cycle 存储

Costmodel 的一个核心设计是 **cycle 结果绑定到目标 Tile**，而非使用全局变量。

#### 设计动机

如果使用全局函数（如 `GetLastPtoInstrCycles()`）返回最近一条指令的 cycle，在多条指令混合执行的场景下会出现错配：

```cpp
TADD(dstA, src0, src1);   // 指令 A 写入 dstA
TMUL(dstB, src2, src3);   // 指令 B 写入 dstB
// 此时全局函数只能返回指令 B 的 cycle，无法查询指令 A 的结果
```

解决方案：每条 PTO 指令执行后，立即评估 trace 并将 cycle 结果写入该指令的目标 Tile。

#### 实现机制

**1. MAP_INSTR_IMPL 宏**

```cpp
#ifdef __COSTMODEL
#define MAP_INSTR_IMPL(API, dst, ...)                                              \
    do {                                                                           \
        ::pto::mocker::PtoInstrScope _scope(#API);                                 \
        API##_IMPL(dst __VA_OPT__(,) __VA_ARGS__);                                 \
        ::pto::mocker::CaptureCycleIntoTile(dst);                                  \
    } while (0)
#else
#define MAP_INSTR_IMPL(API, ...) API##_IMPL(__VA_ARGS__)
#endif
```

宏的第一个参数 `dst` 被分离出来，用于后续的 cycle 捕获。

**2. CaptureCycleIntoTile 函数**

```cpp
template <typename T>
inline void CaptureCycleIntoTile(T &obj)
{
    if constexpr (requires { obj.SetLastCycle(0.0f); }) {
        const auto &trace = GetTrace();
        if (trace.executed_pto.empty()) {
            return;
        }
        const auto &active = trace.active_pto_stack;
        size_t idx = active.empty() ? trace.executed_pto.size() - 1 : active.back();
        auto report = evaluator::EvaluatePtoInstr(
            trace.executed_pto[idx], evaluator::GetDefaultArchConfig());
        obj.SetLastCycle(static_cast<float>(report.total_cycles));
    }
}
```

该函数使用 C++20 concepts（`requires` 表达式）检测目标对象是否支持 `SetLastCycle`。对于 Tile 和 ConvTile 类型会执行评估；对于 GlobalTensor 等不支持该接口的类型，`if constexpr` 会在编译期跳过。

**3. Tile 结构体的扩展**

在 `pto/common/pto_tile.hpp` 中，`Tile` 和 `ConvTile` 通过 `#ifdef __COSTMODEL` 条件编译添加 cycle 存储支持：

```cpp
// Tile 结构体中
#ifdef __COSTMODEL
public:
    float GetCycle() const { return lastCycle_; }
    void SetLastCycle(float c) { lastCycle_ = c; }
#endif

private:
    // ...
#ifdef __COSTMODEL
    float lastCycle_ = 0.0f;
#endif
```

这个修改是 costmodel 专用的，对非 `__COSTMODEL` 构建完全透明。

#### 效果

```cpp
TADD(dstA, src0, src1);
TMUL(dstB, src2, src3);

float addCycles = dstA.GetCycle();   // 返回 TADD 的 cycle
float mulCycles = dstB.GetCycle();   // 返回 TMUL 的 cycle
```

每条指令的 cycle 结果独立存储，不会因为后续指令而丢失。

### 架构相关实现

架构分发由 `common/arch_select.hpp` 负责：

```text
__NPU_ARCH__ == 2201        -> a2a3/cce_stub.hpp
__NPU_ARCH__ == 3101/3510   -> a5/cce_stub.hpp
```

当前结构如下：

- `a2a3/cce_stub.hpp` 为 A2/A3 PTO 路径提供了较广的 fake intrinsic 覆盖。
- `a5/cce_stub.hpp` 与 `a5/compat.hpp` 一起提供受控的 A5 子集支持。

这种拆分方式把共享 host 基础设施与架构专属 stub 行为分离开来，便于逐步扩展。

### 延迟评估

评估层由以下文件组成：

- `arch_config.hpp` —— 架构规则配置
- `evaluator/cce_evaluator.hpp` —— 单条 CCE 调用的 cycle 评估
- `evaluator/trace_evaluator.hpp` —— PTO 指令级 trace 评估

这一层会按照架构规则解释轨迹，并输出 cycle 估算。

```text
记录下来的 CCE 调用
   |
   v
在 ArchConfig 中查找规则
   |
   +--> 固定规则       -> 常量 cycle
   +--> repeat 规则    -> 启动开销 + repeats * 单次代价
   +--> burst 规则     -> 启动开销 + burst/length 代价
   +--> mad 规则       -> 基于 tile 的矩阵估算
   |
   v
得到单个 CCE 调用的 cycle 估算
   |
   v
汇总成 PTO 指令级代价（EvaluatePtoInstr）
   |
   v
通过 CaptureCycleIntoTile 写入目标 Tile
```

### 测试框架

Costmodel 测试基于 Google Test，采用 `EXPECT_CYCLE_NEAR` 宏进行 cycle 精度校验：

```cpp
#define EXPECT_CYCLE_NEAR(tile, profiling, accuracy)
```

- `tile`：执行 PTO 指令后的目标 Tile，通过 `GetCycle()` 获取实际 cycle
- `profiling`：真实设备 Profiling 获取的期望值
- `accuracy`：最小相对精度要求（0~1）

测试目录结构：

```text
tests/costmodel/
├── USER_GUIDE.zh-CN.md          # 使用指南
├── README.md                    # 测试说明
└── st/
    ├── CMakeLists.txt           # 顶层 CMake
    ├── common/
    │   └── cost_check.hpp       # EXPECT_CYCLE_NEAR 宏定义
    └── testcase/
        └── a2a3/
            ├── CMakeLists.txt   # 用例注册
            ├── tadd/tadd.cpp
            ├── tmul/tmul.cpp
            └── ...
```

测试运行入口：

```bash
# 运行全部测试
./tests/run_costmodel_tests.sh

# 运行单个用例
python tests/run_costmodel.py --testcase tadd --clean --verbose
```

## 优点

**职责分离清晰。** costmodel 逻辑集中在 `include/pto/costmodel` 下，而不是分散到主 PTO common 实现中。对 `pto_tile.hpp` 的修改仅限于条件编译的 cycle 存储字段，对非 costmodel 构建完全透明。

**开发效率更高。** 开发者可以在 host 上编译并观察 PTO lowering 行为，而不依赖真实设备运行时。这对于指令打通、轨迹检查和 costmodel 调优都很有价值。

**架构分层明确。** 实现被清晰分成五部分：

- host 运行时兼容
- 架构专属 intrinsic stub
- 轨迹记录
- 延迟评估
- 逐 Tile cycle 存储

**保持了原有 PTO 使用方式。** 用户代码依然通过 `pto/pto-inst.hpp` 进入，因此 costmodel 是一个后端路径，而不是一个新的使用框架。

**逐指令 cycle 精度。** cycle 结果绑定到每条指令的目标 Tile，避免了全局状态错配问题，支持多条指令混合执行的独立查询。

## 目录结构

```text
include/pto/costmodel/
├── README.md                    # 本文档
├── COSTMODEL_PROPOSAL.zh-CN.md  # 方案说明（本文档）
├── runtime_stub.hpp             # host 运行时总入口
├── trace.hpp                    # 轨迹采集
├── arch_config.hpp              # 架构规则配置
├── pto_instr.hpp                # PTO 指令 costmodel 包装
├── common/
│   ├── qualifiers.hpp           # 设备限定符替代
│   ├── aclrt_stub.hpp           # ACL runtime 伪实现
│   ├── runtime_util.hpp         # host helper shim
│   └── arch_select.hpp          # 架构选择分发
├── a2a3/
│   └── cce_stub.hpp             # A2/A3 CCE 伪 stub
├── a5/
│   ├── cce_stub.hpp             # A5 CCE 伪 stub
│   └── compat.hpp               # A5 兼容层
└── evaluator/
    ├── cce_evaluator.hpp        # CCE 调用级评估
    └── trace_evaluator.hpp      # PTO 指令级评估

tests/costmodel/
├── USER_GUIDE.zh-CN.md          # 使用指南
├── README.md                    # 测试说明
└── st/
    ├── CMakeLists.txt
    ├── common/
    │   └── cost_check.hpp
    └── testcase/
        └── a2a3/
            ├── CMakeLists.txt
            ├── tadd/tadd.cpp
            └── ...
```

## 总结

`include/pto/costmodel` 是一个主机侧 PTO 轨迹与代价评估后端。它保留了标准 PTO 入口，通过 host 运行时替代层和可追踪的 intrinsic stub 实现 costmodel 行为。核心设计特点包括：

1. **零侵入**：对非 costmodel 构建完全透明
2. **逐 Tile 绑定**：cycle 结果存储在目标 Tile 上，支持多指令混合场景
3. **架构可扩展**：通过 `__NPU_ARCH__` 分发到不同架构的 CCE stub
4. **自动化测试**：`EXPECT_CYCLE_NEAR` 宏 + GTest 框架，支持 CI 集成
