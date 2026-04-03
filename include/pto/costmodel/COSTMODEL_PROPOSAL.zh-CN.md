# PTO Costmodel 方案说明

## 概述

本文档将 `include/pto/costmodel` 定义为一个面向 PTO-ISA 的主机侧 costmodel 后端，供 PTO-ISA 开发者和管理者评审。

它的目标是：

- 在标准主机工具链上编译 PTO NPU 代码路径
- 记录 PTO 与 CCE 的执行轨迹
- 基于轨迹估算执行代价
- 将 costmodel 专用逻辑与主 PTO common 路径隔离

该后端面向分析与评估场景，不是数值模拟器。

## 目的

`include/pto/costmodel` 旨在为 PTO 开发提供一条轻量级分析路径，主要解决四类问题：

- 主机编译：让原本依赖设备限定符、ACL 运行时和 CCE intrinsic 的 PTO 代码能够在 host 上编译。
- 轨迹可见性：明确顶层 PTO API 实际执行了什么，以及它们向下发出了哪些 CCE 调用。
- 代价评估：把轨迹结果转换为面向架构的 cycle 估算。
- 集成清晰：保持主 PTO include 流程不变，同时把 costmodel 逻辑集中在独立目录下。

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
                              - 包装顶层 PTO API 调用
                              - 记录 PTO 指令名
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
```

### 集成入口

对用户而言，入口仍然是 `pto/pto-inst.hpp`。

当开启 `__COSTMODEL` 时，[`pto-inst.hpp`](/home/lc/pto-isa-costmodel/include/pto/pto-inst.hpp) 会把构建路径切换到：

- [`runtime_stub.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/runtime_stub.hpp)
- [`pto_instr.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/pto_instr.hpp)

这样可以保持外部编程模型不变，使 costmodel 成为一种后端选择，而不是一套新的用户接口。

### Host 运行时兼容层

主机运行时兼容层由以下文件提供：

- [`common/qualifiers.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/common/qualifiers.hpp)
- [`common/aclrt_stub.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/common/aclrt_stub.hpp)
- [`common/runtime_util.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/common/runtime_util.hpp)

这些文件把设备侧假设替换为 host 可用的定义。

```text
设备侧预期                            costmodel 替代方式
-----------------------              -------------------------
__gm__ / __ubuf__ / AICORE           置空的 host 安全宏
ACL runtime API                      malloc/free/memcpy 伪实现
flag/wait helper                     host helper shim
CCE tile 指针访问                    host 可见指针路径
```

核心思想不是重写 PTO 逻辑，而是补齐它在 host 分析模式下所需的外围运行环境。

### 轨迹采集

轨迹采集集中在 [`trace.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/trace.hpp) 中。

它记录：

- 顶层 PTO 指令名
- 发出的 CCE 调用
- 关键参数
- 必要时未归属到 PTO 分组的原始 CCE 调用

在 PTO API 边界上，[`costmodel/pto_instr.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/pto_instr.hpp) 通过 `PtoInstrScope` 包装顶层 PTO API。

因此轨迹会按 PTO 指令聚合：

```text
host 上执行一个 PTO 调用
   |
   v
PtoInstrScope("TLOAD")
   |
   v
原始 PTO 实现继续执行
   |
   v
CCE 伪 stub 记录如下调用：
- copy_gm_to_ubuf_align_*
- vadd / vmul / vsub
- mad
- copy_ubuf_to_gm_align_*
   |
   v
轨迹最终归入 TLOAD / TADD / TMATMUL / TSTORE
```

### 架构相关实现

架构分发由 [`common/arch_select.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/common/arch_select.hpp) 负责：

```text
__NPU_ARCH__ == 2201        -> a2a3/cce_stub.hpp
__NPU_ARCH__ == 3101/3510   -> a5/cce_stub.hpp
```

当前结构如下：

- [`a2a3/cce_stub.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/a2a3/cce_stub.hpp) 为 A2/A3 PTO 路径提供了较广的 fake intrinsic 覆盖。
- [`a5/cce_stub.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/a5/cce_stub.hpp) 与 [`a5/compat.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/a5/compat.hpp) 一起提供受控的 A5 子集支持。

这种拆分方式把共享 host 基础设施与架构专属 stub 行为分离开来，便于逐步扩展。

### 延迟评估

评估层由以下文件组成：

- [`arch_config.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/arch_config.hpp)
- [`evaluator/cce_evaluator.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/evaluator/cce_evaluator.hpp)
- [`evaluator/trace_evaluator.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/evaluator/trace_evaluator.hpp)

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
汇总成 PTO 指令级代价
   |
   v
再汇总成整条 trace 的总代价
```

## 优点

第一，职责分离清晰。costmodel 逻辑集中在 [`include/pto/costmodel`](/home/lc/pto-isa-costmodel/include/pto/costmodel) 下，而不是分散到主 PTO common 实现中。这有利于维护，也能减少对正常 PTO 开发的干扰。

第二，开发效率更高。开发者可以在 host 上编译并观察 PTO lowering 行为，而不依赖真实设备运行时。这对于指令打通、轨迹检查和 costmodel 调优都很有价值。

第三，架构分层明确。实现被清晰分成四部分：

- host 运行时兼容
- 架构专属 intrinsic stub
- 轨迹记录
- 延迟评估

第四，保持了原有 PTO 使用方式。用户代码依然通过 `pto/pto-inst.hpp` 进入，因此 costmodel 是一个后端路径，而不是一个新的使用框架。

## 使用方式

推荐使用流程如下：

```text
1. include pto/pto-inst.hpp
2. 打开 __COSTMODEL 进行构建
3. 通过 __NPU_ARCH__ 选择目标架构
4. 在 host 上运行 PTO 代码
5. 采集 PTO + CCE trace
6. 将 trace 评估为 cycle 估算结果
```

当前相关文档和示例位于：

- [`README.md`](/home/lc/pto-isa-costmodel/include/pto/costmodel/README.md)
- [`a2a3/README.md`](/home/lc/pto-isa-costmodel/include/pto/costmodel/a2a3/README.md)
- [`a5/README.md`](/home/lc/pto-isa-costmodel/include/pto/costmodel/a5/README.md)
- [`tests/costmodel`](/home/lc/pto-isa-costmodel/tests/costmodel)

在实际使用中，这个后端主要用于回答以下问题：

- 某个 PTO 路径实际执行了什么
- 它发出了哪些 CCE 操作
- 哪些参数会影响这些操作
- 从轨迹可以推导出怎样的近似 cycle 代价

## 总结

`include/pto/costmodel` 应被理解为一个主机侧 PTO 轨迹与代价评估后端。它保留了标准 PTO 入口，通过 host 运行时替代层和可追踪的 intrinsic stub 实现 costmodel 行为，并提供了把 PTO 执行轨迹转换为架构相关 cycle 估算结果的结构化方法。
