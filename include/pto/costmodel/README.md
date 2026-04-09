# PTO Costmodel

`include/pto/costmodel` 是 PTO-ISA 的主机侧 trace-based cycle 评估后端。

当开启 `__COSTMODEL` 宏时，PTO 代码可以在标准 host 工具链上编译运行，costmodel 后端会：
- 替换 CCE intrinsic 为 trace-only stub
- 记录 PTO 指令和底层 CCE 调用的执行轨迹
- 基于架构规则估算每条 PTO 指令的 cycle 开销
- 将 cycle 结果绑定到目标 Tile，通过 `tile.GetCycle()` 查询

它不是数值模拟器。如需 host 侧数值行为，请使用 CPU 后端。

## 目录结构

```text
include/pto/costmodel/
├── runtime_stub.hpp       host 运行时总入口
├── pto_instr.hpp          PTO 指令的 costmodel 包装（MAP_INSTR_IMPL 宏）
├── trace.hpp              轨迹采集（PtoInstrScope、TraceState）
├── arch_config.hpp        架构规则配置
├── common/                host 运行时兼容层
│   ├── qualifiers.hpp     设备限定符替代
│   ├── aclrt_stub.hpp     ACL runtime 伪实现
│   ├── runtime_util.hpp   host helper shim
│   └── arch_select.hpp    架构选择分发
├── a2a3/
│   └── cce_stub.hpp       A2/A3 CCE 伪 stub
├── a5/
│   ├── cce_stub.hpp       A5 CCE 伪 stub
│   └── compat.hpp         A5 兼容层
└── evaluator/
    ├── cce_evaluator.hpp  CCE 调用级评估
    └── trace_evaluator.hpp PTO 指令级评估
```

## 架构选择

通过 `__NPU_ARCH__` 选择目标架构：

| 架构 | `__NPU_ARCH__` |
|------|----------------|
| A2/A3 | 2201 |
| A5 | 3101/3510 |

## 核心机制

### MAP_INSTR_IMPL 宏

每条 PTO 指令通过 `MAP_INSTR_IMPL` 宏包装，完成三步操作：
1. `PtoInstrScope` 记录 PTO 指令名（RAII）
2. 执行原始 `XXX_IMPL` 实现（CCE stub 记录底层调用）
3. `CaptureCycleIntoTile(dst)` 评估 trace 并将 cycle 写入目标 Tile

### 逐 Tile Cycle 存储

每条 PTO 指令执行后，cycle 结果存储在该指令的目标 Tile 中，通过 `tile.GetCycle()` 查询。这避免了全局状态错配，支持多指令混合执行的独立查询。

## 相关文档

- [方案说明（中文）](COSTMODEL_PROPOSAL.zh-CN.md) —— 完整的设计与实现说明
- [测试使用指南](../../../tests/costmodel/USER_GUIDE.zh-CN.md) —— 如何编写和运行 costmodel 测试

## 运行测试

```bash
./tests/run_costmodel_tests.sh
```

详见 [tests/costmodel/README.md](../../../tests/costmodel/README.md)。
