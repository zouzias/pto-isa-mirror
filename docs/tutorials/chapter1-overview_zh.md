# 第一章：总体概览介绍

本章介绍 PTO-ISA 的整体架构、目标用户、开发优势以及多框架支持。

---

## 1.1 PTO-ISA 概览

**PTO（Parallel Tile Operation）** 是昇腾 CANN 定义的一套面向 Tile 的虚拟 ISA。PTO-ISA 仓库提供 PTO Tile 指令的高性能实现与配套工具链：把算子/框架映射到 PTO 指令序列后，可以更平滑地在不同昇腾代际之间迁移与复用。

### 设计理念

昇腾硬件架构随代际演进发生了显著变化，导致指令集也产生了较大差异。PTO 指令集通过提升抽象层级来桥接这些差异：

- **跨平台兼容**：在固定 tile shape 下，PTO 指令能够跨平台正确工作并保持向后兼容
- **保留调优空间**：用户可以通过调整 tile size、tile shape、指令顺序等进行精细化优化
- **流水线控制**：对内部流水线具备足够控制力

### 指令集规模

PTO ISA 基于昇腾底层硬件与软件抽象，定义 **90+ 条标准 tile 指令**，涵盖：

| 类别 | 指令示例 | 功能 |
|------|----------|------|
| 数据移动 | TLOAD, TSTORE, TMOV | 在内存层次间移动 Tile |
| 向量计算 | TADD, TMUL, TEXP, TLOG | Tile 级算术运算 |
| 矩阵计算 | TMATMUL | Cube 单元矩阵乘法 |
| 规约操作 | TROWREDUCE, TCOLREDUCE | 行/列方向规约 |
| 填充操作 | TFILLPAD | 处理边界和对齐 |
| 同步机制 | set_flag, wait_flag | 流水线同步 |

### 软件栈位置

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层（用户代码）                          │
│         TileLang / PyPTO / PyTorch / TensorFlow             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      PTO-ISA（本项目）                        │
│            跨平台 Tile 操作指令集 + 高性能实现                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     昇腾 NPU 硬件                            │
│              Ascend 910B / 910C / 950 等                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 1.2 PTO-ISA 的目标用户

PTO Tile Lib **并不面向入门级用户**，主要面向以下开发者群体：

### 框架后端开发者

- 直接对接昇腾硬件的框架后端开发
- 将上层框架（TileLang、PyPTO、PyTorch）映射到 PTO 指令
- 开发编译器 Pass 和代码生成器

### 跨平台应用开发者

- 需要在不同昇腾芯片代际间迁移算子
- 开发可移植的高性能 kernel
- 避免因硬件升级而重写代码

### 高性能算子开发者

- 手工实现算子/内核
- 追求极致性能的融合算子开发
- 需要精确控制内存布局和流水线

---

## 1.3 PTO-ISA 开发融合算子的优势

### 优势一：细粒度内存控制

PTO-ISA 让开发者显式控制 Tile 在片上内存的位置和数据搬运路径：

```cpp
// 示例：来自 demos/baseline/add 的实际代码
constexpr unsigned X_PING = 0x0;                       // ping address of input x in UB buffer
constexpr unsigned X_PONG = (X_PING + 0x8000 + 0x100); // pong address of input x in UB buffer
constexpr unsigned Y_PING = 0x10000;                   // ping address of input y in UB buffer

// 分配 UB 内存
TASSIGN(xTiles[0], X_PING);
TASSIGN(xTiles[1], X_PONG);
TASSIGN(yTiles[0], Y_PING);
```

**优势：**
- 消除不必要的数据拷贝
- 实现乒乓（ping-pong）缓冲优化
- 精确控制缓冲区布局

### 优势二：显式流水线同步

PTO-ISA 的 `set_flag/wait_flag` 机制让流水线并行变得透明可控：

```cpp
// 示例：来自 demos/baseline/add 的实际代码
// 流水线同步初始化
set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

for (uint32_t i = 0; i < loopCount; i++) {
    wait_flag(PIPE_V, PIPE_MTE2, (event_t)(pingpong_flag));
    // 加载数据
    TLOAD(xTiles[pingpong_flag], xGlobal);
    TLOAD(yTiles[pingpong_flag], yGlobal);
    
    set_flag(PIPE_MTE2, PIPE_V, (event_t)(pingpong_flag));
    wait_flag(PIPE_MTE2, PIPE_V, (event_t)(pingpong_flag));
    
    // 执行计算
    TADD(zTiles[pingpong_flag], xTiles[pingpong_flag], yTiles[pingpong_flag]);
    
    // 存储结果
    TSTORE(zGlobal, zTiles[pingpong_flag]);
    
    pingpong_flag = (pingpong_flag == 0) ? 1 : 0;
}
```

**优势：**
- 最大化硬件流水线利用率
- 支持乒乓调优模式
- 避免不必要的同步屏障

### 优势三：跨平台可移植性

同一套 PTO 代码可在不同昇腾芯片上运行：

| 芯片代号 | 芯片名称 | UB 大小 | L0C 大小 | 代码兼容 |
|----------|----------|---------|----------|----------|
| A2 | Ascend910B | 192KB | 128KB | ✅ |
| A3 | Ascend910C | 192KB | 128KB | ✅ |
| A5 | Ascend950 | 256KB | 256KB | ✅ |

PTO-ISA 抽象了硬件差异，通过模板参数适配不同平台。

### 优势四：经过验证的高性能

本仓库包含面向性能的 kernels，并给出参考测量数据：

**GEMM 性能（A3，24 核）：**

| 参数 | TMATMUL 占比 | 执行时间 |
|------|--------------|----------|
| m=k=n=1536 | 54.5% | 0.0388 ms |
| m=k=n=3072 | 79.0% | 0.2067 ms |
| m=k=n=6144 | 86.7% | 1.5060 ms |

详见：[高性能 GEMM 算子示例](https://gitcode.com/cann/pto-isa/blob/master/kernels/manual/a2a3/gemm_performance/README_zh.md)

---

## 1.4 多框架支持

PTO-ISA 支持多种上层框架，用户可根据需求进行选择：

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户选择                                  │
│                                                                 │
│  ┌─────────────────┐  ┌───────────────┐  ┌───────────────────┐ │
│  │    TileLang     │  │    PyPTO      │  │     PyTorch       │ │
│  │  (Tile DSL)     │  │  (Python IR)  │  │   (框架集成)       │ │
│  └────────┬────────┘  └───────┬───────┘  └─────────┬─────────┘ │
│           │                   │                    │            │
│           └───────────────────┼────────────────────┘            │
│                               │                                 │
│                               ▼                                 │
│                   ┌───────────────────────┐                     │
│                   │       PTO-ISA         │                     │
│                   │   (统一指令集后端)     │                     │
│                   └───────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### TileLang Ascend

**仓库：** https://github.com/tile-ai/tilelang-ascend/

**特点：**
- Python DSL，类似 Triton 的 Tile 编程风格
- 自动处理同步和内存管理
- 适合快速原型开发

### PyPTO

**仓库：** https://gitcode.com/cann/pypto/

**特点：**
- Python 级 Tile IR，直接映射到 PTO 指令
- 提供丰富的样例代码（00_hello_world 到 03_advanced）
- 适合编译器开发和精确控制

```python
# PyPTO Hello World 示例
@pypto.frontend.jit(runtime_options={"run_mode": mode})
def add_kernel(
    x: pypto.Tensor([...], pypto.DT_FP32),
    y: pypto.Tensor([...], pypto.DT_FP32),
    out: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    out[:] = x + y
```

### PyTorch 集成

**特点：**
- 通过 `torch_npu` 暴露为 PyTorch 算子
- 使用 `TORCH_LIBRARY` 注册算子 schema
- 兼容现有 PyTorch 工作流

```python
# 使用示例
import torch
import torch_npu
z = torch.ops.npu.my_add(x, y)
```

### 框架选择指南

| 需求 | 推荐框架 | 理由 |
|------|----------|------|
| 快速开发验证 | TileLang | 语法简洁，自动同步 |
| 精确控制底层 | PyPTO | 直接映射 PTO IR |
| 生产环境部署 | PyTorch 集成 | 与现有工作流兼容 |
| 极致性能调优 | PTO-ISA 原生 | 完全控制流水线 |

---

## 1.5 本章小结

本章介绍了：

- **PTO-ISA** 是昇腾 CANN 定义的面向 Tile 的虚拟 ISA，定义 90+ 条标准指令
- **目标用户** 包括框架后端开发者、跨平台应用开发者、高性能算子开发者
- **开发优势** 包括细粒度内存控制、显式流水线同步、跨平台可移植性、经过验证的高性能
- **多框架支持** 包括 TileLang、PyPTO、PyTorch 集成，用户可根据需求选择

### 下一步

- **第二章**：基于 PTO-ISA 实现 ADD 算子 — 完整的开发流程和多框架对接

---

> 📖 继续阅读 [第二章：基于 PTO-ISA 实现 ADD 算子](chapter2-tadd-kernel_zh.md)
