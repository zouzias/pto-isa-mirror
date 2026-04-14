# 第二章：基于 PTO-ISA 实现 ADD 算子

本章详细介绍如何使用 PTO-ISA 实现 ADD 算子，并对接 TileLang、PyPTO 和 PyTorch 框架。

---

## 2.1 环境搭建

### 2.1.1 获取代码

```bash
git clone https://gitcode.com/cann/pto-isa.git
cd pto-isa
```

### 2.1.2 CPU 模拟器环境（推荐新手）

CPU 模拟器是最简单的入门方式，无需专用硬件。

**前置条件：**
- Git
- Python >= 3.8（推荐 3.10+）
- CMake >= 3.16
- 支持 C++20 的编译器：
  - Linux: GCC 13+ 或 Clang 15+
  - macOS: Xcode/AppleClang
  - Windows: Visual Studio 2022

**Linux (Ubuntu 20.04+)：**

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3 python3-pip python3-venv git
```

**创建虚拟环境：**

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

pip install numpy
```

**构建并运行测试：**

```bash
python3 tests/run_cpu.py
```

### 2.1.3 NPU 环境（高级）

**前置条件：**
- Linux 系统
- Ascend 910B/910C 硬件或 CANN 模拟器
- CANN Toolkit 9.0+

**配置 CANN 环境：**

```bash
source /usr/local/Ascend/cann/set_env.sh
```

**运行 NPU 测试：**

```bash
# 在 A3 模拟器上运行
python3 tests/script/run_st.py -r sim -v a3 -t tadd

# 在 NPU 硬件上运行
python3 tests/script/run_st.py -r npu -v a3 -t tadd
```

---

## 2.2 TADD Kernel 代码详解

### 2.2.1 完整内核代码

```cpp
// tadd_kernel.cpp
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTADD(
    __gm__ T __out__ *out,
    __gm__ T __in__ *src0,
    __gm__ T __in__ *src1)
{
    // 1. 定义全局张量类型
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    
    // 2. 定义 Tile 类型
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, 
                          BLayout::RowMajor, -1, -1>;
    
    // 3. 创建 Tile 实例
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    
    // 4. 分配 UB 内存
    TASSIGN(src0Tile, 0x00000);  // UB 偏移 0
    TASSIGN(src1Tile, 0x10000);  // UB 偏移 64KB
    TASSIGN(dstTile,  0x20000);  // UB 偏移 128KB
    
    // 5. 包装全局内存指针
    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    // 6. 声明流水线同步事件
    Event<Op::TLOAD, Op::TADD> event0;
    Event<Op::TADD, Op::TSTORE_VEC> event1;

    // 7. 执行计算流程
    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TADD(dstTile, src0Tile, src1Tile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    
    out = dstGlobal.data();
}

// 主机端启动函数
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTADD(T *out, T *src0, T *src1, void *stream) {
    runTADD<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
}

// 显式实例化
template void LaunchTADD<float, 64, 64, 64, 64>(float*, float*, float*, void*);
```

### 2.2.2 代码逐段解析

**类型定义：**

```cpp
// 5 维形状：[batch, channel, depth, height, width]
using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;

// 步长：定义内存布局
using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;

// Tile 类型：Vec 表示存储在 UB
using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, 
                      BLayout::RowMajor, -1, -1>;
```

**内存分配：**

```cpp
// TASSIGN 将 Tile 映射到 UB 的特定偏移
TASSIGN(src0Tile, 0x00000);  // 64×64×4 = 16KB
TASSIGN(src1Tile, 0x10000);  // 64KB 偏移，避免重叠
TASSIGN(dstTile,  0x20000);  // 128KB 偏移
```

**关键点：**
- 偏移量需要手动计算，确保 Tile 不重叠
- 32 字节对齐可获得最佳性能

**流水线同步：**

```cpp
Event<Op::TLOAD, Op::TADD> event0;      // 加载→计算依赖
Event<Op::TADD, Op::TSTORE_VEC> event1; // 计算→存储依赖

event0 = TLOAD(src1Tile, src1Global);   // 返回事件句柄
event1 = TADD(dstTile, src0Tile, src1Tile, event0);  // 等待 event0
TSTORE(dstGlobal, dstTile, event1);     // 等待 event1
```

### 2.2.3 调优建议

**建议一：Tile 大小选择**

```cpp
// 较小 Tile：更多并行度，更多内核启动开销
using TileSmall = Tile<TileType::Vec, float, 32, 32, ...>;

// 较大 Tile：更好的内存局部性，更少的边界处理
using TileLarge = Tile<TileType::Vec, float, 128, 128, ...>;

// 推荐：根据 UB 大小和数据量选择
// A2A3: UB = 192KB，建议 Tile 总大小 < 150KB
```

**建议二：内存对齐**

```cpp
// 列数对齐到 32 字节
// float (4B): 列数应为 8 的倍数
// half (2B): 列数应为 16 的倍数
// int8 (1B): 列数应为 32 的倍数

// 好的选择
using TileAligned = Tile<TileType::Vec, float, 64, 64, ...>;   // 64 × 4 = 256B ✓

// 避免
using TileUnaligned = Tile<TileType::Vec, float, 64, 63, ...>; // 63 × 4 = 252B ✗
```

**建议三：流水线重叠**

```cpp
// 单流水线（简单但效率低）
TLOAD(tile1, src1);
TLOAD(tile2, src2);
TADD(dst, tile1, tile2);
TSTORE(output, dst);

// 双缓冲（更高效）
for (int i = 0; i < N; i += 2) {
    // 加载 i+1 的同时计算 i
    TLOAD(tileA[1], src[i+1]);
    TADD(dstA, tileA[0], tileB[0]);
    
    // 加载 i+2 的同时计算 i+1
    TLOAD(tileA[0], src[i+2]);
    TADD(dstB, tileA[1], tileB[1]);
}
```

---

## 2.3 TileLang 框架对接

### 2.3.1 TileLang 环境搭建

**安装 TileLang：**

```bash
# 创建虚拟环境
python3 -m venv tilelang-env
source tilelang-env/bin/activate

# 安装 TileLang
pip install tilelang

# 安装昇腾后端支持
pip install tilelang-ascend
```

**验证安装：**

```python
import tilelang as tl
print(tl.__version__)
```

### 2.3.2 TileLang 对接代码详解

```python
# tadd_tilelang.py
import tilelang as tl
from tilelang import T

@tl.jit(out_idx=[2])
def tadd_kernel(M: int, N: int, dtype: T.dtype = T.float16):
    """
    TileLang 实现的 TADD 内核
    
    参数:
        M: 行数
        N: 列数
        dtype: 数据类型
    """
    # 定义块大小（与 PTO Tile 大小对应）
    BLOCK_M = 64
    BLOCK_N = 64
    
    @T.prim_func
    def main(
        X: T.Tensor([M, N], dtype),      # 输入张量 A
        Y: T.Tensor([M, N], dtype),      # 输入张量 B
        Z: T.Tensor([M, N], dtype),      # 输出张量 C
    ):
        # 启动 2D 网格，每个块处理 BLOCK_M × BLOCK_N 的 Tile
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(N, BLOCK_N), threads=128) as (bx, by):
            # 分配片上缓冲区（对应 PTO 的 TASSIGN）
            x_tile = T.alloc_fragment([BLOCK_M, BLOCK_N], dtype)
            y_tile = T.alloc_fragment([BLOCK_M, BLOCK_N], dtype)
            z_tile = T.alloc_fragment([BLOCK_M, BLOCK_N], dtype)
            
            # 计算当前块的全局偏移
            row_start = bx * BLOCK_M
            col_start = by * BLOCK_N
            
            # 加载数据（对应 PTO 的 TLOAD）
            T.copy(X[row_start:row_start + BLOCK_M, col_start:col_start + BLOCK_N], x_tile)
            T.copy(Y[row_start:row_start + BLOCK_M, col_start:col_start + BLOCK_N], y_tile)
            
            # 逐元素加法（对应 PTO 的 TADD）
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                z_tile[i, j] = x_tile[i, j] + y_tile[i, j]
            
            # 存储结果（对应 PTO 的 TSTORE）
            T.copy(z_tile, Z[row_start:row_start + BLOCK_M, col_start:col_start + BLOCK_N])
    
    return main
```

**代码映射关系：**

| TileLang | PTO-ISA | 说明 |
|----------|---------|------|
| `T.alloc_fragment` | `TASSIGN` | 分配片上缓冲区 |
| `T.copy(src, dst)` | `TLOAD/TSTORE` | 数据搬运 |
| `T.Parallel` | 向量化循环 | 并行计算 |
| `+` 运算符 | `TADD` | 逐元素加法 |

### 2.3.3 编译和运行

```python
# run_tadd_tilelang.py
import numpy as np
import tilelang as tl

# 导入内核
from tadd_tilelang import tadd_kernel

def main():
    # 参数设置
    M, N = 1024, 1024
    dtype = np.float16
    
    # 创建输入数据
    x = np.random.randn(M, N).astype(dtype)
    y = np.random.randn(M, N).astype(dtype)
    
    # 编译内核
    kernel = tadd_kernel(M, N, tl.float16)
    
    # 执行内核
    z = kernel(x, y)
    
    # 验证结果
    z_ref = x + y
    max_diff = np.abs(z - z_ref).max()
    print(f"Max difference: {max_diff}")
    assert max_diff < 1e-3, "Verification failed!"
    print("Verification passed!")

if __name__ == "__main__":
    main()
```

**运行：**

```bash
python run_tadd_tilelang.py
```

---

## 2.4 PyPTO 框架对接

### 2.4.1 PyPTO 环境搭建

**安装 PyPTO：**

```bash
# 克隆 PyPTO 仓库
git clone https://gitcode.com/cann/pypto.git
cd pypto

# 创建虚拟环境
python3 -m venv pypto-env
source pypto-env/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装 PyPTO
pip install -e .
```

**配置 CANN 环境：**

```bash
source /usr/local/Ascend/cann/set_env.sh
```

### 2.4.2 PyPTO 对接代码详解

```python
# tadd_pypto.py
import pypto as pl
from pypto import MemorySpace

# 定义动态维度
M = pl.dynamic("M")
N = pl.dynamic("N")

@pl.function(type=pl.FunctionType.InCore)
def tadd_kernel(
    a: pl.Tensor[[M, N], pl.FP32],      # 输入张量 A
    b: pl.Tensor[[M, N], pl.FP32],      # 输入张量 B
    output: pl.Out[pl.Tensor[[M, N], pl.FP32]],  # 输出张量 C
):
    """
    PyPTO 实现的 TADD 内核
    
    这段代码直接映射到 PTO IR，然后编译为 PTO-ISA 指令。
    """
    # 定义 Tile 大小
    TILE_M = 64
    TILE_N = 64
    
    # 获取实际维度
    actual_M = pl.tensor.dim(a, 0)
    actual_N = pl.tensor.dim(a, 1)
    
    # 分块处理
    for i in pl.range(0, actual_M, TILE_M):
        for j in pl.range(0, actual_N, TILE_N):
            # 计算当前 Tile 的有效大小
            tile_m = pl.min(TILE_M, actual_M - i)
            tile_n = pl.min(TILE_N, actual_N - j)
            
            # 加载 Tile 到 UB（对应 TLOAD + TASSIGN）
            tile_a = pl.tile.load(
                a, 
                offsets=[i, j], 
                shapes=[TILE_M, TILE_N],
                valid_shapes=[tile_m, tile_n],
                target_memory=MemorySpace.Vec
            )
            
            tile_b = pl.tile.load(
                b, 
                offsets=[i, j], 
                shapes=[TILE_M, TILE_N],
                valid_shapes=[tile_m, tile_n],
                target_memory=MemorySpace.Vec
            )
            
            # 逐元素加法（对应 TADD）
            tile_c = pl.tile.add(tile_a, tile_b)
            
            # 存储结果（对应 TSTORE）
            pl.tile.store(tile_c, offsets=[i, j], tensor=output)
```

**PyPTO 特有概念：**

| 概念 | 说明 |
|------|------|
| `pl.dynamic("M")` | 声明运行时确定的维度 |
| `MemorySpace.Vec` | 指定 Tile 存储在 UB |
| `valid_shapes` | 处理边界情况（Tile 部分有效） |
| `pl.tile.load/store` | 显式的数据搬运操作 |

### 2.4.3 编译和运行

```python
# run_tadd_pypto.py
import numpy as np
import pypto as pl
from tadd_pypto import tadd_kernel

def main():
    # 编译内核
    compiled_kernel = pl.compile(
        tadd_kernel,
        target="ascend",  # 目标平台
        arch="a3",        # A3 架构（910B/910C）
    )
    
    # 创建输入数据
    M, N = 1024, 1024
    a = np.random.randn(M, N).astype(np.float32)
    b = np.random.randn(M, N).astype(np.float32)
    output = np.zeros((M, N), dtype=np.float32)
    
    # 执行内核
    compiled_kernel(a, b, output, M=M, N=N)
    
    # 验证结果
    expected = a + b
    max_diff = np.abs(output - expected).max()
    print(f"Max difference: {max_diff}")
    assert max_diff < 1e-5, "Verification failed!"
    print("Verification passed!")

if __name__ == "__main__":
    main()
```

**运行：**

```bash
python run_tadd_pypto.py
```

---

## 2.5 PyTorch 框架对接

### 2.5.1 PyTorch 环境搭建

**安装 PyTorch 和昇腾支持：**

```bash
# 创建虚拟环境
python3 -m venv pytorch-npu-env
source pytorch-npu-env/bin/activate

# 安装 PyTorch
pip install torch==2.1.0

# 安装昇腾 PyTorch 插件
pip install torch-npu
```

**配置环境：**

```bash
source /usr/local/Ascend/cann/set_env.sh
```

**验证安装：**

```python
import torch
import torch_npu

print(f"PyTorch version: {torch.__version__}")
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU device count: {torch.npu.device_count()}")
```

### 2.5.2 PyTorch 对接代码详解

**方式一：使用 PTO 自定义算子**

```python
# pto_add_op.py
import torch
import torch_npu
from torch.utils.cpp_extension import load

# 加载 PTO 算子库（假设已编译）
pto_ops = load(
    name='pto_ops',
    sources=['pto_add_cuda.cpp', 'pto_add_kernel.cpp'],
    extra_include_paths=['/path/to/pto-isa/include'],
    extra_ldflags=['-L/path/to/pto-isa/lib', '-lpto'],
)

class PTOAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        """前向传播：调用 PTO TADD 内核"""
        # 确保输入在 NPU 上
        assert x.is_npu and y.is_npu
        
        # 调用 PTO 算子
        output = pto_ops.pto_add(x, y)
        
        # 保存反向传播需要的信息
        ctx.save_for_backward(x, y)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：加法的梯度直接传递"""
        return grad_output.clone(), grad_output.clone()

# 创建便捷函数
def pto_add(x, y):
    return PTOAddFunction.apply(x, y)
```

**方式二：使用 torch.compile 后端**

```python
# compile_backend.py
import torch
import torch_npu

# 注册昇腾后端
@torch.compile(backend="ascend")
def add_with_compile(x, y):
    """通过 torch.compile 自动优化的加法"""
    return x + y

def main():
    # 创建 NPU 张量
    device = torch.device("npu:0")
    x = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    y = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    
    # 执行编译后的函数
    z = add_with_compile(x, y)
    
    # 验证
    z_ref = x + y
    print(f"Max diff: {(z - z_ref).abs().max().item()}")

if __name__ == "__main__":
    main()
```

**方式三：直接使用 torch_npu**

```python
# direct_npu.py
import torch
import torch_npu

def main():
    # 设置设备
    device = torch.device("npu:0")
    
    # 创建张量
    x = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    y = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    
    # 执行加法（自动使用优化的 NPU 实现）
    z = x + y
    
    # 同步并验证
    torch.npu.synchronize()
    
    # 转回 CPU 验证
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    z_cpu = z.cpu()
    z_ref = x_cpu + y_cpu
    
    max_diff = (z_cpu - z_ref).abs().max().item()
    print(f"Max difference: {max_diff}")
    assert max_diff < 1e-5, "Verification failed!"
    print("Verification passed!")

if __name__ == "__main__":
    main()
```

### 2.5.3 编译和运行

```bash
# 运行直接 NPU 方式
python direct_npu.py

# 运行 torch.compile 方式
python compile_backend.py
```

---

## 2.6 性能对比与最佳实践

### 2.6.1 性能对比

| 框架 | 开发效率 | 性能控制 | 学习曲线 | 适用场景 |
|------|----------|----------|----------|----------|
| **PTO-ISA 原生** | 低 | 最高 | 陡峭 | 极致性能优化 |
| **TileLang** | 高 | 高 | 中等 | 快速原型开发 |
| **PyPTO** | 中 | 高 | 中等 | 编译器开发 |
| **PyTorch 集成** | 最高 | 中 | 平缓 | 生产部署 |

### 2.6.2 最佳实践

**1. 选择合适的抽象层级**

```
性能关键路径 → PTO-ISA 原生或 PyPTO
快速验证想法 → TileLang
生产模型部署 → PyTorch 集成
```

**2. Tile 大小选择原则**

- 尽量使用 2 的幂次方（32, 64, 128）
- 考虑 UB 大小限制（A2A3: 192KB）
- 平衡并行度和内存局部性

**3. 数据类型选择**

| 类型 | 精度 | 性能 | 推荐场景 |
|------|------|------|----------|
| FP32 | 高 | 基准 | 训练、高精度推理 |
| FP16 | 中 | 2x | 推理、混合精度训练 |
| BF16 | 中 | 2x | 大模型训练 |
| INT8 | 低 | 4x | 量化推理 |

---

## 2.7 本章小结

本章介绍了：

1. **环境搭建**：CPU 模拟器和 NPU 环境配置
2. **TADD 内核**：完整的 PTO-ISA 实现和调优建议
3. **TileLang 对接**：Python DSL 风格的实现
4. **PyPTO 对接**：直接映射 PTO IR 的实现
5. **PyTorch 对接**：三种集成方式

### 下一步

- 尝试实现更复杂的融合算子（如 Softmax、LayerNorm）
- 探索矩阵乘法（TMATMUL）的实现
- 学习多核并行的调度策略

---

> 📖 继续阅读 [第三章：矩阵乘法与 Cube 单元](chapter3-matmul_zh.md)
