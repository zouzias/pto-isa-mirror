# 第二章：基于 PTO-ISA 实现 ADD 算子

本章详细介绍如何使用 PTO-ISA 实现 ADD 算子，并对接 TileLang、PyPTO 和 PyTorch 框架。

---

## 2.1 环境搭建

参考 PTO-ISA 代码仓中的 [getting-started_zh.md](https://gitcode.com/cann/pto-isa/blob/master/docs/getting-started_zh.md) 完成环境搭建。

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
  - Linux: GCC 13+ 或 Clang 15+（GCC >= 14 启用 bfloat16 支持）
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

### 2.1.3 NPU 环境

**前置条件：**
- Linux 系统
- Ascend 910B/910C 硬件或 CANN 模拟器
- CANN Toolkit（请参考 [msprof 工具](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/Profiling/atlasprofiling_16_0010.html)）

**配置 CANN 环境：**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**设置目标 SoC：**

在目标机器上执行 `npu_smi info` 查询芯片名称，并按 `Ascend<Chip Name>` 的形式填写。

---

## 2.2 TADD Kernel 代码详解

参考 `demos/baseline/add` 和 `demos/torch_jit/add` 中的实际代码。

### 2.2.1 完整内核代码

以下代码来自 `demos/baseline/add/csrc/kernel/add_custom.cpp`：

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// ==================== 常量定义 ====================
constexpr uint32_t BLOCK_DIM = 20;                     // AIV 核心数量
constexpr unsigned BLOCK_ROWS = 20;                    // 行方向 AIV 数量
constexpr unsigned BLOCK_COLS = 1;                     // 列方向 AIV 数量
constexpr uint32_t BUFFER_NUM = 2;                     // 乒乓缓冲区数量
constexpr unsigned UB_SIZE = 0x30000;                  // 192KB UB (A2A3)

// ==================== UB 地址分配（乒乓缓冲） ====================
constexpr unsigned X_PING = 0x0;                       // 输入 x 的 ping 地址
constexpr unsigned X_PONG = (X_PING + 0x8000 + 0x100); // 输入 x 的 pong 地址
constexpr unsigned Y_PING = 0x10000;                   // 输入 y 的 ping 地址
constexpr unsigned Y_PONG = (Y_PING + 0x8000 + 0x100); // 输入 y 的 pong 地址
constexpr unsigned Z_PING = 0x20000;                   // 输出 z 的 ping 地址
constexpr unsigned Z_PONG = (Z_PING + 0x8000 + 0x100); // 输出 z 的 pong 地址

constexpr unsigned MAX_TILE_SIZE = (0x10000 - 0x100);  // 单个 Tile 最大尺寸
constexpr uint32_t tileNum = 2;                        // 每个向量核的 Tile 数量

template <typename T, unsigned tileRows, unsigned tileCols>
AICORE void runTAdd(__gm__ T *z, __gm__ T *x, __gm__ T *y, uint32_t totalLength)
{
    set_mask_norm();
    set_vector_mask(-1, -1);
    
    // ==================== 多核分块 ====================
    // 核间分块
    constexpr unsigned bTileRows = tileRows / BLOCK_ROWS;
    constexpr unsigned bTileCols = tileCols / BLOCK_COLS;
    static_assert(bTileRows * bTileCols * sizeof(T) <= MAX_TILE_SIZE, "UB buffer overflow.");

    // 核内分块
    constexpr unsigned tileSRows = bTileRows;
    constexpr unsigned tileSCols = bTileCols / tileNum / BUFFER_NUM;
    
    // ==================== 定义 GlobalTensor ====================
    using ShapeDim5 = pto::Shape<1, 1, 1, tileSRows, tileSCols>;
    using StridDim5 = pto::Stride<1, 1, 1, tileCols, 1>;
    using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;
    GlobalData xGlobal(x);
    GlobalData yGlobal(y);
    GlobalData zGlobal(z);

    // ==================== 定义 Tile（乒乓缓冲） ====================
    using TileData = Tile<TileType::Vec, T, tileSRows, tileSCols, BLayout::RowMajor, -1, -1>;
    unsigned bLength = totalLength / block_num;
    unsigned vRows = tileRows / block_num;
    unsigned vCols = bLength / tileNum / BUFFER_NUM;
    
    // 为每个缓冲区创建 Tile
    TileData xTiles[BUFFER_NUM] = {TileData(vRows, vCols), TileData(vRows, vCols)};
    TileData yTiles[BUFFER_NUM] = {TileData(vRows, vCols), TileData(vRows, vCols)};
    TileData zTiles[BUFFER_NUM] = {TileData(vRows, vCols), TileData(vRows, vCols)};

    // ==================== 分配 UB 地址 ====================
    TASSIGN(xTiles[0], X_PING);
    TASSIGN(xTiles[1], X_PONG);
    TASSIGN(yTiles[0], Y_PING);
    TASSIGN(yTiles[1], Y_PONG);
    TASSIGN(zTiles[0], Z_PING);
    TASSIGN(zTiles[1], Z_PONG);
    
    int32_t loopCount = tileNum * BUFFER_NUM;
    unsigned offset = block_idx * bTileRows * bTileCols;
    int8_t pingpong_flag = 0;

    // ==================== 流水线同步初始化 ====================
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    
    // ==================== 主循环（乒乓调优） ====================
    for (uint32_t i = 0; i < loopCount; i++) {
        unsigned iterOffset = offset + i * tileSRows * tileSCols;
        TASSIGN(xGlobal, x + iterOffset);
        TASSIGN(yGlobal, y + iterOffset);
        TASSIGN(zGlobal, z + iterOffset);

        // 等待 Vector 完成，开始加载
        wait_flag(PIPE_V, PIPE_MTE2, (event_t)(pingpong_flag));
        TLOAD(xTiles[pingpong_flag], xGlobal);
        TLOAD(yTiles[pingpong_flag], yGlobal);

        // 加载完成，通知 Vector
        set_flag(PIPE_MTE2, PIPE_V, (event_t)(pingpong_flag));
        wait_flag(PIPE_MTE2, PIPE_V, (event_t)(pingpong_flag));

        // 等待存储完成，开始计算
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)(pingpong_flag));
        TADD(zTiles[pingpong_flag], xTiles[pingpong_flag], yTiles[pingpong_flag]);
        set_flag(PIPE_V, PIPE_MTE2, (event_t)(pingpong_flag));

        // 计算完成，开始存储
        set_flag(PIPE_V, PIPE_MTE3, (event_t)(pingpong_flag));
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)(pingpong_flag));
        TSTORE(zGlobal, zTiles[pingpong_flag]);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)(pingpong_flag));
        
        // 切换乒乓缓冲区
        pingpong_flag = (pingpong_flag == 0) ? 1 : 0;
    }
    
    // ==================== 等待所有操作完成 ====================
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

// Kernel 入口
__global__ AICORE void add_custom(__gm__ void *x, __gm__ void *y, __gm__ void *z, uint32_t totalLength)
{
    constexpr unsigned tileRows = 20;
    constexpr unsigned tileCols = 2048;
    runTAdd<half, tileRows, tileCols>((__gm__ half *)z, (__gm__ half *)x, (__gm__ half *)y, totalLength);
}
```

### 2.2.2 调优建议

#### 乒乓调优（Ping-Pong Buffering）

乒乓缓冲是隐藏数据搬运延迟的关键技术：

```
时间线：
  Buffer[0]: [LOAD] [    COMPUTE    ] [STORE]
  Buffer[1]:        [LOAD] [    COMPUTE    ] [STORE]
                          ↑
                    计算和加载重叠
```

**关键点：**
- 使用 `BUFFER_NUM = 2` 创建双缓冲
- 使用 `pingpong_flag` 在两个缓冲区间切换
- 通过 `set_flag/wait_flag` 确保正确的依赖关系

#### Tiling 策略

**多级分块：**
1. **核间分块**：将数据分配到多个 AIV 核心
2. **核内分块**：每个核心内再细分为多个 Tile
3. **乒乓分块**：每个核心使用双缓冲

```cpp
// 核间分块
constexpr unsigned bTileRows = tileRows / BLOCK_ROWS;  // 20 / 20 = 1
constexpr unsigned bTileCols = tileCols / BLOCK_COLS;  // 2048 / 1 = 2048

// 核内分块
constexpr unsigned tileSRows = bTileRows;              // 1
constexpr unsigned tileSCols = bTileCols / tileNum / BUFFER_NUM;  // 2048 / 2 / 2 = 512
```

**Tile 大小选择原则：**
- 不超过 UB 容量（A2A3: 192KB）
- 列数对齐到 32 字节
- 平衡并行度和内存效率

---

## 2.3 TileLang 框架对接

### 2.3.1 TileLang 环境搭建

**安装 TileLang Ascend：**

```bash
# 克隆仓库
git clone https://github.com/tile-ai/tilelang-ascend.git
cd tilelang-ascend

# 创建虚拟环境
python3 -m venv tilelang-env
source tilelang-env/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装 TileLang
pip install -e .
```

**配置 CANN 环境：**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
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
    
    映射关系：
    - T.alloc_fragment → TASSIGN
    - T.copy → TLOAD/TSTORE
    - 逐元素操作 → TADD
    """
    BLOCK_M = 64
    BLOCK_N = 64
    
    @T.prim_func
    def main(
        X: T.Tensor([M, N], dtype),
        Y: T.Tensor([M, N], dtype),
        Z: T.Tensor([M, N], dtype),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(N, BLOCK_N), threads=128) as (bx, by):
            # 分配片上缓冲区
            x_tile = T.alloc_fragment([BLOCK_M, BLOCK_N], dtype)
            y_tile = T.alloc_fragment([BLOCK_M, BLOCK_N], dtype)
            z_tile = T.alloc_fragment([BLOCK_M, BLOCK_N], dtype)
            
            row_start = bx * BLOCK_M
            col_start = by * BLOCK_N
            
            # 加载数据
            T.copy(X[row_start:row_start + BLOCK_M, col_start:col_start + BLOCK_N], x_tile)
            T.copy(Y[row_start:row_start + BLOCK_M, col_start:col_start + BLOCK_N], y_tile)
            
            # 逐元素加法
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                z_tile[i, j] = x_tile[i, j] + y_tile[i, j]
            
            # 存储结果
            T.copy(z_tile, Z[row_start:row_start + BLOCK_M, col_start:col_start + BLOCK_N])
    
    return main
```

### 2.3.3 编译和运行

```python
# run_tilelang.py
import numpy as np
import tilelang as tl
from tadd_tilelang import tadd_kernel

def main():
    M, N = 1024, 1024
    dtype = np.float16
    
    x = np.random.randn(M, N).astype(dtype)
    y = np.random.randn(M, N).astype(dtype)
    
    kernel = tadd_kernel(M, N, tl.float16)
    z = kernel(x, y)
    
    z_ref = x + y
    max_diff = np.abs(z - z_ref).max()
    print(f"Max difference: {max_diff}")
    assert max_diff < 1e-3, "Verification failed!"
    print("TileLang TADD test passed!")

if __name__ == "__main__":
    main()
```

**运行：**

```bash
python run_tilelang.py
```

---

## 2.4 PyPTO 框架对接

参考 https://gitcode.com/cann/pypto 代码仓的样例。

### 2.4.1 PyPTO 环境搭建

**安装 PyPTO：**

```bash
# 克隆仓库
git clone https://gitcode.com/cann/pypto.git
cd pypto

# 创建虚拟环境
python3 -m venv pypto-env
source pypto-env/bin/activate

# 安装依赖（参考 docs/install/prepare_environment.md）
pip install -r requirements.txt

# 安装 PyPTO（参考 docs/install/build_and_install.md）
pip install -e .
```

**配置环境变量（NPU 模式）：**

```bash
# 配置 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 2.4.2 PyPTO 对接代码详解

以下代码参考 `pypto/examples/00_hello_world/hello_world.py`：

```python
# tadd_pypto.py
import os
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


def create_add_kernel(shape: tuple, run_mode: str = "npu"):
    """
    创建 PyPTO ADD 内核
    
    参数：
        shape: 输入张量形状
        run_mode: "npu" 或 "sim"（模拟器模式）
    """
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def add_kernel(
        x: pypto.Tensor([...], pypto.DT_FP32),
        y: pypto.Tensor([...], pypto.DT_FP32),
        out: pypto.Tensor([...], pypto.DT_FP32),
    ):
        # 设置 Tile 形状
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        # 执行加法
        out[:] = x + y

    return add_kernel


def test_add(device_id=None, run_mode: str = "npu"):
    """
    测试 ADD 内核
    """
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (1, 4, 1, 64)
    
    # 准备数据
    input_data0 = torch.rand(shape, dtype=torch.float, device=device)
    input_data1 = torch.rand(shape, dtype=torch.float, device=device)
    output_data = torch.empty(shape, dtype=torch.float32, device=device)
    
    # 执行内核
    create_add_kernel(shape, run_mode)(input_data0, input_data1, output_data)
    
    # 验证结果
    golden = torch.add(input_data0, input_data1)
    max_diff = np.abs(output_data.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Max difference: {max_diff:.6f}")
    
    if run_mode == "npu":
        assert_allclose(np.array(output_data.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)
    
    print("✓ PyPTO ADD test passed")
```

### 2.4.3 编译和运行

**模拟器模式（无需 NPU）：**

```bash
python -c "
from tadd_pypto import test_add
test_add(run_mode='sim')
"
```

**NPU 模式：**

```bash
# 设置环境变量
export TILE_FWK_DEVICE_ID=0

# 运行测试
python -c "
import torch_npu
from tadd_pypto import test_add
test_add(device_id=0, run_mode='npu')
"
```

---

## 2.5 PyTorch 框架对接

参考 `demos/baseline/add` 和 `demos/torch_jit/add` 中的实际代码。

### 2.5.1 PyTorch 环境搭建

**安装依赖：**

```bash
# 安装 PyTorch 和 torch_npu
pip install torch
pip install torch_npu

# 或使用 demos/baseline/add 的 requirements.txt
cd pto-isa/demos/baseline/add
pip install -r requirements.txt
```

**配置 CANN 环境：**

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2.5.2 PyTorch 对接代码详解

#### 方式一：使用 KERNEL_LAUNCH（demos/baseline/add）

**1. 定义算子 Schema：**

```cpp
// csrc/host/add_custom_impl.cpp
#include <torch/torch.h>

// 注册算子 schema
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("my_add(Tensor x, Tensor y) -> Tensor");
}
```

**2. 实现算子：**

```cpp
#include "utils.h"
#include "aclrtlaunch_add_custom.h"

at::Tensor run_add_custom(const at::Tensor &x, const at::Tensor &y)
{
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 20;
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }
    EXEC_KERNEL_CMD(add_custom, blockDim, x, y, z, totalLength);
    return z;
}

// 注册实现
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("my_add", TORCH_FN(run_add_custom));
}
```

**3. Python 调用：**

```python
import torch
import torch_npu

# 调用自定义算子
z = torch.ops.npu.my_add(x, y)
```

#### 方式二：使用 JIT 编译（demos/torch_jit/add）

```python
# add_compile_and_run.py
import torch
import torch_npu
from jit_util_add import jit_compile

def test_add():
    device = "npu"
    dtype = torch.float16
    shape = [20, 2048]
    
    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    # JIT 编译内核
    add_func = jit_compile("add_custom.cpp")
    add_func(x, y, z)
    torch.npu.synchronize()

    # 验证结果
    z_ref = x + y
    torch.testing.assert_close(z, z_ref)
    print("ADD test pass!")

if __name__ == "__main__":
    test_add()
```

### 2.5.3 编译和运行

**使用 demos/baseline/add：**

```bash
cd pto-isa/demos/baseline/add

# 设置 PTO_LIB_PATH
export PTO_LIB_PATH=$(pwd)/../../..

# 编辑 CMakeLists.txt 设置 SOC_VERSION
# 例如 A2/A3 使用 "Ascend910B1"

# 构建 wheel
python3 setup.py bdist_wheel

# 安装
cd dist
pip install *.whl

# 运行测试
cd ../test
python3 test.py
```

**使用 demos/torch_jit/add：**

```bash
cd pto-isa/demos/torch_jit/add
python3 add_compile_and_run.py
```

---

## 2.6 本章小结

本章介绍了：

1. **环境搭建**：CPU 模拟器和 NPU 环境配置
2. **TADD Kernel 代码详解**：基于 `demos/baseline/add` 的实际代码
   - 乒乓调优（Ping-Pong Buffering）
   - 多级 Tiling 策略
   - 流水线同步机制
3. **TileLang 框架对接**：环境搭建、代码实现、编译运行
4. **PyPTO 框架对接**：基于官方样例的实现
5. **PyTorch 框架对接**：KERNEL_LAUNCH 和 JIT 两种方式

### 推荐学习路径

1. 基于 Auto Mode 开发算子，验证功能正确性
2. 在 CPU 模拟器中调试
3. 移植到 NPU 硬件，采集性能数据
4. 定位瓶颈（CUBE Bound / MTE Bound / Vector Bound），进行优化

详见：[性能优化指南](https://gitcode.com/cann/pto-isa/blob/master/docs/coding/opt_zh.md)

---

> 📖 更多资源：
> - [PTO 指令列表](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/README_zh.md)
> - [Tile 编程模型](https://gitcode.com/cann/pto-isa/blob/master/docs/coding/Tile_zh.md)
> - [事件与同步](https://gitcode.com/cann/pto-isa/blob/master/docs/coding/Event_zh.md)
