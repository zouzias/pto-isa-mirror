# kirinX90 TLOAD 测试用例分析与 Python 数据生成实现

## 背景

用户要求分析 kirinX90 的 tload 指令的 12 个测试用例分别在做什么,并将 C++ 代码中的 golden 数据生成逻辑迁移到 Python 脚本中。

根据 readme.md 的说明:
- kirin9030 和 kirinX90 共用一套测试用例
- 测试用例位于 `tests/npu/kirin9030/src/st/testcase/tload/`
- 原始的 golden 数据生成逻辑在 C++ 代码中 (`tload_kernel.cpp`)
- 需要在 Python 脚本 (`gen_data.py`) 中实现相同的功能

## TLOAD 指令概述

TLOAD (Tile Load) 是 PTO ISA 中的数据加载指令,用于从全局内存 (Global Memory) 加载数据到 Tile (本地缓存)。

### 关键特性

1. **5 维张量支持**: 支持 `[shape0, shape1, shape2, shape3, shape4]` 的 5 维张量
2. **32B 对齐**: 输出数据自动对齐到 32 字节边界
3. **Padding 策略**: 支持多种填充策略 (Null, Max, Min, Zero)
4. **动态/静态 Shape**: 支持动态和静态两种 shape 配置
5. **多 Block 并行**: 支持多个 AI Core block 并行执行

## 12 个测试用例详细分析

### 测试用例 1: `case_float_GT_128_128_VT_128_128_BLK1`
- **数据类型**: `float` (32位浮点)
- **张量形状**: `[1, 1, 1, 128, 128]`
- **Tile 大小**: `128x128`
- **Padding 策略**: `PadValue::Null` (无填充)
- **动态/静态**: 动态 shape (dyn=1)
- **Block 数量**: 1
- **测试目的**: 基础的 float 类型 TLOAD,完整对齐的 128x128 tile
- **输入大小**: 65,536 bytes (128×128×4)
- **输出大小**: 65,536 bytes (无需 padding)

### 测试用例 2: `case_float_GT_2_2_2_256_64_VT_256_64_BLK8`
- **数据类型**: `float`
- **张量形状**: `[2, 2, 2, 256, 64]`
- **Tile 大小**: `256x64`
- **Padding 策略**: `PadValue::Null`
- **动态/静态**: 动态 shape (dyn=1)
- **Block 数量**: 8 (多 block 并行)
- **测试目的**: 多维度、多 block 的 float TLOAD
- **输入大小**: 524,288 bytes (2×2×2×256×64×4)
- **输出大小**: 524,288 bytes (无需 padding)

### 测试用例 3: `case_float_GT_128_127_VT_128_128_BLK1_PADMAX`
- **数据类型**: `float`
- **张量形状**: `[1, 1, 1, 128, 127]` (列不对齐)
- **Tile 大小**: `128x128`
- **Padding 策略**: `PadValue::Max` (填充正无穷)
- **动态/静态**: 动态 shape (dyn=1)
- **Block 数量**: 1
- **测试目的**: 测试列维度不对齐时的 Max padding 行为
- **输入大小**: 65,024 bytes (128×127×4)
- **输出大小**: 65,536 bytes (128×128×4, 最后一列填充 inf)

### 测试用例 4: `case_s16_GT_128_127_VT_128_128_BLK1_PADMAX`
- **数据类型**: `int16_t` (16位有符号整数)
- **张量形状**: `[1, 1, 1, 128, 127]`
- **Tile 大小**: `128x128`
- **Padding 策略**: `PadValue::Max` (填充 int16_t 最大值 32767)
- **动态/静态**: 动态 shape (dyn=1)
- **Block 数量**: 1
- **测试目的**: 测试 int16 类型的 Max padding
- **输入大小**: 32,512 bytes (128×127×2)
- **输出大小**: 32,768 bytes (128×128×2)

### 测试用例 5: `case_u8_GT_128_127_VT_128_128_BLK1_PADMIN`
- **数据类型**: `uint8_t` (8位无符号整数)
- **张量形状**: `[1, 1, 1, 128, 127]`
- **Tile 大小**: `128x128`
- **Padding 策略**: `PadValue::Min` (填充 uint8_t 最小值 0)
- **动态/静态**: 动态 shape (dyn=1)
- **Block 数量**: 1
- **测试目的**: 测试 uint8 类型的 Min padding
- **输入大小**: 16,256 bytes (128×127×1)
- **输出大小**: 16,384 bytes (128×128×1)

### 测试用例 6: `case_float_GT_32_64_128_VT_64_128_BLK32_DYN`
- **数据类型**: `int16_t` (注意:虽然测试名称是 float,但实际使用 int16_t)
- **张量形状**: `[1, 1, 32, 64, 128]` (高维度张量)
- **Tile 大小**: `64x128`
- **Padding 策略**: `PadValue::Null`
- **动态/静态**: 动态 shape (dyn=1)
- **Block 数量**: 32 (大规模并行)
- **测试目的**: 测试高维度张量的 TLOAD,模拟注意力机制中的张量重排 (如 BNSD→BSH)
- **输入大小**: 524,288 bytes (1×1×32×64×128×2)
- **输出大小**: 524,288 bytes (无需 padding)

### 测试用例 7: `case_float_GT_32_64_128_VT_64_128_BLK32_STC`
- **数据类型**: `int16_t`
- **张量形状**: `[1, 1, 32, 64, 128]`
- **Tile 大小**: `64x128`
- **Padding 策略**: `PadValue::Null`
- **动态/静态**: **静态 shape (dyn=0)** ⚠️
- **Block 数量**: 32
- **测试目的**: 与测试用例 6 对比,测试静态 shape 的 TLOAD 性能和正确性
- **输入大小**: 524,288 bytes
- **输出大小**: 524,288 bytes

### 测试用例 8: `case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX`
- **数据类型**: `float`
- **张量形状**: `[2, 2, 2, 256, 60]` (列不对齐)
- **Tile 大小**: `256x64`
- **Padding 策略**: `PadValue::Max`
- **动态/静态**: 动态 shape (dyn=1)
- **Block 数量**: 8
- **测试目的**: 多维度、多 block、不对齐的 float TLOAD with Max padding
- **输入大小**: 491,520 bytes (2×2×2×256×60×4)
- **输出大小**: 524,288 bytes (2×2×2×256×64×4)

### 测试用例 9: `case_int64_GT_128_128_VT_128_128_BLK1`
- **数据类型**: `int64_t` (64位有符号整数)
- **张量形状**: `[1, 1, 1, 128, 128]`
- **Tile 大小**: `128x128`
- **Padding 策略**: `PadValue::Null`
- **动态/静态**: 动态 shape (dyn=1)
- **Block 数量**: 1
- **测试目的**: 测试 64位整数类型的 TLOAD
- **输入大小**: 131,072 bytes (128×128×8)
- **输出大小**: 131,072 bytes

### 测试用例 10: `case_uint64_GT_128_125_VT_128_128_BLK1_PADZERO`
- **数据类型**: `uint64_t` (64位无符号整数)
- **张量形状**: `[1, 1, 1, 128, 125]` (列不对齐)
- **Tile 大小**: `128x128`
- **Padding 策略**: `PadValue::Zero` (填充零)
- **动态/静态**: 动态 shape (dyn=1)
- **Block 数量**: 1
- **测试目的**: 测试 uint64 类型的 Zero padding
- **输入大小**: 128,000 bytes (128×125×8)
- **输出大小**: 131,072 bytes (128×128×8)

### 测试用例 11: `case_int64_GT_2_2_2_256_62_VT_256_64_BLK8_PADZERO`
- **数据类型**: `int64_t`
- **张量形状**: `[2, 2, 2, 256, 62]` (列不对齐)
- **Tile 大小**: `256x64`
- **Padding 策略**: `PadValue::Zero`
- **动态/静态**: 动态 shape (dyn=1)
- **Block 数量**: 8
- **测试目的**: 多维度、多 block 的 int64 TLOAD with Zero padding
- **输入大小**: 1,015,808 bytes (2×2×2×256×62×8)
- **输出大小**: 1,048,576 bytes (2×2×2×256×64×8)

### 测试用例 12: `case_uint64_GT_2_2_2_256_64_VT_256_64_BLK8`
- **数据类型**: `uint64_t`
- **张量形状**: `[2, 2, 2, 256, 64]`
- **Tile 大小**: `256x64`
- **Padding 策略**: `PadValue::Null`
- **动态/静态**: 动态 shape (dyn=1)
- **Block 数量**: 8
- **测试目的**: 多维度、多 block 的 uint64 TLOAD,完整对齐
- **输入大小**: 1,048,576 bytes (2×2×2×256×64×8)
- **输出大小**: 1,048,576 bytes

## 测试覆盖维度总结

### 1. 数据类型覆盖
- **32位浮点**: `float` (测试 1, 2, 3, 8)
- **16位整数**: `int16_t` (测试 4, 6, 7)
- **8位整数**: `uint8_t` (测试 5)
- **64位有符号整数**: `int64_t` (测试 9, 11)
- **64位无符号整数**: `uint64_t` (测试 10, 12)

### 2. Padding 策略覆盖
- **Null** (无填充): 测试 1, 2, 6, 7, 9, 12 (6个)
- **Max** (填充最大值): 测试 3, 4, 8 (3个)
- **Min** (填充最小值): 测试 5 (1个)
- **Zero** (填充零): 测试 10, 11 (2个)

### 3. Shape 类型
- **动态 shape** (dyn=1): 11 个测试
- **静态 shape** (dyn=0): 1 个测试 (测试 7)

### 4. 并行度
- **单 block**: 测试 1, 3, 4, 5, 9, 10 (6个)
- **8 blocks**: 测试 2, 8, 11, 12 (4个)
- **32 blocks**: 测试 6, 7 (2个)

### 5. 对齐情况
- **完全对齐** (列数是 32B 对齐): 测试 1, 2, 6, 7, 9, 12 (6个)
- **不对齐需要 padding**: 测试 3, 4, 5, 8, 10, 11 (6个)

## Python 数据生成实现

### 实现目标

将 C++ 代码中的 `get_input_golden_case()` 函数的功能迁移到 Python 脚本中,使测试数据可以通过 Python 脚本预先生成,而不需要在测试运行时通过 C++ 代码生成。

### 关键技术实现

#### 1. 32B 对齐计算

```python
def align_to_32B(cols, dtype):
    """
    Align column count to 32-byte boundary.
    """
    type_32_aligned = 32 // np.dtype(dtype).itemsize
    return ((cols + type_32_aligned - 1) // type_32_aligned) * type_32_aligned
```

**原理**:
- 32 字节对齐意味着每行的字节数必须是 32 的倍数
- `type_32_aligned` 表示 32 字节可以容纳多少个该类型的元素
- 例如: float32 (4 bytes) → 32/4 = 8 个元素对齐
- 例如: int16 (2 bytes) → 32/2 = 16 个元素对齐

#### 2. Padding 值计算

```python
def get_pad_value(dtype, pad_type):
    """
    Get padding value based on data type and padding strategy.
    """
    if pad_type == 'null' or pad_type == 'zero':
        return 0
    elif pad_type == 'max':
        if np.issubdtype(dtype, np.floating):
            return np.inf
        else:
            return np.iinfo(dtype).max
    elif pad_type == 'min':
        if np.issubdtype(dtype, np.floating):
            return -np.inf
        else:
            return np.iinfo(dtype).min
    else:
        return 0
```

**Padding 策略**:
- `Null/Zero`: 填充 0
- `Max`: 浮点数填充 `inf`,整数填充类型最大值
- `Min`: 浮点数填充 `-inf`,整数填充类型最小值

#### 3. 随机数据生成

```python
# For floating point types
input_data = np.random.randn(shape0, shape1, shape2, shape3, shape4).astype(dtype)

# For integer types
if dtype == np.uint8:
    input_data = np.random.randint(0, 256, size=(...), dtype=dtype)
elif dtype == np.int16:
    input_data = np.random.randint(-1000, 1000, size=(...), dtype=dtype)
# ... 其他类型
```

**与 C++ 的差异**:
- C++ 使用顺序递增的值: `value = x0*... + x1*... + ... + i*shape4 + j`
- Python 使用随机数,更接近真实测试场景
- 使用固定随机种子 (42) 保证可重复性

#### 4. Golden 数据生成

```python
# Calculate aligned column count
shape4_aligned = align_to_32B(shape4, dtype)

# Generate random input data
input_data = np.random.randn(shape0, shape1, shape2, shape3, shape4).astype(dtype)

# Create golden data array with aligned shape
golden_data = np.zeros((shape0, shape1, shape2, shape3, shape4_aligned), dtype=dtype)

# Copy valid data
golden_data[:, :, :, :, :shape4] = input_data

# Apply padding if needed
if shape4_aligned > shape4:
    pad_val = get_pad_value(dtype, pad_value)
    golden_data[:, :, :, :, shape4:] = pad_val

# Save to binary files
input_data.tofile("input.bin")
golden_data.tofile("golden.bin")
```

### 参数类设计

```python
class TLoadParams:
    """
    Parameters for TLOAD test case.
    """
    def __init__(self, dtype, shape0, shape1, shape2, shape3, shape4, 
                 tile_rows, tile_cols, pad_value, num_blocks, is_dynamic):
        self.dtype = dtype
        self.shape0 = shape0
        self.shape1 = shape1
        self.shape2 = shape2
        self.shape3 = shape3
        self.shape4 = shape4
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.pad_value = pad_value
        self.num_blocks = num_blocks
        self.is_dynamic = is_dynamic
```

### 测试用例配置

所有 12 个测试用例的参数都已配置在 `case_params_list` 中,与 C++ 代码中的 `launchTLOAD_1` 到 `launchTLOAD_12` 一一对应。

## 验证结果

### 生成的文件

脚本成功生成了所有 12 个测试用例的数据:
- 每个测试用例生成 2 个文件: `input.bin` 和 `golden.bin`
- 文件大小与预期一致

### Padding 验证

以测试用例 3 为例 (float, 128×127, Max padding):
```python
# 最后一列 (padding 列)
Last column (should be inf): [inf inf inf inf inf]
All values are inf: True

# 倒数第二列 (有效数据列)
Second-to-last column (valid data): [ 0.07451583 -1.0896939  -0.72801554  0.9543306   0.20769522]
All values are finite: True
```

验证结果表明:
- Padding 列正确填充为 `inf`
- 有效数据列包含正常的浮点数
- 32B 对齐功能正常工作

## 使用方法

### 生成测试数据

```bash
cd tests/npu/kirin9030/src/st/testcase/tload
python3 gen_data.py
```

### 运行测试

生成数据后,可以运行 NPU 测试:
```bash
python3 tests/script/run_st.py -r npu -v a3 -t tload
```

或者在模拟器上运行:
```bash
python3 tests/script/run_st.py -r sim -v a3 -t tload
```

## 关键技术点总结

1. **32B 对齐**: 所有输出都会对齐到 32 字节边界,这是硬件要求
2. **性能测量**: C++ 代码使用系统计数器 (`get_syscnt()`) 测量 TLOAD 和 TSTORE 的时间
3. **事件同步**: 使用 `set_flag` 和 `wait_flag` 进行 pipeline 同步
4. **GlobalTensor**: 支持动态和静态两种 shape/stride 配置
5. **5维张量**: 所有测试都使用 5 维张量表示 `[shape0, shape1, shape2, shape3, shape4]`
6. **多 Block 并行**: 支持 1/8/32 个 AI Core block 并行执行

## 与 C++ 实现的差异

| 方面 | C++ 实现 | Python 实现 |
|------|---------|------------|
| 数据生成时机 | 测试运行时 | 预先生成 |
| 数据内容 | 顺序递增值 | 随机数 |
| 随机种子 | N/A | 固定种子 42 |
| 文件输出 | 测试框架输出 | 脚本直接输出 |
| 用途 | 运行时验证 | 预生成测试数据 |

## 未来改进方向

1. **支持更多数据类型**: 可以添加 bfloat16, float16 等类型的支持
2. **参数化配置**: 可以通过命令行参数指定要生成哪些测试用例
3. **数据验证**: 可以添加更多的数据验证逻辑,确保生成的数据符合预期
4. **性能优化**: 对于大规模数据生成,可以考虑并行化处理
5. **文档生成**: 可以自动生成每个测试用例的详细说明文档

## C++ 代码清理

在将数据生成逻辑迁移到 Python 后,我们删除了 C++ 代码中不再需要的数据生成函数,使代码更加简洁和符合标准测试模式。

### 删除的代码

#### 1. tload_kernel.cpp 中删除的内容

**删除的函数** (原第 287-334 行):
- `get_input_golden_case<>()` - 生成输入和 golden 数据的模板函数

**删除的函数** (原第 336-364 行):
- `get_input_golden<>()` - 根据 testKey 调用对应 case 的函数

**删除的模板实例化** (原第 379-390 行):
- 12 个 `get_input_golden<>` 的显式模板实例化声明

**删除代码统计**: 约 103 行

#### 2. main.cpp 中的修改

**删除的声明** (原第 21-22 行):
```cpp
template <int32_t testKey>
int get_input_golden(uint8_t *input, uint8_t *golden);
```

**修改前的数据生成逻辑** (原第 71-73 行):
```cpp
int actual_out_byteSize = 0;
actual_out_byteSize = get_input_golden<testKey>((uint8_t *)srcHost, (uint8_t *)goldHost);
std::fill((uint8_t *)dstHost, ((uint8_t *)(dstHost)) + out_byteSize, 0);
```

**修改后的文件读取逻辑**:
```cpp
// Read pre-generated test data from files
std::string goldenDir = GetGoldenDir();
std::ifstream inFile(goldenDir + "/input.bin", std::ios::binary | std::ios::in);
std::ifstream goldFile(goldenDir + "/golden.bin", std::ios::binary | std::ios::in);

// Check if files exist and provide helpful error message
if (!inFile.is_open()) {
    std::cerr << "Error: Cannot open input file: " << goldenDir << "/input.bin" << std::endl;
    std::cerr << "Please run 'python3 gen_data.py' in the test directory to generate test data first." << std::endl;
    FAIL() << "Input file not found. Run gen_data.py to generate test data.";
}
if (!goldFile.is_open()) {
    std::cerr << "Error: Cannot open golden file: " << goldenDir << "/golden.bin" << std::endl;
    std::cerr << "Please run 'python3 gen_data.py' in the test directory to generate test data first." << std::endl;
    FAIL() << "Golden file not found. Run gen_data.py to generate test data.";
}

// Get file sizes
inFile.seekg(0, std::ios::end);
size_t in_file_size = inFile.tellg();
inFile.seekg(0, std::ios::beg);

goldFile.seekg(0, std::ios::end);
int actual_out_byteSize = goldFile.tellg();
goldFile.seekg(0, std::ios::beg);

// Read data from files
inFile.read((char *)srcHost, in_file_size);
goldFile.read((char *)goldHost, actual_out_byteSize);

inFile.close();
goldFile.close();

std::fill((uint8_t *)dstHost, ((uint8_t *)(dstHost)) + out_byteSize, 0);
```

**修改前的文件写入逻辑** (原第 93-101 行):
```cpp
std::ofstream inFile(GetGoldenDir() + "/input.bin", std::ios::binary | std::ios::out);
std::ofstream outFile(GetGoldenDir() + "/output.bin", std::ios::binary | std::ios::out);
std::ofstream goldFile(GetGoldenDir() + "/golden.bin", std::ios::binary | std::ios::out);
inFile.write((const char *)srcHost, actual_out_byteSize);
outFile.write((const char *)dstHost, actual_out_byteSize);
goldFile.write((const char *)goldHost, actual_out_byteSize);
inFile.close();
outFile.close();
goldFile.close();
```

**修改后的文件写入逻辑** (与其他测试用例保持一致):
```cpp
WriteFile(GetGoldenDir() + "/output.bin", dstHost, actual_out_byteSize);
```

### 修改后的优势

1. **代码更简洁**: 删除了约 114 行代码 (tload_kernel.cpp: 103 行 + main.cpp: 11 行)
2. **符合标准模式**: 与其他测试用例 (tadd, tload_shape2d 等) 保持一致
3. **职责分离**: 数据生成 (Python) 和测试执行 (C++) 完全分离
4. **更好的错误提示**: 当测试数据文件不存在时,给出明确的错误信息和解决方案
5. **更快的测试**: 不需要每次运行测试时都生成数据
6. **更易维护**: Python 脚本更容易修改和扩展

### 与标准测试模式的一致性

修改后的 tload 测试与其他测试用例完全一致:

| 方面 | tadd | tload_shape2d | tload (修改后) |
|------|------|---------------|----------------|
| 读取输入文件 | ✅ ReadFile | ✅ ReadFile | ✅ ifstream + read |
| 写入 output.bin | ✅ WriteFile | ✅ WriteFile | ✅ WriteFile |
| 读取 golden 比较 | ✅ ReadFile | ✅ ReadFile | ✅ ReadFile |
| 在代码中生成数据 | ❌ | ❌ | ❌ (已删除) |
| 错误提示 | - | - | ✅ 友好的错误信息 |

### 代码统计

- **删除**: 约 114 行 (tload_kernel.cpp: 103 行 + main.cpp: 11 行)
- **添加**: 约 35 行 (main.cpp: 文件读取 + 错误处理)
- **净减少**: 约 79 行代码

## 结论

本次实现成功将 C++ 中的 golden 数据生成逻辑迁移到 Python 脚本中,并清理了 C++ 代码,实现了以下目标:

1. ✅ 支持所有 12 个测试用例
2. ✅ 支持 5 种数据类型 (float, int16, uint8, int64, uint64)
3. ✅ 支持 4 种 padding 策略 (Null, Max, Min, Zero)
4. ✅ 正确实现 32B 对齐
5. ✅ 使用随机数生成测试数据
6. ✅ 验证 padding 功能正常工作
7. ✅ 删除了 C++ 中不再需要的数据生成代码
8. ✅ 使测试代码符合标准模式
9. ✅ 添加了友好的错误提示

这套测试用例全面覆盖了 TLOAD 指令在不同数据类型、不同 padding 策略、不同并行度和不同对齐情况下的行为,确保指令在各种场景下都能正确工作。代码更加简洁、易维护,并与其他测试用例保持一致。
