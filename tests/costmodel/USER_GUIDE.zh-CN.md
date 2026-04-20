# PTO Costmodel 使用指南

## 1. Costmodel 是什么

Costmodel 是 PTO-ISA 的主机侧性能评估工具。它可以在不依赖真实 NPU 设备的情况下，通过记录 PTO 指令的执行轨迹（Trace），估算出每条 PTO 指令在目标架构上的 cycle 开销。

**核心能力：**

- 在普通 x86 主机上编译和运行 PTO 代码
- 记录每条 PTO 指令发出的底层 CCE 调用序列
- 基于架构规则估算每条 PTO 指令的 cycle 开销
- 将 cycle 结果绑定到目标 Tile 上，支持逐指令精度查询

**Costmodel 不是数值模拟器。** 它不计算 Tile 中的数据，只估算执行代价。如需主机侧数值模拟，请使用 CPU 后端。

## 2. 适用场景

| 场景 | 说明 |
|------|------|
| 指令性能回归测试 | 在 CI 中检查 PTO 指令的 cycle 是否符合预期 |
| 新指令 costmodel 验证 | 为新添加的 PTO 指令编写 cycle 校准测试 |
| 算子性能预估 | 在设备不可用时，对算子级 PTO 代码进行初步性能评估 |
| CCE 调用路径审查 | 观察某条 PTO 指令实际 lowering 到了哪些 CCE 调用 |

## 3. 快速开始

### 3.1 前置条件

- C++23 编译器（clang++ >= 15 或 g++ >= 13）
- CMake >= 3.16
- Python 3（用于测试运行器）
- Google Test（由 CMake 自动下载）

### 3.2 运行全部测试

```bash
# 在仓库根目录下执行
./tests/run_costmodel_tests.sh
```

该脚本会自动发现 `tests/costmodel/st/testcase/a2a3/` 下的所有测试用例，依次构建并运行。

### 3.3 运行单个测试

```bash
# 通过 Python 运行器直接运行单个用例
python tests/run_costmodel.py --testcase tadd --clean --verbose

# 不重新构建，只运行已构建的二进制
python tests/run_costmodel.py --testcase tadd --no-build
```

### 3.4 打印日志与 Trace

`run_costmodel.py` 使用 `--log-level` 控制单次运行的 costmodel 输出，并默认把结果写到 `<build_dir>/result.out`。

```bash
# log_level=0: 不打印单个 case 的 cycle / trace
python tests/run_costmodel.py --testcase tadd --no-build --log-level 0

# log_level=1: 打印每个 case 的实际 cycle 和期望 cycle
python tests/run_costmodel.py --testcase tadd --clean --log-level 1

# log_level=2: 打印 cycle + 完整 PTO / CCE trace
python tests/run_costmodel.py --testcase tadd --no-build --log-level 2

# 指定输出目录，结果写到 <output_dir>/result.out
python tests/run_costmodel.py --testcase tadd --log-level 2 --output-dir /tmp/costmodel_logs
```

输出文件按一次 `run_costmodel.py` 调用聚合。一次运行多个 testcase 时，所有选中 testcase 的日志会顺序写入同一个 `result.out`。

示例：

```text
== tadd ==
[COSTMODEL] TAdd.float_64x64 actual=96 expected=0.114514 precision=0 accuracy=0
[TRACE] TAdd.float_64x64
  pto: TADD
  total_cycles: 96
  cce_calls: 1
    [0] name=vadd cycles=78 args=[0x8000, 0x..., 0x4000, ...]
```

### 3.5 通过脚本运行指定用例

修改 `tests/run_costmodel_tests.sh` 中的 `TESTCASES` 数组：

```bash
TESTCASES=(tadd tmul tsub)
```

### 3.6 使用 GTest 过滤器

```bash
python tests/run_costmodel.py --testcase tadd --gtest_filter "TAdd.float_64x64"
```

## 4. 编写 Costmodel 测试

### 4.1 目录结构

每个测试用例位于 `tests/costmodel/st/testcase/a2a3/<用例名>/` 下：

```text
tests/costmodel/st/testcase/a2a3/
├── CMakeLists.txt              # 注册所有用例
├── tadd/
│   └── tadd.cpp                # 自包含的测试源文件
├── tmul/
│   └── tmul.cpp
└── ...
```

### 4.2 测试代码模板

```cpp
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <gtest/gtest.h>
#include "cost_check.hpp"

using namespace pto;

namespace {

template <typename T, int rows, int cols, float profiling, float accuracy>
void runTAdd()
{
    // 1. 定义 Tile 类型
    using TileData = Tile<TileType::Vec, T, rows, cols>;

    // 2. 创建 Tile 并分配地址
    TileData src0Tile(rows, cols);
    TileData src1Tile(rows, cols);
    TileData dstTile(rows, cols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(dstTile, 0x8000);

    // 3. 执行 PTO 指令
    TADD(dstTile, src0Tile, src1Tile);

    // 4. 校验 cycle 结果
    EXPECT_CYCLE_NEAR(dstTile, profiling, accuracy);
}

} // namespace

TEST(TAdd, float_64x64)
{
    runTAdd<float, 64, 64, 132.0f, 0.8f>();
}
```

### 4.3 关键接口说明

#### `EXPECT_CYCLE_NEAR(tile, profiling, accuracy)`

校验宏，比较 Tile 中记录的实际 cycle 与期望的 profiling 值。

- `tile`：执行 PTO 指令时的目标 Tile
- `profiling`：通过真实设备 Profiling 获取的期望 cycle 值
- `accuracy`：精度要求，范围 [0, 1]，代表最小相对精度

精度计算公式：

```text
precision = 1 - |profiling - actual| / profiling
```

测试通过条件：`precision >= accuracy`

当 `accuracy = 1.0` 时，要求实际值与期望值完全相等。

#### `tile.GetCycle()`

每个 Tile 在 PTO 指令执行后，会自动记录该指令的 cycle 估算结果。调用 `GetCycle()` 可获取这个值。

```cpp
TADD(dstTile, src0Tile, src1Tile);
float cycles = dstTile.GetCycle();  // 返回 TADD 的 cycle 估算
```

该接口仅在 `__COSTMODEL` 构建模式下可用。

#### `TASSIGN(tile, addr)`

在 costmodel 模式下为 Tile 分配伪地址。这不会产生真实的内存操作，只是让 PTO 内部逻辑可以正常运行。

### 4.4 注册新测试用例

1. 在 `tests/costmodel/st/testcase/a2a3/` 下创建新目录，例如 `tmyop/`
2. 编写 `tmyop/tmyop.cpp`
3. 在 `tests/costmodel/st/testcase/a2a3/CMakeLists.txt` 的 `A2A3_TESTCASES` 列表中添加 `tmyop`

```cmake
set(A2A3_TESTCASES
    tabs
    tadd
    ...
    tmyop    # 新增
)
```

### 4.5 Profiling 值获取

`EXPECT_CYCLE_NEAR` 中的 `profiling` 参数应来自真实 NPU 设备的 Profiling 数据。流程如下：

1. 在真实设备上运行对应 PTO 指令
2. 通过 Profiling 工具获取该指令的 cycle 数
3. 将该值写入测试代码中的 `profiling` 参数

## 5. 支持的 PTO 指令

当前 A2/A3 架构已支持 costmodel 测试的指令：

| 类别 | 指令 |
|------|------|
| 算术运算 | TADD, TSUB, TMUL, TDIVS, TMULS, TADDS, TSUBS |
| 元素级运算 | TEXP, TSQRT, TABS, TNEG |
| 比较运算 | TMINS, TMAXS |
| 类型转换 | TCVT |
| 数据搬运 | TLOAD, TMOV, TEXTRACT, TSCATTER |
| 矩阵运算 | TMATMUL, TMRGSORT, TSORT32 |
| 归约运算 | TROWSUM, TCOLSUM, TROWMAX, TROWMIN, TCOLMAX, TROWEXPAND |
| 选择/转置 | TSEL, TTRANS |
| 载入转换 | TLOADCONV |

## 6. 架构选择

当前支持的目标架构：

| 架构 | `__NPU_ARCH__` | 说明 |
|------|----------------|------|
| A2/A3 | 2201 | 默认架构 |
| A5 | 3101 | 需单独支持 |

测试入口默认使用 A2/A3 架构。

## 7. 常见问题

### Q: 测试输出中 `precision` 是什么？

A: `precision = 1 - |expected - actual| / expected`，值越接近 1 表示估算越精确。当 `precision >= accuracy` 时测试通过。

### Q: cycle 估算结果和真实设备差异有多大？

A: 差异取决于 CCE stub 的覆盖程度和架构规则的精度。对于已校准的指令，精度通常可以达到 80% 以上。

### Q: 如何调试 cycle 估算不准的问题？

A: 使用 `--log-level 2` 运行测试，可以看到完整的 trace 输出，包括 PTO 指令和 CCE 调用序列。`--verbose` 只控制构建和测试运行器的原始输出。

```bash
python tests/run_costmodel.py --testcase tadd --clean --log-level 2
```

### Q: 能否在 CI 中使用？

A: 可以。`run_costmodel_tests.sh` 在失败时会返回非零退出码，适合集成到 CI 流水线中。
