# PTO Costmodel Tests

Costmodel 测试基于 Google Test 框架，通过 `EXPECT_CYCLE_NEAR` 宏校验 PTO 指令的 cycle 估算精度。

详细使用指南请参阅 [USER_GUIDE.zh-CN.md](USER_GUIDE.zh-CN.md)。

## 快速开始

```bash
# 运行全部测试
./tests/run_costmodel_tests.sh

# 运行单个用例
python tests/run_costmodel.py --testcase tadd --clean --verbose
```

## 测试目录结构

```text
tests/costmodel/st/
├── CMakeLists.txt               # 顶层 CMake，定义编译选项和架构宏
├── common/
│   └── cost_check.hpp           # EXPECT_CYCLE_NEAR 宏定义
└── testcase/
    └── a2a3/
        ├── CMakeLists.txt       # 注册所有 A2/A3 测试用例
        ├── tadd/tadd.cpp
        ├── tmul/tmul.cpp
        └── ...
```

## 构建说明

测试通过 `tests/run_costmodel.py` 自动构建，不需要手动运行 cmake。

关键编译宏：

| 宏 | 值 | 说明 |
|----|----|------|
| `__COSTMODEL` | （自动定义） | 启用 costmodel 后端 |
| `PTO_NPU_ARCH_A2A3` | （自动定义） | 选择 A2/A3 架构 |
| `__NPU_ARCH__` | 2201 | NPU 架构编号 |
| `__CPU_SIM` / `__COSTMODEL` | （按测试路径自动定义） | host 运行时条件 |

## 添加新测试用例

1. 在 `tests/costmodel/st/testcase/a2a3/` 下创建 `<用例名>/` 目录
2. 编写 `<用例名>/<用例名>.cpp`（自包含的 GTest 源文件）
3. 在 `tests/costmodel/st/testcase/a2a3/CMakeLists.txt` 的 `A2A3_TESTCASES` 列表中添加用例名

## 可选参数

```bash
# 使用 GTest 过滤器
python tests/run_costmodel.py --testcase tadd --gtest_filter "TAdd.float_64x64"

# 不重新构建
python tests/run_costmodel.py --testcase tadd --no-build

# 强制清理重建
python tests/run_costmodel.py --testcase tadd --clean
```

## 日志输出示例

`run_costmodel.py` 现在支持 `--log-level` 控制单次运行的 costmodel 输出，并默认把结果写到 `<build_dir>/result.out`。

```bash
# log_level=0: 不输出单个 case 的 cycle/trace，result.out 为空
python tests/run_costmodel.py --testcase tadd --no-build --log-level 0

# log_level=1: 输出每个 gtest case 的 actual / expected cycles
python tests/run_costmodel.py --testcase tadd --clean --log-level 1

# log_level=2: 输出 cycles + 完整 PTO/CCE trace
python tests/run_costmodel.py --testcase tadd --no-build --log-level 2

# 自定义输出目录，结果写到 <output_dir>/result.out
python tests/run_costmodel.py --testcase tadd --log-level 2 --output-dir /tmp/costmodel_logs
```

`result.out` 按一次 `run_costmodel.py` 调用聚合输出。单次运行多个 testcase 时，所有选中 testcase 的日志都会按执行顺序追加到同一个文件中。

示例输出：

```text
== tadd ==
[COSTMODEL] TAdd.float_64x64 actual=96 expected=0.114514 precision=0 accuracy=0
[TRACE] TAdd.float_64x64
  pto: TADD
  total_cycles: 96
  cce_calls: 1
    [0] name=vadd cycles=96 args=[0x8000, 0x..., 0x4000, ...]
```
