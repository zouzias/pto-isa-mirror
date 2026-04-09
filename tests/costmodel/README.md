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
| `PTO_HOST_RUNTIME` | （自动定义） | host 运行时模式 |

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
