# 基础 Topk 算子示例

## 概览

本示例展示如何使用 PTO 实现一个基础 Topk 算子，包含工程结构、构建与执行流程。

## 支持的 AI 处理器

- A2/A3

## 目录结构

```
kernels/topk/
├── scripts/
│   └── gen_data.py              # 生成输入与 golden 输出
├── CMakeLists.txt               # 构建配置
├── topk_kernel.cpp        # Kernel 实现
├── main.cpp                     # Host 侧入口
└── run.sh                       # 便捷脚本
```

## 算子说明

### 计算功能

本示例实现固定维度 `[rows, cols] = [48, 1024]` 的 Topk：

### 规格

| 项目        | 值 |
| ----------- | ----- |
| OpType      | `topk` |
| 输入        | `[rows, cols] = [48, 1024]` |
| 输出        | `data`, `index` |
| Kernel 名称 | `topk_kernel` |

### Tiling 参数

验证平台有 24 个核。

每核形状：

- `rows = 2`, `cols = 1024`

## 实现说明

### 类型定义

Topk 实现： 从GM上加载数据到两个Tile上， 使用TCI生成索引， 使用TSort对每32个数据进行排序，使用TMrgsort对每个Tile内部做归并排序. 对两个排好序的tile做归并排序，分别取出topk个数据和索引，存回GM.

### 流水线调度

本示例通过在 L1 与 L0 使用双缓冲来重叠数据搬运与计算，以提高利用率。同步点用于确保依赖关系正确，包括：

- 正向同步：`MTE2 -> Vec -> Vec`
- 反向同步：`Vec -> MTE2`

流水线概览：
流水线示意图（TODO）

## 构建与运行

1. 配置 Ascend CANN 环境（示例路径）：

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. 运行示例：

```bash
cd ${git_clone_path}/demos/baseline/topk
bash run.sh -r npu -v Ascend910B1
```

成功时输出：

```text
test success
```
