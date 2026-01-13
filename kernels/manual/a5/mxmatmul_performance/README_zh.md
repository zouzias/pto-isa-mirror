# 高性能 MxMatmul 算子示例(如何命名？)

## 概览

本样例演示如何使用 PTO 实现高性能的带有量化系数的矩阵乘法，即高性能MxMatmul计算

并覆盖常见优化手段（多核切分、base-block 选择、L1 缓存与双缓冲）。

## 支持的 AI 处理器

- A5

## 目录结构

```
kernels/manual/a5/mxmatmul_performance/
├── scripts/
│   └── gen_data.py                      # 生成输入与 golden 输出
├── CMakeLists.txt                       # 构建配置
├── mxmatmul_performance_kernel.cpp      # Kernel 实现
├── main.cpp                             # Host 侧入口
└── run.sh                               # 便捷脚本
```

## 算子说明

### 计算功能

本示例实现 MxMatmul, 先分别将左 / 右量化系数矩阵与左 / 右输入矩阵进行广播乘法，再对两组乘积结果执行矩阵乘法运算，计算表达式如下：

$$
C = (scaleA ⊗ A) * (scaleB ⊗ B)
$$

其中 “⊗” 表示广播乘法，“*” 表示矩阵乘法

- `A` 为 `m×k`
- `scaleA` 为 `m×scaleK`
- `B` 为 `k×n`
- `scaleB` 为 `scaleK×n`
- `C` 为 `m×n`

`main.cpp` 中默认的参考配置为 `m=k=n=6144`， `scaleK=k/32=192`。

### 规格

| 项目        | 值 |
| ----------- | ----- |
| OpType          | `MxMatmul` |
| data输入         | `a`: `m×k`, `float8_e5m2_t`, `ND`; `b`: `n×k`, `float8_e5m2_t`, `ND` |
| scale输入        | `scaleA`: `m×scaleK`, `float8_e8m0_t`, `ND`; `scaleB`: `n×scaleK`, `float8_e8m0_t`, `ND` |
| 输出             | `c`: `m×n`, `float`, `ND` |
| Kernel 名称      | `MxMatmulPerformance` |

## 优化说明

本示例以 32 核的 A5 平台作为性能验证平台。

- **多核切分（core partitioning）**：在 Cube 核之间切分工作量，尽量把并行度吃满。由于 `m`、`n`、`k` 相等，通常不建议在单核内再切 `k`，而是把 `m` 与 `n` 分摊到 32 核。本示例使用 `4 × 8` 分组，对应 `singleCoreM=1536`、`singleCoreK=6144`、`singleCoreN=768`。
- **Base block 选择（base block selection）**：选一个算力/访存比更高、且更贴合片上容量与对齐约束的 base block。对 FP6，常见选择 `[baseM, baseN, baseK] = [128, 256, 128]`，该基本块计算访存比最高，同时更容易保持 GM 写回的 512 字节对齐。
- **L1 缓存（L1 caching）**：一次从 GM 搬入多个 base block 到 L1，提高带宽利用率。本示例 `stepKa=stepKb=4`，每次缓存 4 个 `k` block。
- **L1 上scale和data独立缓存（L1 independent caching）**：引入mxScalePara参数，表示L1上scale与data缓存的比例关系。在复用已有tiling参数的基础上，保证scale读地址满足128B对齐，提高带宽利用率。
- **双缓冲（double buffering）**：在 L1/L0A/L0B/scaleA/scaleB 开启双缓冲，让 DMA 与计算尽可能重叠。

## Tiling 参数

| 参数          | 值     |
| ------------- | ----- |
| `m`           | 6144  |
| `k`           | 6144  |
| `n`           | 6144  |
| `singleCoreM` | 1536  |
| `singleCoreK` | 6144  |
| `singleCoreN` | 768   |
| `baseM`       | 128   |
| `baseK`       | 128   |
| `baseN`       | 256   |
| `stepM`       | 1     |
| `stepKa`      | 4     |
| `stepKb`      | 4     |
| `stepN`       | 1     |
| `mxScalePara` | 8     |

## 实测性能（参考）

以下数据在 Ascend A5（32 核）上测得，覆盖多个 `m=k=n` 尺寸（fp8 输入 → fp32 输出）。

| 参数 | TMATMUL（Cube）占比 | TEXTRACT 占比 | TLOAD 占比 | TSTORE 占比 | 执行时间（ms） |
| --- | --- | --- | --- | --- | --- |
| `m=6144` `k=6144` `n=6144` | 88.9% | 55.3% | 97.0% | 6.0% | 0.6225 |

## 构建与运行

1. 配置 Ascend CANN 环境：

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. 生成输入 + golden 输出：

```bash
cd ${git_clone_path}/kernels/manual/a5/mxmatmul_performance
python3 scripts/gen_data.py
```

3. 运行示例：

```bash
bash run.sh -r npu -v Ascend910_9599
```

成功时输出：

```text
test success
```
