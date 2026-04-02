# AllGather + GEMM 通信计算融合（M 维切分 + 流式流水线）

基于昇腾 AI Core 的 AllGather 与 GEMM 融合实现：**M 维切分** + **Tile 流式流水线**，通过通信与计算并发及细粒度 tile 重叠隐藏通信延迟。

---

## 使用方式

### 1. 运行 AllGather-GEMM 示例程序

```bash
bash run.sh <device_list> [options]
```

**参数说明**：

- `device_list`：指定用于运行的设备（NPU）编号列表，以逗号分隔。
- `--perf`：启用性能测试模式（默认为功能验证模式）。
- `-M/-K/-N`：指定单个 shape 进行测试（不指定则使用 `scripts/test_shapes.csv` 中的所有 shape）。
- `-v/--soc-version`：SoC 版本（默认 `Ascend910B1`）。
- `-r/--run-mode`：运行模式 `npu`/`sim`（默认 `npu`）。

**示例**：

```bash
# 使用第 6 和第 7 个 NPU 设备运行 2 卡功能验证（CSV 中所有 shape）
bash run.sh 6,7

# 使用 4 卡进行性能测试（CSV 中所有 shape）
bash run.sh 0,1,2,3 --perf

# 指定单个 shape 进行性能测试
bash run.sh 6,7 --perf -M 131072 -K 4096 -N 4096

# 指定 SoC 版本
bash run.sh 6,7 -v Ascend910B3
```

### 2. 配置计算规模

矩阵形状参数（M、K、N）可在配置文件 `scripts/test_shapes.csv` 中进行设置。修改该文件以定义测试用例的输入维度。

### 3. 性能分析

```bash
bash run_with_msprof.sh -r npu -v Ascend910B1 -s large -n 2 -p application
```

---

## 项目概述

在多卡 LLM 推理等场景中，各 rank 持有 A 的 M 维切片，需先通过 AllGather 收齐完整 A，再与全局 B 做 GEMM 得到 C。本 demo 将 AllGather（通信）与 GEMM（计算）在同一设备上并发执行，并使用 **summary 单调计数器 + TWAIT** 实现 tile 级"来一块算一块"。

### 执行模式对比

```
串行执行（Sequential）:
  [ AllGather 全部完成 ] ──► [ GEMM 全部完成 ]

流式流水线（Streaming Pipelined）:
  Comm (AIV):  [tile0 传输][summary++] [tile1 传输][summary++] ...
  Compute(AIC): [本地直算]  [TWAIT tile0][算 tile0] [TWAIT tile1][算 tile1] ...
```

---

## 并发架构

### 双 Stream 并发

Comm 与 Compute 使用两个独立 stream：

- **Comm Kernel**（AIV）：按 tile_idx 升序传输，每完成一个 tile 对远程 summary 做 AtomicAdd +1。
- **Compute Kernel**（AIC）：本地 rank 数据直接计算（零等待），远程 rank 数据用 TWAIT 阻塞等待 summary 达到目标值。

### AI Core 资源

| 单元 | 用途 | 本 demo 角色 |
|------|------|--------------|
| **AIC (Cube)** | 矩阵乘加 | Compute Kernel：GEMM |
| **AIV (Vector)** | 向量/搬运 | Comm Kernel：TPUT + 信号 |

---

## 关键优化

- **Summary-only 单调计数器 + TWAIT**：零 polling 开销的硬件等待。
- **本地数据零等待优先计算**：Compute kernel 先算本地 rank 的 row-group。
- **通信发送顺序与计算消费顺序对齐**：减少等待时间。
- **连续 K 累加流水**：跨 K-block 持续 TMATMUL_ACC。
- **L1/L0 两级双缓冲**：形成连续流水。
- **并行 AIV 通信**：full-mesh 下每个 rank 直接 TPUT 到其余 rank。

---

## 文件结构

```
allgather_gemm_multi/
├── main.cpp                           # 入口 + 测试框架（warmup、验证、性能测量、统计打印）
├── allgather_gemm_comm_kernel.cpp     # 通信 Kernel (AIV) - AllGather TPUT
├── allgather_gemm_compute_kernel.cpp  # 计算 Kernel (AIC) - streaming GEMM
├── ready_queue.hpp                    # TileFlagMatrix / Summary 计数器元数据
├── run.sh                             # 构建运行（device_list + CSV 批量测试）
├── run_with_msprof.sh                 # msprof 分析
├── scripts/
│   ├── gen_data.py                    # 数据生成
│   ├── test_shapes.csv                # 测试 shape 配置
│   └── verify_result.py               # 结果验证
└── CMakeLists.txt
```
