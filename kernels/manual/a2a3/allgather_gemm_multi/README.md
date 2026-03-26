# AllGather + GEMM 通信计算融合（M 维切分 + 流式流水线）

基于昇腾 AI Core 的 AllGather 与 GEMM 融合实现：**M 维切分** + **Tile 流式流水线**，通过通信与计算并发及细粒度 tile 重叠隐藏通信延迟。

---

## 1. 项目概述

在多卡 LLM 推理等场景中，各 rank 持有 A 的 M 维切片，需先通过 AllGather 收齐完整 A，再与全局 B 做 GEMM 得到 C。本 demo 将 AllGather（通信）与 GEMM（计算）在同一设备上并发执行，并使用 **TileFlagMatrix + summary** 实现 tile 级“来一块算一块”。

### 1.1 执行模式对比

```
串行执行（Sequential）:
  [ AllGather 全部完成 ] ──► [ GEMM 全部完成 ]

流式流水线（Streaming Pipelined）:
  Comm (AIV):  [tile0 传输][置 flag] [tile1 传输][置 flag] ...
  Compute(AIC): [等 tile0][算 tile0] [等 tile1][算 tile1] ...
```

---

## 2. 并发架构

### 2.1 双 Stream 并发

Comm 与 Compute 使用两个独立 stream：

- **Comm Kernel**（AIV）：按 tile 做传输通信，并写 tile-ready flag + summary。
- **Compute Kernel**（AIC）：先轮询 summary，再轮询 tile flag，tile 就绪即计算。

### 2.2 AI Core 资源

| 单元 | 用途 | 本 demo 角色 |
|------|------|--------------|
| **AIC (Cube)** | 矩阵乘加 | Compute Kernel：GEMM |
| **AIV (Vector)** | 向量/搬运 | Comm Kernel：TPUT + 信号 |

Ascend 910B 上 AIC 与 AIV 可同时调度，因此 Comm 与 Compute 可真实并发。

---

## 3. M 切分与 Tile 映射

- 每个 rank 持有本地 `A_local`：形状 `M_local x K`，`M_local = M / n_ranks`。
- 全局 row-group 数：`m_tiles = M / G_BASE_M`，每 rank `m_tiles_local = m_tiles / n_ranks`。
- 每个 row-group 需要 `k_chunks = K / G_BASE_N` 个通信块。
- 对单个源 rank，block 编码为：
  - `block_idx = mi_local * k_chunks + kb`
  - `mi_local in [0, m_tiles_local)`，`kb in [0, k_chunks)`
- tile 由 `TILE_SIZE` 个 block 组成，Comm 按 tile 通知，Compute 按 tile 消费。

---

## 4. 关键优化

### 4.1 TileFlagMatrix + summary 两级轮询

- 一级：`summary[src] >= 1` 时才进入该 src 的 tile 扫描。
- 二级：`IsTileReady(src, tile)` 判定具体 tile 是否可算。

### 4.2 连续 K 累加流水

- 使用 `ProcessKIterationContinuous`，跨 K-block 持续 TMATMUL_ACC。
- 每个 N-tile 只做一次 pipeline drain + 一次 TSTORE。

### 4.3 L1/L0 两级双缓冲

- `G_STEP_KA = G_STEP_KB = 4`，单次 TLOAD 缓存 4 个 K-slice。
- L1 与 L0 双缓冲配合 `wait_flag/set_flag` 形成连续流水。

### 4.4 并行 AIV 通信

- `launchRingCommStreaming` 按 `COMM_BLOCK_NUM` 并行调度 AIV block。
- 通信与计算在双 stream 下并发，按 tile 写 flag 实现细粒度重叠。

---

## 5. 运行方法

### 5.1 编译与运行

```bash
./run.sh -r npu -v Ascend910B1 -s large -n 2
```

参数：

- `-r`: 运行模式（`npu`/`sim`）
- `-v`: SoC 版本
- `-s`: 规模名
- `-n`: rank 数

### 5.2 性能分析

```bash
./run_with_msprof.sh -r npu -v Ascend910B1 -s large -n 2 -p application
```

## 6. 文件结构

```
allgather_gemm_multi/
├── allgather_gemm_comm_kernel.cpp     # 通信 Kernel (AIV) + Host 测速与打印
├── allgather_gemm_compute_kernel.cpp  # 计算 Kernel (AIC) - streaming GEMM
├── ready_queue.hpp                    # TileFlagMatrix + summary 元数据
├── main.cpp                           # 入口
├── run.sh                             # 构建运行
├── run_with_msprof.sh                 # msprof 分析
├── scripts/gen_data.py                # M 切分数据生成
└── CMakeLists.txt
```
