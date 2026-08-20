# PTO MegaMoE Dispatch + Combine 融合算子示例

## 概览

本示例演示如何使用 PTO Manual kernel 实现 MegaMoE 中 `dispatch -> GMM1 -> SwiGLU -> GMM2 -> combine -> unpermute` 的端到端融合流程。算子把传统 MoE 中多次重排、AlltoAllV 通信和 grouped FFN 计算合并到一个大 kernel 内，以 local expert 为流水粒度做 AIC/AIV 重叠。

## 支持的 AI 处理器

- Ascend950PR（A5，arch35 / DAV_3510 系列）
- 工具链兼容别名 `Ascend910_9599`

本项目固定使用 `dav-c310` 和 `PTO_NPU_ARCH_A5` 构建 mixed-core kernel。`Ascend910B` 属于 A3，不能作为本目录的 `--soc` 参数。

## 目录结构

```text
kernels/manual/a5/dispatch_mega_combine/
├── CMakeLists.txt                  # 构建配置，生成 host 可执行文件和 device kernel so
├── run.sh                          # 数据生成、构建、mpirun 执行的一键脚本
├── main.cpp                        # Host 入口：加载 case、初始化 ACL/HCCL/MPI、启动 kernel 和结果校验
├── kernel_launch.cpp               # Device kernel launch 包装
├── runtime_context.*               # 单 rank runtime、HCCL window、device/context 管理
├── tiling_builder.*                # Host 侧 tiling 构造和 workspace 规划
├── data_utils.*                    # case 数据文件读写和校验辅助
├── comm_mpi.h                      # MPI 动态加载包装
├── scripts/
│   └── gen_data.py                 # synthetic 输入、权重、golden 生成
├── op_kernel/
│   ├── dispatch_mega_combine.h     # MegaMoe device 主流程入口
│   ├── front_reorder.h             # token-order 量化、route mask 和 partial count 发布
│   ├── dispatch.h                  # mask 解码、远端 token 拉取和 GMM1 输入构造
│   ├── gmm_common.h                # GMM1/GMM2 共享 tile 调度和 helper
│   ├── gmm1.h                      # 第一层 grouped matmul
│   ├── swiglu.h                    # SwiGLU + dynamic quant
│   ├── gmm2.h                      # 第二层 grouped matmul
│   ├── combine.h                   # route metadata 预取和 route-slot 远端写回
│   ├── unpermute.h                 # route-slot topK reduce 和原 token 顺序还原
│   └── utils/                      # PTO vector、sync、HCCL window、GMM pipeline helper
```

## 算子说明

### 计算功能

本示例实现多 rank MoE FFN 主体流程：

```text
x[rank, M, K] + expertId[rank, M, topK] + probs[rank, M, topK]
  -> route mask 拉取为目标侧 expert-major token rows
  -> grouped GMM1
  -> SwiGLU activation + dynamic quant
  -> grouped GMM2
  -> combine 回源 rank
  -> topK weighted reduce
  -> out[rank, M, K]
```

逻辑公式可以理解为：

```text
for each rank, token:
  out[token] = sum_{topK route} probs[token, route] * FFN_expert(x[token])
```

其中 `FFN_expert` 由 MXFP8 GMM1、SwiGLU、MXFP8 GMM2 组成；前后通信通过 HCCL RDMA window 和 PTO 通信/同步 helper 完成。

### 规格

| 项目 | 值 |
| --- | --- |
| OpType | `MegaMoE Dispatch + FFN + Combine` |
| 输入 | `x`: `[M, K]`, `bfloat16`; `expertId`: `[M, topK]`, `int32`; `probs`: `[M, topK]`; `weight1/weight2`: 每个本地 expert 的 E4M3 数据；`scale1/scale2`: E8M0 scale |
| 输出 | `out`: `[M, K]`, `half/bfloat16` |
| Kernel 名称 | `dispatch_mega_combine_kernel` |
| Host 可执行文件 | `dispatch_mega_combine` |
| 默认脚本 case | `worldSize=2, M=2048, K=7168, N=4096, topK=8, expertPerRank=16, maxOutputSize=81940` |

## 优化说明

- **expert 级流水重叠**：AIC 侧 GMM1/GMM2 和 AIV 侧 dispatch/SwiGLU/combine 按 local expert group 轮转推进，通过 cache-line 隔离的 GM epoch 串接阶段边界。
- **Mask Pull Front**：源 token 只量化一次；PTO `TCMPS` 生成逐 expert route mask，Dispatch 只拉取命中记录。
- **PTO 双缓冲**：Front quant/mask、Dispatch 8-row gather、Combine 16-row metadata/scale 预取分别使用独立
  ping-pong UB 和流水事件。
- **GMM PTO tile 优化**：GMM1/GMM2 使用 PTO tile 编程，包含 output tile swizzle、L1 -> L0 多级复用、双缓冲和 fixpipe quant/cast。
- **SwiGLU expert 粒度 overlap**：GMM1 每完成一个 expert 就通知 SwiGLU；全部 AIV 写完该 expert 后，SwiGLU
  立即通知 GMM2。流程不再使用 segment metadata workspace 或 segment 级调度。
- **流式 Combine/Unpermute**：Combine 写回源 rank route slot 并发布 expert progress；Unpermute 不等最终
  全 rank barrier，直接消费已完成 route。

Host 通过 `aclrtGetDeviceInfo(..., ACL_DEV_ATTR_AICORE_CORE_NUM, ...)` 查询 AICore 数并选择已验证的
mixed-core 分配。每个 physical block 包含一个 AIC 和两个 AIV subblock。默认分配如下：

| AIC | AIV | 默认 GMM1 / Dispatch / SwiGLU block | 默认 GMM2 / Combine block |
| ---: | ---: | --- | --- |
| 28 | 56 | `0..19`（20） | `20..27`（8） |
| 32 | 64 | `0..20`（21） | `21..31`（11） |
| 36 | 72 | `0..23`（24） | `24..35`（12） |

GMM1 第一个 wave 使用全部可用 AIC，之后采用默认稳态分组；GMM1 结束后，其 AIC 可加入 GMM2
尾部。RankStreaming 先放行与 GMM1 组配对的两组 AIV；active Combine lane 完成且本地元数据清理后，
与 GMM2 组配对的 AIV 再作为 helper 加入。

`run.sh --aicore-num 0|28|32|36` 选择有效 launch 核数（`0` 表示采用 runtime 查询值）。请求值不能超过
实际物理核数。

## Tiling 参数

| 参数 | 默认值 / 说明 |
| --- | --- |
| `M` | 由 `run.sh --m` 或 case.json 指定 |
| `K` | GMM 输入 hidden size；要求满足 packed row 和 GMM tile 对齐 |
| `N` | FFN 中间维度；GMM1 输出，SwiGLU 后进入 `N/2` |
| `topK` | 每 token 路由专家数 |
| `expertPerRank` | 每 rank 本地 expert 数 |
| `worldSize` | MPI/HCCL rank 数 |
| `maxOutputSize` | 每 rank routed row workspace 上限 |
| `aicNum` | 有效 AICore launch 核数；已验证 28、32 和 36 |
| `aivNum` | 按 `2 * aicNum` 推导：56、64 或 72 |
| `GMM baseM/baseN` | 主要 tile 口径为 `128 x 256` output tile |
| `Front Mask Pull` | 唯一 Front 实现；host 拒绝 clipping、非法 expert、inactive token 和接收容量不足 |
| `固定角色 AIV UB` | Dispatch、SwiGLU、Combine、Unpermute 使用 216 KiB 主区；尾部 40 KiB 保留给同步快照 |
| `Dispatch / Unpermute tile` | Dispatch 每个 UB batch 搬 8 个 packed row；Unpermute 在 216 KiB 主区内动态选择最多 8192 列 |
| `Combine metadata batch` | 每批预取 16 个连续 GMM row 的 metadata/scale，使用两组 UB |

## 支持 Case

代表性验证 case 固定除 `M` 外的其它主参数：

```text
worldSize=8
K=7168
N=4096
topK=8
expertPerRank=16
aicNum=runtime（28、32 或 36），或通过 `run.sh --aicore-num 28|32|36` 指定
aivNum=2 * aicNum
```

典型 case 列表：

| M | maxOutputSize | 命令 |
| --- | --- | --- |
| 16 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 16 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 32 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 32 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 64 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 64 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 128 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 128 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 512 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 1024 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 1024 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 2048 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 2048 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |

## 整体架构

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Front Mask Pull (AIV)                                                       │
│   token-order quant + per-expert route mask/partial count                   │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ 独立 front-ready epoch
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Expert-level overlapped pipeline                                             │
│                                                                              │
│ AIV: Dispatch(group i) -> SwiGLU(group i) -> Combine(group i)                 │
│ AIC:                    GMM1(group i)      -> GMM2(group i)                  │
│                                                                              │
│ Stages communicate with per-expert GM arrival/ready epochs                    │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ final boundary
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Unpermute (AIV)                                                              │
│   route-slot output + probs -> topK weighted reduce -> out[M, K]              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Front Mask Pull 阶段

Front 在源 rank 保留每个 token 的一份量化记录，并向目标 rank 发布逐 global expert 的 route mask：

```text
x[M, K] + expertId[M, topK]
  -> sourceTokenRecords[M, K + 32]
  -> routeMaskSlots[localExpert, srcRank, mask + laneCapacity * 32B partial counts]
  -> cumsumMM[srcRank, localExpert] / expertTokenNums[localExpert]
```

所有 AIV 按 token 维均分量化任务，不切 K。同一个 expert 可由多个 AIV lane 并行生成 mask；lane 只写
完整的 32B mask block，每个 block 覆盖 256 个 route slot，并写独立的 32B partial-count record，因此
不存在跨 lane cache-line 写冲突。M=16、topK=8 时 mask 只有一个 block，每 expert 只能启用一个 lane。

quant 和 mask 都使用两组 PTO UB。所有 source rank 发布独立 front-ready epoch 后，coordinator 用 PTO
`TLOAD` 把 active lane count 搬入 UB，先做 lane 归约，再构造 source-major `cumsumMM` 和
`expertTokenNums`，最后释放 Dispatch。

## Dispatch 阶段

Dispatch 在目标 rank 运行。每个 source rank 分到 `dispatchGroupSize / worldSize` 个 AIV0 lane；所有 lane
扫描同一 mask，但只消费各自的命中序号区间：

```text
srcRank.sourceTokenRecords[routeSlot / topK]
  -> workspace.gmA[dstRowBase : dstRowBase + rows, 0:K]
  -> workspace.perTokenScale1[dstRowBase : dstRowBase + rows]
  -> workspace.routeMeta[dstRow] = {srcRank, routeSlot, 0...}
```

命中 route 按 8 row 聚合。两组 mask UB 重叠下一批 mask MTE2 和当前 scalar bit scan；两组 packed/meta UB
重叠远端 MTE2 与连续 MTE3 写。所有 active lane 完成一个 local expert 后，Dispatch 沿用现有 expert 级
epoch 放行 GMM1。

## GMM1 / SwiGLU / GMM2 阶段

### GMM1

GMM1 在 AIC 上执行第一层 grouped matmul：

```text
gmA[int8] x weight1[int8]
  -> int32 accumulator
  -> fixpipe scale1
  -> gmC[half]
```

每个 local expert 的输出 tile 网格按 `128 x 256` output tile 切分。线性 tile id 会通过 swizzle 映射到 `(blockM, blockN)`，让相邻 tile 更容易复用 L1 中的 B 侧权重。

### SwiGLU

SwiGLU 在 AIV 上消费 GMM1 的 `gmC` 和 dispatch 生成的 `perTokenScale1`：

```text
gmC * perTokenScale1
  -> silu(up) * gate
  -> dynamic quant
  -> gmPermutedToken[int8] + perTokenScale2[float]
```

SwiGLU 按 expert 逐个推进。它先等待 GMM1 ready epoch，再由 active SwiGLU AIV 分担当前 expert 的
row；activation 和 scale 全部写回后，协调核汇总 producer arrival，并按 GMM2 AIC 数发布 ready slot。

### GMM2

GMM2 在 AIC 上执行第二层 grouped matmul：

```text
gmPermutedToken[int8] x weight2[int8]
  -> int32 accumulator
  -> fixpipe scale2
  -> gmm2Output[half]
```

GMM2 完成每个 local expert group 后，由 GMM2 组或全部可用 AIC 发布 arrival；一个 AIV 协调核汇总后，
通过与 GMM2 AIC 数相同的共享 ready slot 放行 active Combine 组。

## Combine / Unpermute 阶段

Combine 在 AIV 上把 GMM2 输出乘 `perTokenScale2` 后直接写回源 rank 的 route slot：

```text
gmm2Output[srcRow, 0:K] half
  -> fp32
  -> * perTokenScale2[srcRow]
  -> OutputElement
  -> srcRank.combineOutputByRouteSlot[routeMeta[srcRow].routeSlot, 0:K]
```

每个 source-rank lane 每次预取 16 个连续 `routeMeta` 和 scale 到两组 UB；row 数据再用两组
`C/FP32/D` UB 做 `TLOAD -> TCVT/scale/TCVT -> TSTORE` 跨行重叠。该实现去掉逐 row 的 `DCCI` 和
scalar GM scale load，同时保留完整 K 行处理和 expert-progress 发布。

Unpermute 是最后的源 rank 还原阶段：

```text
combineOutputByRouteSlot[token * topK + topk] + probs
  -> 按原 token/topK 加权累加
  -> out[M, K]
```

## 内存布局与 HCCL 窗口

HCCL remote window 主要承载跨 rank 可见的数据：

| Buffer | 位置 | 用途 |
| --- | --- | --- |
| `sourceTokenRecords` | HCCL window | 每个源 token 一份 packed int8 记录；Dispatch 按 `routeSlot / topK` 拉取 |
| `routeMaskSlots` | HCCL window | 每个 local expert/source rank 的 mask 和对齐 lane partial count |
| `combineOutputByRouteSlot` | HCCL window | Combine 写源 route slot，Unpermute 直接消费 |
| `gmA` | workspace GM | Dispatch 生成的 GMM1 输入 |
| `gmC` | workspace GM | GMM1 输出，SwiGLU 输入 |
| `gmPermutedToken` | workspace GM | SwiGLU dynamic quant 后的 GMM2 输入 |
| `gmm2Output` | workspace GM | GMM2 输出，Combine 输入 |
| `routeMeta` | workspace GM | GMM row 到 `{srcRank, routeSlot}` 的 32B 映射 |
| `cumsumMM` | workspace GM | Dispatch 目标行地址使用的 source-major 累计 row |

`run.sh` 根据 `M`、`topK`、`K`、expert 拓扑和 mask lane 容量估算 HCCL window，并在需要时自动提高
`HCCL_BUFFSIZE`。

## 构建与运行

配置 Ascend CANN 环境：

```bash
source ~/zy/set_evn.sh
```

只编译 A5 mixed-core kernel 和 host：

```bash
cd ${git_clone_path}/kernels/manual/a5/dispatch_mega_combine
bash run.sh --build-only
```

请使用连续且可用的 A5 设备运行。环境需要提供 MPICH，MPI 兼容包装不支持 OpenMPI。

```bash
bash run.sh --soc Ascend910_9599 --world-size 2 --first-device 2 --m 2048 --k 7168 --n 4096 \
  --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

该命令将 rank 0、1 映射到物理卡 2、3。上面的 `Ascend910_9599` 是 A5 工具链兼容别名，不是 A3 的
`Ascend910B`。

切换其它 `M` 档位时可保持两卡首验配置，例如：

```bash
bash run.sh --soc Ascend910_9599 --world-size 2 --first-device 2 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

### 环境变量说明

| 环境变量 | 用途 | 默认行为 |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | CANN 安装目录 | 必须提前设置 |
| `CMAKE_COMPILER` | CMake 使用的编译器 | `bisheng` |
| `FIRST_DEVICE` | 连续 rank 映射中的第一张物理卡 | 默认 `0`，可由 `--first-device` 覆盖 |
| `MPI_LIB_PATH` | 可选的 MPICH `libmpi.so` 绝对路径 | 默认从 `LD_LIBRARY_PATH` 解析 |
| `MPI_RUNNER` | MPICH 启动命令 | 使用已 source 环境中的 `mpirun` |
| `HCCL_BUFFSIZE` | HCCL RDMA window 大小 | `run.sh` 按 case 自动抬高到安全值 |
| `DISPATCH_MEGA_COMBINE_AICORE_NUM` | 有效 AIC 数 | 默认 `0`，使用 runtime 查询值 |
| `DISPATCH_MEGA_COMBINE_WARMUP_ITERS` | 计时前的 warmup 启动次数 | `3` |
| `DISPATCH_MEGA_COMBINE_MEASURE_ITERS` | 用于整体 kernel 统计的计时启动次数 | `5` |

### 整体 Kernel 性能统计

计时启动完成后，rank 0 输出一份 `[KERNEL_PERF]` 整体 kernel 统计。每个 AIC/AIV 只记录整体起止 syscnt，统计以
每轮各 rank 最大耗时为口径，输出平均值、最小值、最大值、标准差、token 吞吐、等效计算 TFLOPS 和等效通信带宽。

## 修改 Case 参数

修改 `M` 时，其它主参数需要符合支持的拓扑，并按需调整接收容量。

```bash
bash run.sh --world-size 8 --first-device 0 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

常用约束：

- `K` 需要满足 packed row、GMM1/GMM2 tile 和量化路径的对齐要求。
- `N` 是 GMM1 输出维度，SwiGLU 后进入 GMM2 的维度为 `N / 2`。
- `expertPerRank` 必须是 `4`、`8`、`16` 或 `32`，与已编译的 kernel 特化一致；底层 C2V/V2C flag 布局的上限
  是 45 个 expert。
- `maxOutputSize` 必须覆盖单 rank 接收的 routed rows 上限。
- synthetic `expert_idx` 默认使用 global token round-robin，使小 M case 也能覆盖全局 expert。

## 常见问题

| 问题 | 原因与解决 |
| --- | --- |
| `ASCEND_HOME_PATH must be set` | 运行 `run.sh` 前需要 source CANN 环境并导出 `ASCEND_HOME_PATH` |
| HCCL window too small | 手动设置的 `HCCL_BUFFSIZE` 低于 case 需求；取消覆盖或调大该变量 |
| MPI 启动失败 | source 项目环境，并确认 `mpirun --version` 显示 MPICH/HYDRA；不支持 OpenMPI |
| golden 生成很慢 | 首次生成后使用 `--reuse-data` 复用文件；chunk 大小由程序内部固定 |
| 结果 diff 异常 | 先检查 data cache 是否复用旧分布；改变 expert 分布或 case 关键参数后不要使用旧 `out/` |

## 构建系统

- **编译器**：`bisheng`
- **Device kernel flags**：`-xcce --cce-aicore-arch=${CCE_AICORE_ARCH}`
- **Host executable**：`-xc++ -std=c++17`
- **输出 target**：`dispatch_mega_combine_kernel`、`dispatch_mega_combine`
- **链接库**：`stdc++`、`ascendcl`、`hcomm`、`runtime`、`tiling_api`、`platform`、`nnopbase`、`pthread` 等
- **PTO include**：仓库根目录 `include/` 会被放入 include path，用于 PTO tile/comm helper

## 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-06-26 | 新增 `dispatch_mega_combine` README，整理 MegaMoE 算子说明、阶段流程、构建运行和 FAQ |
| 2026-07-27 | 将 A3 优化调度迁移到 A5 后端，并完成仅编译验证 |
| 2026-08-14 | 删除开发期观测代码，完成 A5 生产化整理 |
