---
name: comm-architecture-overview
description: 昇腾通信架构总览与编程模型选择指南。提供通信体系全景图（HCCL/HCOMM/SHMEM）、通信语义抽象定义、编程模型决策树、关键概念解释。触发：用户提及"通信算子"、"多卡通信"、"集合通信"、"AllReduce"、"AllGather"、"ReduceScatter"、"跨卡"、"分布式"等关键词时；需要选择通信编程模型时；需要了解昇腾通信体系架构时。
---

# 昇腾通信架构总览

## 定位

本 Skill 是通信相关任务的**入口级知识库**。Agent 遇到通信算子开发、通算融合、多卡并行等任务时，应首先参考本 Skill 建立整体认知，再根据决策树选择合适的编程模型和对应的专项 Skill。

---

## 通信体系全景

昇腾通信体系采用分层架构，从上到下：

| 层级 | 组件 | 职责 | 典型使用者 |
|------|------|------|-----------|
| **框架层** | PyTorch/MindSpore HCCL 后端 | 框架集成，对用户透明 | 框架开发者 |
| **融合算子层** | MC2 算子（ops-transformer） | 通算融合（MatMul+AllReduce 等） | 算子开发者 |
| **集合通信层** | HCCL | 标准集合通信 API（AllReduce 等） | 算子/框架开发者 |
| **通信原语层** | HCOMM | 控制面（拓扑/资源）+ 数据面（RMA/Notify） | 通信算子开发者 |
| **共享内存层** | SHMEM | 对称内存管理 + Device 侧 RMA/集合通信 | 融合算子/内核开发者 |
| **硬件传输层** | RDMA/SDMA/UDMA/MTE | 物理数据搬运引擎 | 底层通信开发者 |

### 架构关系

```
应用/框架
    │
    ├─ MC2 融合算子 ─── HCCL + Catlass/Catcoc + SHMEM
    │
    ├─ HCCL（集合通信）
    │      │
    │      └─ HCOMM（通信原语）
    │             │
    │             ├─ 控制面：拓扑查询、资源管理、通信域
    │             └─ 数据面：Write/Read/Reduce + Notify/Wait
    │
    └─ SHMEM（共享内存）
           │
           ├─ Host：初始化、对称堆、通信域管理
           └─ Device：RMA（put/get）、集合通信、P2P 同步
                  │
                  └─ 引擎层：MTE / RDMA / SDMA / UDMA
```

---

## 通信语义抽象（实现无关）

无论使用哪种编程模型，通信操作都可归结为以下语义原语：

### 点对点通信（P2P）

| 语义 | 说明 | HCOMM 对应 | SHMEM 对应 |
|------|------|-----------|-----------|
| **远端写（Put）** | 将本地数据写到远端节点内存 | `HcommWriteOnThread` | `aclshmem_put` / MTE/RDMA/SDMA 引擎 |
| **远端读（Get）** | 从远端节点内存读取数据到本地 | `HcommReadOnThread` | `aclshmem_get` |
| **远端写归约（WriteReduce）** | 写到远端并与远端数据做归约 | `HcommWriteReduceOnThread` | `aclshmem_put` + 原子操作 |

### 集合通信（Collective）

| 语义 | 说明 | HCCL API |
|------|------|---------|
| **AllReduce** | 所有 rank 的数据归约，结果广播到所有 rank | `HcclAllReduce` |
| **AllGather** | 每个 rank 贡献一份，所有 rank 获得全量 | `HcclAllGather` / `HcclAllGatherV` |
| **ReduceScatter** | 归约后将结果分片分发给各 rank | `HcclReduceScatter` / `HcclReduceScatterV` |
| **Broadcast** | 从 root rank 广播到所有 rank | `HcclBroadcast` |
| **AllToAll** | 每个 rank 向每个 rank 发送不同数据 | `HcclAlltoAll` / `HcclAlltoAllV` / `HcclAlltoAllVC` |
| **Reduce** | 所有 rank 归约到 root rank | `HcclReduce` |
| **Scatter** | root rank 将数据分片分发给各 rank | `HcclScatter` |
| **Send/Recv** | 点对点发送/接收 | `HcclSend` / `HcclRecv` |

### 同步原语

| 语义 | 说明 |
|------|------|
| **Notify（信号发送）** | 向远端/本地发送同步信号 |
| **Wait（信号等待）** | 阻塞等待信号条件满足 |
| **Barrier（栅栏同步）** | 所有参与者都到达后才继续 |
| **Test（非阻塞检测）** | 非阻塞检查信号状态 |

---

## 关键概念

### 通信域（Communicator / Team）

- **HCCL Communicator**（`HcclComm`）：HCCL 层的通信域，包含参与集合通信的所有 rank
- **SHMEM Team**：SHMEM 层的通信域，管理 PE（Processing Element）集合
- **重要约束**：一个模型中所有 MC2 融合算子必须使用**同一通信域**（A2 平台）

### 对称内存（Symmetric Heap）

- SHMEM 提供的跨 rank 共享内存机制
- 通过 `aclshmem_malloc` 分配，**所有 rank 分配大小必须一致**（不一致会导致静默数据错误）
- 支持 Host 侧管理 + Device 侧直接 RMA 访问

### PE / Rank 映射

- **PE**（Processing Element）：SHMEM 中的处理单元标识（`aclshmem_my_pe()` / `aclshmem_n_pes()`）
- **Rank**：HCCL 中的进程/设备标识（`HcclGetRankId`）
- 通常一个 NPU 设备对应一个 PE/Rank

### 链路类型

| 链路 | 场景 | 特征 |
|------|------|------|
| **HCCS** | 节点内（同板 NPU 间） | 高带宽、低延迟 |
| **PCIe** | 节点内（跨板/跨 CPU） | 中等带宽 |
| **RoCE** | 跨节点 | 标准 RDMA over Ethernet |

### 硬件传输引擎

| 引擎 | 特征 | 适用场景 |
|------|------|---------|
| **MTE** | AICore 直驱，支持 UB2GM/GM2GM | 计算核内直接搬运 |
| **SDMA** | 专用 DMA 引擎，节点内优先 | 大块连续数据搬运 |
| **RDMA** | RoCE/ibverbs，跨节点 | 跨机通信 |
| **UDMA/UB** | 统一总线协议 | 特定硬件路径 |

---

## 编程模型选择决策树

根据任务类型选择合适的编程层级：

```
需要通信功能？
├── 仅需标准集合通信（AllReduce 等）
│   └── 使用 HCCL 标准 API → 参考 comm-api-reference
│
├── 需要通算融合（MatMul+通信等）
│   ├── 已有 MC2 算子覆盖场景？
│   │   ├── 是 → 使用 MC2 aclnn API → 参考 comm-compute-fusion-develop
│   │   └── 否 → 基于 SHMEM+Catcoc 开发新融合算子 → 参考 comm-compute-fusion-develop
│   │
│   └── 需要自定义融合策略？
│       └── 使用 SHMEM Device API + Catlass → 参考 comm-compute-fusion-develop
│
├── 需要开发新的通信算子（新算法/拓扑）
│   └── 使用 HCOMM 原语 + SHMEM → 参考 comm-operator-develop
│
└── 需要底层传输优化
    └── 选择引擎（RDMA/SDMA/MTE） → 参考 comm-hw-transport
```

---

## 相关 Skills

| Skill | 用途 |
|-------|------|
| `comm-compute-fusion-develop` | 通算融合算子开发 |
| `comm-operator-develop` | 通信算子开发 |
| `comm-api-reference` | 通信 API 速查 |
| `comm-hw-transport` | 硬件通信通路详解 |
| `comm-tiling-design` | 通信/融合算子 Tiling 设计 |
| `comm-performance-optimization` | 通信性能优化 |
| `comm-debug-troubleshooting` | 通信调试排障 |
| `ascendc-npu-arch` | NPU 架构信息 |
