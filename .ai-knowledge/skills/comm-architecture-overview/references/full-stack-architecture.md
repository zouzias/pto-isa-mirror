# 昇腾通信全栈架构详解

## HCCL 层

HCCL（Huawei Collective Communication Library）是昇腾平台的集合通信库，对标 NVIDIA NCCL。

### 算子架构（三段式）

HCCL 中每个集合通信算子采用统一的三层架构：

1. **Selector（算法选择器）**
   - 根据拓扑结构、数据量、硬件能力选择最优通信算法
   - 区分执行后端：CCU / AICPU / AIV
   - 示例：`AllReduceAutoSelector` 选择 Mesh vs NHR vs Ring

2. **Executor（资源编排器）**
   - 计算所需通信资源（Buffer、Notify、Stream）
   - 调用 `Orchestrate` 方法编排通信步骤
   - 支持 Op-base 和 Graph 两种执行模式

3. **Template（算法模板）**
   - 编码具体通信算法步骤
   - CCU kernel / AICPU 指令模板 / AIV kernel 三种实现形式
   - 示例：`ccu_temp_all_reduce_mesh_1D_mem2mem`

### 支持的集合通信

| 操作 | API | 变体 |
|------|-----|------|
| AllReduce | `HcclAllReduce` | 标准 |
| Broadcast | `HcclBroadcast` | 标准 |
| AllGather | `HcclAllGather` | + AllGatherV（异构 count） |
| ReduceScatter | `HcclReduceScatter` | + ReduceScatterV |
| AlltoAll | `HcclAlltoAll` | + AlltoAllV / AlltoAllVC |
| Reduce | `HcclReduce` | 到 root |
| Scatter | `HcclScatter` | 从 root |
| Send/Recv | `HcclSend`/`HcclRecv` | + BatchSendRecv |

### 通信算法

| 算法 | 适用场景 | 特征 |
|------|---------|------|
| **Ring** | 通用 | 带宽利用率高，延迟随 rank 数线性增长 |
| **Mesh** | 节点内全连接 | 低延迟，利用 HCCS 全 mesh 拓扑 |
| **NHR** | 大规模跨节点 | 分层策略，节点内 Mesh + 节点间 Ring/其他 |
| **分层** | 异构拓扑 | 节点内/节点间使用不同算法 |

---

## HCOMM 层

HCOMM 是通信原语层，为 HCCL 提供底层通信能力。

### 控制面

| 功能类别 | 关键接口 | 说明 |
|---------|---------|------|
| 拓扑查询 | `HcclGetRankId` / `HcclGetRankSize` | 查询 rank 信息 |
| 层级拓扑 | `HcclRankGraphGetLayers` / `GetRanksByLayer` | 查询拓扑层级结构 |
| 链路信息 | `HcclRankGraphGetLinks` | 查询 rank 间连接类型 |
| 资源管理 | `HcclGetHcclBuffer` / `HcclThreadAcquire` | 获取通信内存和线程 |
| 通道管理 | `HcclChannelAcquire` / `HcclChannelGetHcclBuffer` | 建立通信通道 |
| 引擎上下文 | `HcclEngineCtxCreate` / `Get` / `Copy` | 管理通信引擎上下文 |

### 数据面

| 功能类别 | 关键接口 | 说明 |
|---------|---------|------|
| 本地拷贝 | `HcommLocalCopyOnThread` | 设备内存拷贝 |
| 本地归约 | `HcommLocalReduceOnThread` | 本地元素级归约 |
| 远端写 | `HcommWriteOnThread` | 通过 Channel 向远端写数据 |
| 远端读 | `HcommReadOnThread` | 通过 Channel 从远端读数据 |
| 远端写归约 | `HcommWriteReduceOnThread` / `ReadReduceOnThread` | RMA + Reduce |
| 线程同步 | `HcommThreadNotifyRecord/Wait` | 线程间 Notify/Wait |
| 通道同步 | `HcommChannelNotifyRecord/Wait` | 跨 Channel 的 Notify/Wait |
| 批量模式 | `HcommBatchModeStart` / `End` | 缓存多个操作后统一下发 |

---

## SHMEM 层

SHMEM（Shared Memory）提供面向昇腾的 OpenSHMEM 风格共享内存编程接口。

### Host 侧

| 模块 | 功能 |
|------|------|
| **Init** | `aclshmem_init` / `aclshmem_finalize` 生命周期管理 |
| **Heap** | `aclshmem_malloc` / `aclshmem_free` 对称内存分配 |
| **Team** | `aclshmem_team_*` 通信域管理 |
| **Data Plane** | Host 驱动的 RMA / 信号 / 集合通信 |

### Device 侧

| 路径 | 接口类别 | 说明 |
|------|---------|------|
| **GM2GM** | RMA（`aclshmem_put/get`）| 高层远端内存访问 |
| **GM2GM** | AMO（原子操作） | 原子内存操作 |
| **GM2GM** | CC（集合通信） | `aclshmem_barrier_all` 等 |
| **GM2GM** | P2P-Sync | 点对点同步 |
| **GM2GM** | Signal（信号操作） | 信号量操作 |
| **GM2GM** | 引擎直驱 | MTE/RDMA/SDMA/UDMA 低阶接口 |
| **UB2GM** | RMA | AICore UB 到 GM 的远端访问 |
| **UB2GM** | MTE 引擎 | AICore 直驱 MTE |

### 关键约束

- `aclshmem_malloc` 所有 rank **分配大小必须一致**，否则静默数据错误
- `barrier` 当前**必须在 Mix Kernel 内使用**（MMAD + GM2UB/UB2GM）
- RDMA 高层 API 需先调用 `aclshmemx_rdma_config` 配置 UB buffer 和 sync_id
- SDMA 高层 API 需先调用 `aclshmemx_sdma_config`，预留 UB >= 64 字节
- 910B 16 卡拓扑为两个 8P FullMesh 组，MTE 不能跨组使用
