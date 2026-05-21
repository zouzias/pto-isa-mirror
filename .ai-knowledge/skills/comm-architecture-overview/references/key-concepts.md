# 通信关键概念详解

## 通信域（Communicator / Team）

### HCCL Communicator

- 类型：`HcclComm`
- 初始化：`HcclCommInitRootInfo` 或 `HcclCommInitAll`
- 销毁：`HcclCommDestroy`
- 查询 rank：`HcclGetRankId(comm, &rankId)`
- 查询大小：`HcclGetRankSize(comm, &rankSize)`
- 获取通信域名称（MC2 使用）：`HcclGetCommName(comm, commName)`

**重要约束**：
- A2 平台上，同一模型内的所有 MC2 融合算子必须使用**同一个 Communicator**
- 所有参与同一集合操作的 rank 必须使用相同的参数（count、dataType、op）

### SHMEM Team

- PE 标识：`aclshmem_my_pe()` 获取当前 PE 编号
- PE 总数：`aclshmem_n_pes()` 获取参与 PE 总数
- Team 管理：`aclshmem_team_*` 系列接口

---

## 对称内存（Symmetric Heap）

对称内存是 SHMEM 编程模型的核心：所有 PE 在相同的虚拟地址偏移处分配相同大小的内存。

### 分配规则

```
PE 0: aclshmem_malloc(size)  →  addr_0
PE 1: aclshmem_malloc(size)  →  addr_1  (逻辑偏移与 PE 0 相同)
...
```

### 关键约束

| 约束 | 说明 | 违反后果 |
|------|------|---------|
| **大小对称** | 所有 rank 的 `aclshmem_malloc` 大小必须一致 | 静默数据错误（无报错） |
| **`local_mem_size` 一致** | `aclshmemx_set_attr_uniqueid_args` 的 `local_mem_size` 必须每个 PE 相同 | "local size diffs" 错误 |
| **2MB 对齐** | `local_mem_size` 按 2MB 对齐 | 奇数大小可能导致不同错误 |
| **内部预留** | 框架约预留 ~6MB 内部使用 | 实际可用 < 标称值 |

### 调试技巧

使用 debug 模式编译可以发现分配不对称问题：
```bash
bash scripts/build.sh -examples -debug
```

---

## 链路类型与拓扑

### 节点内拓扑

**910B 8P FullMesh**：8 个 NPU 通过 HCCS 全互连
- 任意两个 NPU 之间直接 HCCS 链路
- MTE/SDMA 均可用于数据搬运

**910B 16P（双 8P）**：
- 两组 8P FullMesh
- 组内：HCCS 全互连
- 组间：PCIe Switch 互连
- **MTE 不能跨组使用**（仅限同组 NPU 间）

### 拓扑层级查询

HCOMM 提供分层拓扑查询：

```
HcclRankGraphGetLayers(comm, ...)      → 获取拓扑层级列表
HcclRankGraphGetRanksByLayer(comm, layer, ...)  → 获取指定层级的 rank 列表
HcclRankGraphGetTopoTypeByLayer(...)   → 查询层级拓扑类型
HcclRankGraphGetLinks(...)             → 查询 rank 间链路信息
```

### 链路选择影响

| 链路类型 | 带宽特征 | 通信引擎选择 |
|---------|---------|------------|
| HCCS | 高带宽、低延迟 | SDMA / MTE / RDMA 均可 |
| PCIe | 中等带宽 | SDMA 优先 |
| RoCE | 取决于网络配置 | RDMA（ibverbs） |

---

## 归约操作（Reduce Operations）

### HCCL 支持的归约操作

| 操作 | HCCL 常量 | 说明 | 数据类型限制 |
|------|----------|------|------------|
| Sum | `HCCL_REDUCE_SUM` | 求和 | 全部 |
| Prod | `HCCL_REDUCE_PROD` | 乘积 | int16/bfp16 不支持（A3/A2） |
| Max | `HCCL_REDUCE_MAX` | 最大值 | 全部 |
| Min | `HCCL_REDUCE_MIN` | 最小值 | 全部 |

### 数据类型支持

HCCL 支持：int8/16/32/64, uint8/16/32/64, float16/32, bfloat16

**注意**：
- int64 在 A2 平台可能较慢
- 不同 API（AllReduce vs ReduceScatterV）支持的类型范围**可能不同**，需查阅具体 API 文档

---

## 执行模式

### Op-base 模式（单算子执行）

标准执行模式，每个集合操作独立下发到 stream：

```
HcclAllReduce(sendBuf, recvBuf, count, dtype, op, comm, stream);
aclrtSynchronizeStream(stream);
```

### Graph 模式

将多个通信操作编排为 Graph 后一次性提交：
- 减少 Host-Device 交互开销
- 支持更好的资源预分配
- 需要 `interface_graph_mode` 层的资源计算
