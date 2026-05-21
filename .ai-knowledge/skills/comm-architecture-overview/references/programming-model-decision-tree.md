# 编程模型选择详细指南

## 选择矩阵

| 场景 | 推荐模型 | 原因 | 对应 Skill |
|------|---------|------|-----------|
| 框架中调用标准集合通信 | HCCL API | 标准接口，最易用 | `comm-api-reference` |
| MatMul + AllReduce/ReduceScatter 等标准融合 | MC2 aclnn API | 已优化，直接使用 | `comm-compute-fusion-develop` |
| MoE dispatch/combine | MC2 aclnn API | 已有完整实现 | `comm-compute-fusion-develop` |
| 自定义通算融合算子 | SHMEM + Catlass/Catcoc | 灵活的融合编程框架 | `comm-compute-fusion-develop` |
| 开发新通信算法 | HCOMM 原语 | 直接控制通信步骤 | `comm-operator-develop` |
| 底层通信引擎优化 | SHMEM 引擎直驱 API | 控制具体 DMA 引擎 | `comm-hw-transport` |
| Kernel 内直接通信 | SHMEM Device API | AICore 内直接 RMA/同步 | `comm-operator-develop` |

## 各编程模型的特征对比

### HCCL 标准 API

```
优势：最简单、最稳定、框架原生支持
劣势：无法实现计算与通信的 kernel 级融合
典型代码：
    HcclAllReduce(send, recv, count, HCCL_DATA_FLOAT, HCCL_REDUCE_SUM, comm, stream);
```

### MC2 aclnn API

```
优势：预优化的通算融合，覆盖主流场景
劣势：场景固定，不支持自定义融合策略
典型代码：
    aclnnMatmulAllReduceGetWorkspaceSize(x1, x2, bias, group, reduceOp, ..., &wsSize, &executor);
    aclnnMatmulAllReduce(workspace, wsSize, executor, stream);
```

### SHMEM + Catcoc

```
优势：灵活的通算融合，AICore 内通信，tile 级重叠
劣势：编程复杂度高，需理解 SHMEM/Catcoc 编程模型
典型模式：
    Host: aclshmem_init → aclshmem_malloc → launch kernel → sync
    Device: BlockMmad (计算) + CommEpilogue (通信) + PE-aware 远程拷贝
```

### HCOMM 原语

```
优势：完全控制通信步骤和算法
劣势：需手动管理资源、拓扑、同步
典型模式：
    HcclThreadAcquire → HcclChannelAcquire → 
    HcommWriteOnThread → HcommChannelNotifyRecordOnThread →
    HcommChannelNotifyWaitOnThread → ...
```

## 场景决策流程

### 场景1：大模型 TP（张量并行）

TP 的典型通信模式是 AllReduce 或 ReduceScatter + AllGather：

1. 标准方案：用 HCCL `HcclAllReduce`
2. 性能优化方案：用 MC2 `aclnnMatmulAllReduce`（将 MatMul 和 AllReduce 融合）
3. 极致优化方案：用 SHMEM + Catcoc 自定义融合

### 场景2：大模型 MoE

MoE 的通信模式是 AllToAll（dispatch/combine）：

1. 标准方案：MC2 `moe_distribute_dispatch` / `moe_distribute_combine`
2. 需要量化通信时：MC2 quant 变体

### 场景3：自定义通信算法

当标准算法不适用（如特殊拓扑、非标通信模式）：

1. 使用 HCOMM 控制面查询拓扑
2. 使用 HCOMM 数据面原语编排通信步骤
3. 或使用 SHMEM Device API 在 kernel 内直接通信
