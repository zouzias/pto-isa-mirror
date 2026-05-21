# 死锁诊断指南

## 通信死锁常见原因

### 原因 1：Notify/Wait 不匹配

```
Rank 0: NotifyRecord(id=1) → Wait(id=2)
Rank 1: NotifyRecord(id=2) → Wait(id=3)  ← id=3 永远不会被 record
```

**诊断**：检查所有 Notify/Wait 的 ID 配对是否完整。

### 原因 2：循环等待

```
Rank 0: Wait(from Rank 1)
Rank 1: Wait(from Rank 0)
// 双方互等，永远无法推进
```

**诊断**：画出等待关系图，检查是否有环。

### 原因 3：部分 rank 未参与

```
Rank 0: barrier_all()
Rank 1: barrier_all()
Rank 2: (跳过了 barrier) ← 其他 rank 永远等待
```

**诊断**：确认所有 rank 的代码路径都经过了集合同步点。

### 原因 4：条件分支导致同步点不一致

```cpp
if (myRank == 0) {
    // 只有 rank 0 执行了额外的 notify
    HcommChannelNotifyRecordOnThread(thread, channel, extraNotify);
}
// 其他 rank 等待这个 extraNotify → 可能死锁也可能不死锁
```

**诊断**：确保所有条件分支中的同步操作在所有 rank 上一致。

## 诊断步骤

### Step 1：确认卡死位置

通过日志/profiling 确定每个 rank 最后执行到哪一步：
- 使用 HCCL_DEBUG 日志
- 使用 SHMEM profiling
- MC2：使用 dump_analysis 的 status 分析

### Step 2：绘制同步关系图

```
Rank 0: [Compute] → Notify(R1) → Wait(R1) → [Compute] → ...
Rank 1: [Compute] → Wait(R0) → [Process] → Notify(R0) → ...
```

确认：
- 每个 Wait 有对应的 Notify
- 没有循环依赖
- 所有 rank 执行相同次数的集合同步

### Step 3：检查边界条件

- rank 数变化时（1/2/4/8 卡）是否仍正确
- 数据量为 0 时是否跳过了必要的同步
- 最后一个 iteration 的同步是否完整

## 预防措施

1. **设计阶段**：画出完整的同步时序图
2. **实现阶段**：使用统一的 notifyId 编号方案
3. **测试阶段**：测试不同 rank 数（1/2/4/8 卡）
4. **代码审查**：使用 `comm-operator-develop` 的正确性检查清单
