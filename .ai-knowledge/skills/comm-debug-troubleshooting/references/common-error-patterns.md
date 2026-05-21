# 通信算子常见错误模式

## 错误 1：对称内存大小不对称

### 症状
- 数据不一致（部分正确、部分错误）
- 无明显错误日志

### 原因
不同 rank 的 `aclshmem_malloc` 传入了不同的 size。

### 排查
```bash
# 使用 debug 构建
bash scripts/build.sh -examples -debug
# debug 模式会检测并报告分配不对称
```

### 修复
确保所有 rank 使用完全相同的 size 参数。

---

## 错误 2：FFTS 未配置

### 症状
- `aclshmem_barrier_all()` 静默失败
- 集合通信操作无效果
- 数据全零或未更新

### 原因
在 Mix Kernel 中使用 barrier/集合通信前未调用 FFTS 配置。

### 修复
```cpp
__global__ void my_kernel(__gm__ uint8_t *fftsAddr, ...) {
    // 必须在任何集合通信前调用
    util_set_ffts_config(fftsAddr);
    // 然后才能使用
    aclshmem_barrier_all();
}
```

---

## 错误 3：高层 RMA 并发调用

### 症状
- 数据被覆盖
- 间歇性数据错误

### 原因
`aclshmem_put/get` 使用内部默认 buffer，不支持并发。

### 修复
- 方案 A：串行化 RMA 调用
- 方案 B：使用引擎直驱接口（`aclshmem_mte_put_nbi` 等）

---

## 错误 4：MTE 跨组使用

### 症状
- Stream sync failure
- 通信操作异常

### 原因
910B 16P 配置下，MTE 不支持跨 8P 组通信。

### 排查
检查通信的源 PE 和目标 PE 是否在同一 8P 组内：
- 组 A：rank 0-7
- 组 B：rank 8-15

### 修复
跨组通信使用 SDMA 或 RDMA 引擎。

---

## 错误 5：通信参数不一致

### 症状
- 死锁（部分 rank 等待，其他 rank 已完成）
- 数据错误
- HCCL 错误返回

### 原因
不同 rank 使用了不同的 count / dataType / reduceOp。

### 排查
在每个 rank 上打印实际传入的参数值，确认一致。

---

## 错误 6：引擎未配置

### 症状
- RDMA/SDMA 操作失败或使用默认配置
- UB 空间不足

### 原因
使用 RDMA/SDMA 引擎前未调用配置接口。

### 默认值（可能不适用）
| 引擎 | 默认 UB | 默认 sync_id |
|------|--------|-------------|
| RDMA | ~190 KB | EVENT_ID0 |
| SDMA | ~191 KB | EVENT_ID0 |

### 修复
显式调用配置：
```cpp
aclshmemx_rdma_config(ub_buffer, ub_size, sync_id);
aclshmemx_sdma_config(ub_buffer, ub_size, sync_id);
```

---

## 错误 7：MC2 group 参数错误

### 症状
- MC2 aclnn 调用返回 ACLNN_ERR_PARAM_INVALID (161002)
- 通信域查找失败

### 原因
group 字符串不是通过 `HcclGetCommName` 获取的。

### 修复
```cpp
char commName[128];
HcclGetCommName(hcclComm, commName);
// 使用 commName 作为 group 参数
```

---

## 错误 8：版本不匹配

### 症状
- BUS ERROR
- 不可解释的段错误

### 原因
驱动/固件/CANN 版本不配套。

### 修复
确认使用 CANN 8.0.RC2 及以上，且驱动固件版本配套。

---

## 错误 9：local_mem_size 不一致

### 症状
- "local size diffs" 错误
- SHMEM 初始化失败

### 原因
`aclshmemx_set_attr_uniqueid_args` 的 `local_mem_size` 各 PE 不同。

### 注意
- `local_mem_size` 按 2MB 对齐
- 框架内部约使用 ~6MB
- 实际可用 < 标称 `local_mem_size`
