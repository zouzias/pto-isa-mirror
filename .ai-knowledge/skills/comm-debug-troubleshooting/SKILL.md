---
name: comm-debug-troubleshooting
description: 通信算子调试与排障指南。覆盖通信算子常见错误模式（死锁、内存越界、数据不一致、rank 映射错误）、mssanitizer 内存检测、MC2 异常 dump 分析、HCCL 日志定位、SHMEM Troubleshooting FAQ、通信精度验证。触发：通信算子报错、卡死、超时、数据不一致、HCCL 错误码、MC2 异常、集合通信失败时。
---

# 通信算子调试与排障

## 定位

本 Skill 是**诊断型 Skill**，帮助排查通信算子开发中的各类问题。与 `ascendc-precision-debug` 和 `ascendc-runtime-debug` 互补，专注于通信特有的故障模式。

---

## 快速诊断表

| 症状 | 可能原因 | 排查方向 |
|------|---------|---------|
| **程序卡死/超时** | 死锁、同步不匹配、rank 未参与 | [死锁诊断](references/deadlock-diagnosis.md) |
| **数据全零/随机值** | 对称内存未正确分配、FFTS 未配置 | [常见错误模式](references/common-error-patterns.md) |
| **数据不一致** | 通信参数不匹配、引擎选择错误 | [常见错误模式](references/common-error-patterns.md) |
| **内存越界** | Buffer 过小、偏移计算错误 | [mssanitizer](references/mssanitizer-usage.md) |
| **HCCL 错误码** | API 参数错误、版本不匹配 | [HCCL 日志](#hccl-日志定位) |
| **MC2 异常 dump** | MoE 状态异常、Expert 不匹配 | [MC2 dump 分析](references/mc2-dump-analysis.md) |
| **精度偏差** | 归约顺序、量化损失 | [精度验证](references/precision-verification.md) |
| **aclnn 错误 161001/161002** | 参数为空/参数非法 | 检查输入 tensor 和参数 |
| **BUS ERROR** | 驱动/CANN 版本不匹配 | 确认 CANN 8.0.RC2+ |

---

## 调试工具

### SHMEM Debug 构建

```bash
bash scripts/build.sh -examples -debug
```

开启后可检测：
- 对称内存分配大小不对称
- 内存越界的更多提示

### mssanitizer 内存检测

```bash
# 编译时启用
bash scripts/build.sh -examples -mssanitizer

# 运行时使用
mssanitizer -- ./my_program arg1 arg2
```

**详细指南**：[mssanitizer 使用指南](references/mssanitizer-usage.md)

### SHMEM Profiling

```bash
# 设置 profiling PE
export SHMEM_CYCLE_PROF_PE=0

# 运行
bash run.sh -ranks 2
```

Kernel 内使用宏：
```cpp
#include "utils/prof/shmemi_prof.h"
SHMEMI_PROF_START(frame_id);
// ... 被测代码 ...
SHMEMI_PROF_END(frame_id);
```

Host 侧输出：
```cpp
aclrtSynchronizeStream(stream);
aclshmemx_show_prof(nullptr, true);
```

### HCCL 日志

HCCL 内置三级日志：
- `HCCL_DEBUG`：详细调试信息
- `HCCL_INFO`：关键流程信息
- `HCCL_ERROR`：错误信息

配合 plog 日志系统使用。

---

## HCCL 日志定位

### 开启详细日志

通过环境变量控制 HCCL 日志级别，配合 plog 查看详细通信过程。

### 常见 HCCL 错误排查

| 错误类型 | 典型原因 | 排查步骤 |
|---------|---------|---------|
| HCCL 初始化失败 | 网络配置错误、端口被占用 | 检查 IP、端口、网卡配置 |
| 通信超时 | rank 间网络不通 | `netstat -tuln` 检查端口 |
| 数据类型错误 | rank 间 dtype 不一致 | 确认所有 rank 参数一致 |
| OOM | 通信 Buffer 分配失败 | 减小 batch size 或 Buffer |

---

## SHMEM 常见问题 FAQ

### Q：对称内存分配后数据不正确

**原因**：各 rank `aclshmem_malloc` 大小不一致。
**排查**：使用 debug 构建 (`-debug` 编译选项)。
**解决**：确保所有 rank 传入完全相同的 size。

### Q：barrier 调用后程序卡死

**原因 1**：未配置 FFTS → `util_set_ffts_config(fftsAddr)`
**原因 2**：不在 Mix Kernel 中使用 barrier
**原因 3**：部分 rank 未到达 barrier（缺少参与者）

### Q：RDMA/SDMA 操作失败

**排查**：
1. 是否调用了 `aclshmemx_rdma_config` / `aclshmemx_sdma_config`
2. UB 预留是否足够（SDMA >= 64B）
3. 是否超过 `SDMA_SEND_MAX_SIZE`

### Q：MTE 跨组通信失败（910B 16P）

**原因**：910B 16 卡为两个 8P FullMesh 组，组间通过 PCIe-SW 连接，MTE 不支持跨组。
**症状**：stream sync failure 等。
**解决**：跨组使用 SDMA 或 RDMA。

### Q：网络连接失败

**排查**：
1. 检查 `SHMEM_UID_SESSION_ID` 或 `SHMEM_UID_SOCK_IFNAME` 设置
2. 确认 IP 可达、端口未被占用：`netstat -tuln | grep <port>`
3. 注意：同时设置时仅 `SHMEM_UID_SESSION_ID` 生效

### Q：-O0 -g 编译后运行时崩溃

**原因**：debug 构建可能导致最小栈大小超过默认 32768。
**解决**：在 aclInit 的 JSON 配置中增大栈大小。

---

## 排障流程

```
问题出现
│
├── 程序卡死？
│   └── 检查同步点 → 死锁诊断 → 检查所有 rank 是否参与
│
├── 数据错误？
│   ├── 全零 → 检查 FFTS 配置、对称内存分配
│   ├── 随机值 → 检查同步（通信完成前就读了数据）
│   └── 部分错 → 检查边界/padding 处理
│
├── 内存错误？
│   └── 使用 mssanitizer 定位越界
│
├── 性能差？
│   └── 使用 SHMEM profiling + msprof 分析
│
└── MC2 算子异常？
    └── 使用 dump_analysis 工具分析
```

---

## 相关 Skills

| Skill | 用途 |
|-------|------|
| `ascendc-precision-debug` | 计算精度调试（互补） |
| `ascendc-runtime-debug` | 运行时错误调试（互补） |
| `comm-api-reference` | API 约束查阅 |
| `comm-operator-develop` | 正确性检查清单 |
| `comm-performance-optimization` | 性能问题诊断 |
