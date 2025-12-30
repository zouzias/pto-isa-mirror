# pto::comm 扩展

## 目标
为 Ascend 上的 tile-lib 增加Tile通信能力，方便未来pto-lib 支撑Triton-Distributed/IRIS 等第三方Tile编程库

pto::comm 支持如下特性：

* 支持TPut TGet TWait （目前实现）以及TAllReduce TBroadcast 等tile级通信指令
* 支持多后端实现，方便对接CANN-SHMEM等第三方通信库，从而低成本实现跨机、多种通信硬件的实现
* 支持统一的context管理接口，对称堆分配，rankID 维护等

当前穿刺版本后端基于 SHMEM，提供点到点 Put/Get，并计划扩展 Team（类似 OpenSHMEM team_split_strided）以及基于 tile 的广播 / All-Reduce 等 collective。

## 组件总览
- `include/pto/comm/comm_types.hpp`：后端枚举、Copy2DParams（行/列步长拷贝描述）。
- `include/pto/comm/context_manager.hpp`：统一的初始化/销毁、对称内存申请（SymmetricAlloc/Free），选择后端（现仅 Shmem）。
- `include/pto/comm/backend/shmem/*.hpp`：SHMEM 后端实现，包含 Copy2D 参数推导、类型特化的 put/get 调用。
- `include/pto/comm/pto_comm_inst.hpp`：设备侧 API：`TPUT/TGET/TWAIT`。

## SHMEM 依赖与配置
1) 依赖
   - 安装CANN-SHMEM 库： https://gitee.com/ascend/shmem
   - 需要可用的 `shmem_api.h` 以及对应静态/动态库（在 PATH/LD_LIBRARY_PATH 或编译链接路径中）。
2) 运行时配置
   - 对称堆大小：通过 `InitOptions::symmetricHeapBytes` 传入，后端调用 `shmem_set_attr(rank, nranks, heapBytes, ipPort, &attr)`。
   - Rank/Size：`InitOptions::rank`、`size` 对应 `shmem_set_attr` 的参数。
   - IP/Port：如后端需要 out-of-band 配置，可通过 `InitOptions::ipPort` 传递（可选）。

## Host 侧初始化流程
```cpp
#include "pto/comm/context_manager.hpp"

int world_rank = ...; // 由作业启动器提供
int world_size = ...;
pto::comm::InitOptions opts;
opts.backend = pto::comm::BackendKind::Shmem;
opts.rank = world_rank;
opts.size = world_size;
opts.symmetricHeapBytes = 256 * 1024 * 1024; // 示例：256MB 对称堆
// opts.ipPort = "10.0.0.1:23456"; // 如底层需要

int ret = pto::comm::ContextManager::Init(opts);
// 进程退出前
pto::comm::ContextManager::Finalize();
```

若需要对称内存：
```cpp
void* ptr = pto::comm::ContextManager::SymmetricAlloc(bytes);
// ...
pto::comm::ContextManager::SymmetricFree(ptr);
```

## 设备侧使用示例（Put/Get）
```cpp
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"

// 假设已创建 GlobalTensor ND/DN/NZ 之一，且 src/dst 形状、布局一致
pto::GlobalTensor<float, MyShape, MyStride> src(gmSrcPtr);
pto::GlobalTensor<float, MyShape, MyStride> dst(gmDstPtr);

int peer = (my_rank + 1) % world_size;
pto::comm::TPUT(dst, src, peer); // 非阻塞 put
pto::comm::TWAIT();              // 对应 shmem_quiet
```

## 测试
- 通信 ST 用例位于 `tests/comm/st/testcase/`（如 `tput/tget`）。需先准备 SHMEM 运行环境并为每个进程设置 rank/size。
- 参考主 README 的测试脚本：`python3 tests/script/run_st.py -r comm -v a3 -t tput`（示例，按实际 SoC/模式调整）。


## 计划中的 Team 与 Collective（设计稿）
> 说明：下述接口/语义为规划设计，需对应实现落地后方可使用。

### Team 概念
- World：`rank=[0, size)` 的默认 team。
- SplitStrided：按 `(parentStart, parentStride, teamSize)` 从 parent team 派生，语义对齐 OpenSHMEM `team_split_strided`。
- 能力：仅提供 rank 映射（team rank -> world rank），用于封装 collective 所需的拓扑。

### 预期接口（头文件规划）
- `team.hpp`：`Team`, `TeamWorld(parentRank, parentSize)`, `TeamSplitStrided(parent, start, stride, teamSize, myParentRank)`.
- `team_comm.hpp`：`TPUT_TEAM(dst, src, team, peerTeamRank)`, `TGET_TEAM(...)`.

### Tile Broadcast（规划）
- 接口：`TBroadcast(dstGlobal, srcGlobal, rootRank)`；Team 版：`TBroadcast(dstGlobal, srcGlobal, team, rootTeamRank)`.
- 约束：`GlobalTensor` 布局/元素类型一致；root 负责提供源数据。
- 后端：Shmem Backend，通过对称内存和分层/二叉树或扁平广播实现；优先实现扁平（root 逐个 TPUT + TWAIT）。

### Tile All-Reduce（规划）
- 接口：`TAllReduce(dstGlobal, srcGlobal, op, team)`；首版支持 `sum/max/min`。
- 约束：`dst/src` 同形同布局，元素类型受 ShmemOps 支持。
- 后端：Shmem Backend，优先实现 ring 算法（分段循环 TPUT/TGET + reduce，最后全量 broadcast），再视需要补 tree/recursive doubling。

### 线程/同步语义
- 非阻塞 Put/Get 后需调用 `TWAIT`（封装 `shmem_quiet`）。
- Collective 内部应保证所有参与 rank 在算法步骤上同步；首版可要求外部确保所有参与者按同序列调用。

### 待办
- 补充 `team.hpp/team_comm.hpp` 及 `TBroadCast.hpp/TAllReduce.hpp` 实现。
- 增加 ST 用例：broadcast/all-reduce（至少 ring 实现）。
- 在 `run_st.py/build_st.py` 中加入新的用例目标。
- 将本节更新为正式用法示例（包含代码）。

## 数据布局与限制
- 支持的 `GlobalTensor` 布局：`ND/DN/NZ`，要求 src/dst 布局一致。
- 元素类型需匹配（ShmemOps 针对基础类型和 half/bfloat16 提供特化）。
- Copy2DParams 自动从 `GlobalTensor` 形状/stride 推导；如需非 contiguous 2D 拷贝，可直接调用 `ShmemBackend::Put/Get` 并传入自定义 `Copy2DParams`。


## 常见问题
- 初始化失败：检查 `shmem_set_attr` 返回值；确认 rank/size、ipPort、对称堆大小与底层 SHMEM 实现匹配。
- 传输异常：确保 src/dst `GlobalTensor` 布局一致且元素类型匹配；NZ 布局需保证形状对齐 C0/fractal 要求。
- 内存不足：调大对称堆（`symmetricHeapBytes` 或底层环境变量）。 

