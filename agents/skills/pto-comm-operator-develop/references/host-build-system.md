# Host 侧与构建系统

## 标准初始化流程

下面流程适用于现有同步指令以及 SDMA/URMA 路径。异步 RDMA 还需要独立的
Host 控制面，见后文“RDMA/HNS1825 异步路径”。

```cpp
int main(int argc, char **argv) {
    // 1. MPI 初始化
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // 2. ACL 初始化
    aclInit(nullptr);
    aclrtSetDevice(rank % device_count);
    aclrtStream computeStream, commStream;
    aclrtCreateStream(&computeStream);
    aclrtCreateStream(&commStream);

    // 3. HCCL 通信域创建
    HcclRootInfo rootInfo;
    if (rank == 0) HcclGetRootInfo(&rootInfo);
    MPI_Bcast(&rootInfo, sizeof(rootInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
    HcclComm hcclComm;
    HcclCommInitRootInfo(nranks, &rootInfo, rank, &hcclComm);

    // 4. 获取通信上下文（远端地址）

    // 5. 内存分配
    uint8_t *buffer;
    aclrtMalloc((void**)&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST);

    // 6. 信号矩阵初始化（清零）
    aclrtMemset(signal_matrix, 0, signal_size);

    // 7. 启动 kernel
    launchCommKernel(buffer, ..., commStream);
    aclrtSynchronizeStream(commStream);

    // 8. 验证结果

    // 9. 清理
    HcclCommDestroy(hcclComm);
    aclrtDestroyStream(computeStream);
    aclrtDestroyStream(commStream);
    aclrtResetDevice(rank % device_count);
    aclFinalize();
    MPI_Finalize();
}
```

---

## RDMA/HNS1825 异步路径

`DmaEngine::RDMA` 与 SDMA/URMA 可以同时存在；它不是 HCCL 通信域的别名。
当前 PTO 只实现 HNS1825 后端，并且只支持 A5。

### 构建选择

当前 A5 通信 ST 在首次 CMake 配置时读取：

```bash
export PTO_RDMA_BACKEND=HNS_1825
```

CMake 必须把该选择同时转换为 Host 与 Device 的
`PTO_RDMA_SUPPORTED`、`PTO_RDMA_BACKEND_HNS_1825_SUPPORTED` 编译定义。
生成的二进制不再读取 `PTO_RDMA_BACKEND`。未设置、空值或不支持的值均表示
不编入 RDMA 后端；改变取值后必须重新配置构建目录。

集成自己的 CMake 工程时，还需要：

- Host 和 Device 都配置 PTO 的公开 `include` 根目录；使能 RDMA 时还要配置 PR 1452 引入的内部
  `pkg_inc` 根目录；
- Host 侧链接 `ascendcl`、`hcomm`、`dl`（并按工程现有方式链接 CANN Runtime）；Device 数据面不直接调用
  HCOMM API；
- 若 HNS1825 verbs provider 不在默认路径，通过 `IBV_EXTEND_DRIVERS` 指向
  `libhrn5-rdmav34.so`。

### Host 生命周期

应用负责在各 rank 间交换网卡与注册内存信息，然后由
`RdmaWorkspaceManager` 创建 HCOMM endpoint、注册本地通信缓冲区、建立
peer channel 并发布 Device workspace：

```cpp
#include "pto/comm/async/rdma/rdma_workspace_manager.hpp"

namespace rdma = pto::comm::rdma;

rdma::RdmaWorkspaceManager manager;
if (manager.Preflight() != rdma::WorkspaceInitResult::READY) {
    return HandleRdmaUnavailable();
}

rdma::WorkspaceConfig config;
config.rankId = rank;
config.rankCount = nranks;
config.phyId = localPhyId;
config.localIp = localRdmaIp;
config.basePort = basePort;
config.peerIps = peerRdmaIps;
config.peerPhyIds = peerPhyIds;
config.peerSymAddrs = peerDeviceAddrs;
config.symmetricAddr = localDeviceAddr;
config.symmetricSize = registeredBytes;

if (manager.Init(config) != rdma::WorkspaceInitResult::READY) {
    return HandleRdmaInitFailure();
}

void *rdmaWorkspace = manager.GetWorkspaceAddr();
// 将 rdmaWorkspace 传给 kernel，并用它构建 DmaEngine::RDMA AsyncSession。
LaunchAndSynchronizeKernel(rdmaWorkspace);

// 必须先释放 HCOMM channel/MR/endpoint，再释放已注册的 Device buffer。
if (!manager.Finalize()) {
    return HandleRdmaFinalizeFailure();
}
```

应用层 bootstrap 可以使用 MPI，但 Manager 本身不创建 MPI/HCCL 通信域。每个
rank 必须提供自身和所有 peer 的物理设备 id、RDMA IPv4 与已注册 Device
虚拟地址；各 rank 地址可以不同。`Preflight()` 会拒绝在非 A5 架构上使用
HNS1825；网卡、provider 或网络配置问题由后续 HCOMM 初始化返回错误。

---

## 信号矩阵清零

**关键**：每次 kernel 执行前必须清零信号矩阵，否则上次的残留值导致同步错误。

```cpp
aclrtMemset(signal_matrix, signal_size, 0, signal_size);
aclrtSynchronizeStream(stream);
```

---

## Kernel 启动函数模式

```cpp
// kernel_launchers.h 声明
void launchCommKernel(uint8_t *data, uint8_t *signal, uint8_t *ctx,
                      int rank, int nranks, void *stream);

// comm_kernel.cpp 实现
void launchCommKernel(uint8_t *data, uint8_t *signal, uint8_t *ctx,
                      int rank, int nranks, void *stream)
{
    CommKernelEntry<<<COMM_BLOCK_NUM, nullptr, stream>>>(
        data, signal, ctx, rank, nranks, COMM_BLOCK_NUM);
}
```

---

## CMakeLists.txt 模板

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_comm_operator)

set(CMAKE_CXX_COMPILER bisheng)
set(CMAKE_CXX_STANDARD 17)

# PTO 头文件路径（优先使用仓库内版本）
set(PTO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include")
include_directories(BEFORE ${PTO_INCLUDE_DIR})

# CANN 环境
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()
include_directories(${ASCEND_HOME}/include)
link_directories(${ASCEND_HOME}/lib64)

# 通信 Kernel（Vec 架构）
add_library(comm_kernel SHARED comm_kernel.cpp)
target_compile_options(comm_kernel PRIVATE
    --cce-aicore-arch=dav-c220-vec
    -DMEMORY_BASE
    -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_options(comm_kernel PRIVATE --cce-fatobj-link)
target_link_libraries(comm_kernel runtime)

# 计算 Kernel（Cube 架构，如需通算融合）
add_library(compute_kernel SHARED compute_kernel.cpp)
target_compile_options(compute_kernel PRIVATE
    --cce-aicore-arch=dav-c220-cube
    -DMEMORY_BASE
    -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_options(compute_kernel PRIVATE --cce-fatobj-link)
target_link_libraries(compute_kernel runtime)

# Host 可执行文件
add_executable(my_operator main.cpp)
target_link_libraries(my_operator
    comm_kernel compute_kernel
    ascendcl hccl tiling_api platform)
```

### 关键配置项

| 配置 | 说明 |
|------|------|
| `--cce-aicore-arch=dav-c220-vec` | 通信 kernel 使用 Vec 架构 |
| `--cce-aicore-arch=dav-c220-cube` | 计算 kernel 使用 Cube 架构 |
| `-DMEMORY_BASE` | 启用远端地址计算宏 |
| `--cce-fatobj-link` | 启用 fat object 链接 |

---

## MPI 运行

```bash
# 单机多卡
mpirun -np 8 ./my_operator

# 多机
mpirun -np 16 -H host1:8,host2:8 ./my_operator
```
