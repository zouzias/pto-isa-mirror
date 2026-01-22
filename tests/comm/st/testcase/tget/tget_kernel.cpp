/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>

#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <string>

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#include <pto/pto-inst.hpp>

// #define __RAW_SHMEM__
#define __ALL_TILE__

template <typename T, size_t count>
__global__ AICORE void TGetKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ T *shmem)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, count);
    StrideDyn stride(count, count, count, count, 1);

    int my_rank = shmem_my_pe();
    int nranks  = shmem_n_pes();
    int next_rank = (my_rank + 1) % nranks;

    __gm__ int32_t *shmem_sync = (__gm__ int32_t *)shmem;
    __gm__ T *shmem_data = (__gm__ T *)((__gm__ T *)shmem + 64 * sizeof(int32_t));
    __gm__ T *send_shmem = (__gm__ T *)((__gm__ T *)shmem_data + 0);

    __ubuf__ T *tmp_buff = reinterpret_cast<__ubuf__ T *>(uint64_t(1024));

    const uint32_t ub_size = 64 * 1024;
    int32_t magic = 459000 + my_rank;

#ifdef __RAW_SHMEM__
    #if defined(ASCEND_SHMEM)
        shmem_mte_put_mem_nbi(send_shmem, src, tmp_buff, ub_size, count, my_rank, EVENT_ID0);
        shmem_quiet();
        shmemi_barrier_core_soft();

        shmemx_signal_op(shmem_sync + 0, magic, SHMEM_SIGNAL_SET, my_rank);
        shmem_signal_wait_until((__gm__ int32_t *)shmem_ptr(shmem_sync, next_rank) + 0, CANN_SHMEM_CMP_EQ, 459000 + next_rank);

        shmem_mte_get_mem_nbi(dst, send_shmem, tmp_buff, ub_size, count, next_rank, EVENT_ID0);
    #elif defined(CANN_SHMEM)
        aclshmemx_mte_put_nbi(send_shmem, src, tmp_buff, ub_size, count, my_rank, EVENT_ID0);
        aclshmem_quiet();
        aclshmemi_barrier_core_soft();

        aclshmemx_signal_op(shmem_sync + 0, magic, CANN_SHMEM_SIGNAL_SET, my_rank);
        aclshmem_signal_wait_until((__gm__ int32_t *)shmem_ptr(shmem_sync, next_rank) + 0, CANN_SHMEM_CMP_EQ, 459000 + next_rank);

        aclshmemx_mte_get_nbi(dst, send_shmem, tmp_buff, ub_size, count, next_rank, EVENT_ID0);
    #endif
#endif    


#ifdef __ALL_TILE__
    int prev_rank = (my_rank + nranks - 1) % nranks;

    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    __gm__ T *recv_shmem = (__gm__ T *)((__gm__ T *)shmem_data + count);

    Global srcG(src, shape, stride);
    Global dstG(dst, shape, stride);

    Global sendG(send_shmem, shape, stride);
    Global recvG(recv_shmem, shape, stride);

    TileData srcBufTile(1, count);
    TileData dstBufTile(1, count);

    TASSIGN(srcBufTile, 0x0);
    TASSIGN(dstBufTile, 0x10000);

    TLOAD(srcBufTile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(sendG, srcBufTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    pto::comm::TQUIET();
    pto::comm::TBARRIER();

    sendG.SetRank(next_rank);
    pto::comm::TGET(recvG, sendG);
    pto::comm::TQUIET();

    TLOAD(dstBufTile, recvG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstG, dstBufTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

#endif
}



template <typename T, size_t count>
bool RunGetRingKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, uint64_t local_mem_size){
    
    int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);
    if(ret != 0){
        std::cerr << "[ERROR] Failed to init shmem tls\n";
        return false;
    }

    if (n_devices <= 0 || n_ranks <= 0) {
        std::cerr << "[ERROR] n_devices and n_ranks must be > 0\n";
        return false;
    }
    // derived from shmem/examples/allgather/main.cpp
    const int32_t device_id = rank_id % n_devices + first_device_id;
    int status = 0;
    aclrtStream stream = nullptr;

    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);


    ShmemEnv env;
    const char *ip = "tcp://127.0.0.1:8766";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    
    if(!ShmemInitFromEnv(env)){
        return false;
    }

    uint64_t fftsAddr = shmemx_get_ffts_config();

    void *input_ptr, *output_ptr;
    aclrtMalloc(&input_ptr, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&output_ptr, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

    uint8_t *input_host, *output_host;
    aclrtMallocHost(reinterpret_cast<void**>(&input_host), count * sizeof(T));
    aclrtMallocHost(reinterpret_cast<void**>(&output_host), count * sizeof(T));

    // 初始化Input/Output Host
    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T*>(input_host)[i] = static_cast<T>(i + (rank_id + 1) * 10000); // 使不同 rank 有不同数据
        reinterpret_cast<T*>(output_host)[i] = static_cast<T>(-1);
    }
    // 把主机数据拷到 device 端
    aclrtMemcpy(input_ptr, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    // 共享内存对称堆的分配 (sync buffer + data buffer)
    void* shmem_ptr = pto::comm::ContextManager::SymmetricAlloc(64 * sizeof(int32_t) + 4 * count * sizeof(T));

    TGetKernelImpl<T, count><<<1, nullptr, stream>>>((T*)output_ptr, (T*)input_ptr, (T*)shmem_ptr);
    status = aclrtSynchronizeStream(stream);

    aclrtMemcpy(output_host, count * sizeof(T), output_ptr, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    for(int i = 0; i < count; ++i){
        T value = reinterpret_cast<T*>(output_host)[i];
        if(value != static_cast<T>(i + ((rank_id + 1) % n_ranks + 1) * 10000)){
            std::cout << "Rank " << rank_id  << " " 
                      << "Expected value: " << static_cast<T>(i + ((rank_id + 1) % n_ranks + 1) * 10000) << " "
                      << "Actual value: " << value << std::endl;
            is_ok = false;
            break;
        }
    }

    status |= aclrtFreeHost(input_host);
    status |= aclrtFreeHost(output_host);
    status |= aclrtFree(input_ptr);
    status |= aclrtFree(output_ptr);
    pto::comm::ContextManager::SymmetricFree(shmem_ptr);


    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}



template <typename T, size_t count>
bool RunGetRing(int n_ranks, int n_devices, int first_rank_id, int first_device_id){
    std::vector<pid_t> pids;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    for (int r = 0; r < n_ranks; ++r){
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunGetRingKernel<T, count>(first_rank_id + r, n_ranks, n_devices, first_device_id, local_mem_size);
            _exit(ok ? 0 : 1); // 子进程退出，避免继续父流程
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            return 1; // fork 失败
        }
    }
    bool success = true;
    for (pid_t p : pids) {
        int status = 0;
        waitpid(p, &status, 0);
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) success = false;
    }
    return success;    
}

// 显式实例化，便于链接
template bool RunGetRing<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunGetRing<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunGetRing<uint8_t, 512>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);