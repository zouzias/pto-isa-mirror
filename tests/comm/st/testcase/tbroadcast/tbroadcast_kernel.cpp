#include <cstddef>
#include <cstdint>

#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#include <pto/pto-inst.hpp>

#define ENABLE_DEBUG_PRINT 1

template <typename T, size_t count>
__global__ AICORE void TBroadCastKernelImpl(__gm__ T *input, __gm__ T *output, int nranks, int root)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    int my_rank = shmem_my_pe();

    ShapeDyn shape(1, 1, 1, 1, count);
    StrideDyn stride(count, count, count, count, 1);
    
    Global tempG(input, shape, stride);
    Global outputG(output, shape, stride);
    Global *tensorPtrs[16];
    Global tensors[16];
    
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        tensors[i] = outputG; // ParallelGroup should be destinations
        tensors[i].SetRank(i);
        tensorPtrs[i] = &tensors[i];
    }
    
    pto::comm::ParallelGroup<Global> pg(tensorPtrs, actual_nranks, my_rank);
    
    pto::comm::TBROADCAST(pg, tempG, root);
    pto::comm::TQUIET();
}

template <typename T, size_t count>
bool RunBroadCastKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int root){
    
    int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);
    if(ret != 0){
        std::cerr << "[ERROR] Failed to init shmem tls\n";
        return false;
    }

    if (n_devices <= 0 || n_ranks <= 0) {
        std::cerr << "[ERROR] n_devices and n_ranks must be > 0\n";
        return false;
    }
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

    void *global_data_ptr;
    global_data_ptr = pto::comm::ContextManager::SymmetricAlloc(count * sizeof(T));

    T *input_host;
    aclrtMallocHost(reinterpret_cast<void**>(&input_host), count * sizeof(T));

    T* output_host;
    aclrtMallocHost(reinterpret_cast<void**>(&output_host), count * sizeof(T));

    T* output_device;
    output_device = (T*)pto::comm::ContextManager::SymmetricAlloc(count * sizeof(T));

    // Initialize input data: Rank root has data i + root * 100, others have 0
    for (size_t i = 0; i < count; ++i) {
        if (rank_id == root) {
            input_host[i] = static_cast<T>(i + rank_id * 100);
        } else {
            input_host[i] = static_cast<T>(0);
        }
        output_host[i] = static_cast<T>(0);
    }

    aclrtMemcpy(global_data_ptr, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(output_device, count * sizeof(T), output_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

#if ENABLE_DEBUG_PRINT
    if (rank_id == root) {
        std::cout << "[DEBUG] Rank " << rank_id << " (Root) input: ";
        for (int i = 0; i < 5 && i < count; ++i) std::cout << (float)input_host[i] << " ";
        std::cout << std::endl;
    }
#endif

    TBroadCastKernelImpl<T, count><<<1, nullptr, stream>>>((T*)global_data_ptr, (T*)output_device, n_ranks, root);
    status = aclrtSynchronizeStream(stream);

    aclrtMemcpy(output_host, count * sizeof(T), output_device, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    // Verify: All ranks should have data Sum_{R=root} (i + root * 100)
    bool is_ok = true;
    for (size_t i = 0; i < count; ++i) {
        T expected = static_cast<T>(i + root * 100);
        T actual = output_host[i];
        if (actual != expected) {
            std::cout << "Rank " << rank_id << " validation failed at index " << i 
                      << ": expected " << (float)expected << ", got " << (float)actual << std::endl;
            is_ok = false;
            break;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == (root + 1) % n_ranks) { // Print from one non-root rank if possible
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank " << rank_id << ": TBROADCAST SUCCESSFUL!" << std::endl;
        std::cout << "Sample Result (First 5 elements): [ ";
        for (size_t i = 0; i < (count > 5 ? 5 : count); ++i) {
            std::cout << (float)output_host[i] << " ";
        }
        if (count > 5) std::cout << "... ";
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    pto::comm::ContextManager::SymmetricFree(output_device);
    pto::comm::ContextManager::SymmetricFree(global_data_ptr);

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

template <typename T, size_t count>
bool RunBroadCast(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root){
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r){
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunBroadCastKernel<T, count>(first_rank_id + r, n_ranks, n_devices, first_device_id, root);
            _exit(ok ? 0 : 1);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            return false;
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

// Explicit instantiations
template bool RunBroadCast<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);
template bool RunBroadCast<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);

