/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "sdma_host_init.h"

#include <dlfcn.h>
#include <functional>
#include <numeric>
#include <vector>
#include <cstring>
#include <iostream>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

// ============================================================================
// Internal Type Definitions
// ============================================================================

typedef enum {
    RT_STREAM_CREATE_ATTR_FLAGS = 1,    // Stream creation flags
    RT_STREAM_CREATE_ATTR_PRIORITY = 2, // Stream priority
    RT_STREAM_CREATE_ATTR_MAX = 3
} rtStreamCreateAttrId;

typedef union {
    uint32_t flags;
    uint32_t priority;
    uint32_t rsv[4];
} rtStreamCreateAttrValue_t;

typedef struct {
    rtStreamCreateAttrId id;         // Attribute ID
    rtStreamCreateAttrValue_t value; // Attribute value
} rtStreamCreateAttr_t;

typedef struct {
    rtStreamCreateAttr_t *attrs; // Array of attributes
    size_t numAttrs;             // Number of attributes
} rtStreamCreateConfig_t;

// Function pointer types for runtime APIs
using rtsStreamCreateFunc = rtError_t (*)(rtStream_t *, rtStreamCreateConfig_t *);
using rtGetDeviceInfoFunc = rtError_t (*)(uint32_t, int32_t, int32_t, int64_t *);

// ============================================================================
// Global Variables
// ============================================================================

pto_sdma_op_res_info_t g_pto_sdma_op_res_info;
void *g_pto_sdma_op_res_info_device_ptr = nullptr;
uint64_t g_pto_sdma_workspace_addr = 0;

// ============================================================================
// Internal Helper Macros
// ============================================================================

#define PTO_SDMA_CHECK_RET(expr) \
    do { \
        auto _ret = (expr); \
        if (_ret != 0) { \
            std::cerr << "[PTO_SDMA] Error: " << #expr << " failed with code " << _ret << std::endl; \
            return PTO_SDMA_ERROR; \
        } \
    } while (0)

#define PTO_SDMA_LOG_INFO(msg) \
    std::cout << "[PTO_SDMA] INFO: " << msg << std::endl

#define PTO_SDMA_LOG_ERROR(msg) \
    std::cerr << "[PTO_SDMA] ERROR: " << msg << std::endl

#define PTO_SDMA_LOG_DEBUG(msg) \
    std::cout << "[PTO_SDMA] DEBUG: " << msg << std::endl

// ============================================================================
// Internal Helper Functions
// ============================================================================

namespace {

/**
 * @brief Create an ACL tensor for AICPU kernel input/output
 * @tparam T Data type of the tensor elements
 * @param host_data Host-side data to copy to device
 * @param shape Shape of the tensor
 * @param device_addr Output: allocated device address
 * @param data_type ACL data type
 * @param tensor Output: created ACL tensor
 * @return 0 on success, non-zero on failure
 */
template <typename T>
int create_acl_tensor(const std::vector<T> &host_data, const std::vector<int64_t> &shape,
                      void **device_addr, aclDataType data_type, aclTensor **tensor)
{
    auto size = std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1),
        [](int64_t a, int64_t b) { return a * b; }) * sizeof(T);
    
    PTO_SDMA_CHECK_RET(aclrtMalloc(device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST));
    PTO_SDMA_CHECK_RET(aclrtMemcpy(*device_addr, size, host_data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), data_type, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *device_addr);
    return 0;
}

/**
 * @brief Create SDMA streams for data transfer
 * @return 0 on success, non-zero on failure
 */
int create_sdma_streams()
{
    // Load runtime library
    auto rt_handle = dlopen("libruntime.so", RTLD_NOW);
    if (rt_handle == nullptr) {
        PTO_SDMA_LOG_ERROR("Failed to open libruntime.so, error: " << dlerror());
        return PTO_SDMA_ERROR;
    }

    // Get rtsStreamCreate function pointer
    auto rts_stream_create = reinterpret_cast<rtsStreamCreateFunc>(dlsym(rt_handle, "rtsStreamCreate"));
    auto dlsym_err = dlerror();
    if (dlsym_err || rts_stream_create == nullptr) {
        PTO_SDMA_LOG_ERROR("Failed to load rtsStreamCreate, error: " << (dlsym_err ? dlsym_err : "NULL"));
        dlclose(rt_handle);
        return PTO_SDMA_ERROR;
    }

    // Get rtGetDeviceInfo function pointer
    auto rt_get_device_info = reinterpret_cast<rtGetDeviceInfoFunc>(dlsym(rt_handle, "rtGetDeviceInfo"));
    dlsym_err = dlerror();
    if (dlsym_err || rt_get_device_info == nullptr) {
        PTO_SDMA_LOG_ERROR("Failed to load rtGetDeviceInfo, error: " << (dlsym_err ? dlsym_err : "NULL"));
        dlclose(rt_handle);
        return PTO_SDMA_ERROR;
    }

    // Get current device ID
    int32_t device_id = -1;
    PTO_SDMA_CHECK_RET(aclrtGetDevice(&device_id));

    // Get physical die ID
    int64_t die_id = -1;
    const int INFO_TYPE_PHY_DIE_ID = 19;
    PTO_SDMA_CHECK_RET(rt_get_device_info(device_id, 0, INFO_TYPE_PHY_DIE_ID, &die_id));
    PTO_SDMA_LOG_INFO("Device ID: " << device_id << ", Die ID: " << die_id);

    // Create SDMA streams
    for (int i = 0; i < PTO_SDMA_MAX_CHAN; i++) {
        g_pto_sdma_op_res_info.streams[i].stream_ = 0;

        rtStream_t rt_stream = nullptr;
        constexpr size_t num_attrs = 2;
        rtStreamCreateAttr_t attrs[num_attrs];
        attrs[0].id = RT_STREAM_CREATE_ATTR_PRIORITY;
        attrs[0].value.priority = 0;
        attrs[1].id = RT_STREAM_CREATE_ATTR_FLAGS;
        attrs[1].value.flags = 0x800;  // SDMA stream flag
        rtStreamCreateConfig_t config = {attrs, num_attrs};
        
        auto rt_err = rts_stream_create(&rt_stream, &config);
        if (rt_err != RT_ERROR_NONE) {
            PTO_SDMA_LOG_ERROR("Failed to create stream " << i);
            dlclose(rt_handle);
            return PTO_SDMA_ERROR;
        }
        g_pto_sdma_op_res_info.streams[i].stream_ = reinterpret_cast<uint64_t>(rt_stream);

        // Get stream ID
        int32_t stream_id = 0;
        auto acl_ret = aclrtStreamGetId(reinterpret_cast<aclrtStream>(g_pto_sdma_op_res_info.streams[i].stream_),
                                        &stream_id);
        if (acl_ret != ACL_SUCCESS) {
            PTO_SDMA_LOG_ERROR("Failed to get stream ID " << i);
            dlclose(rt_handle);
            return PTO_SDMA_ERROR;
        }

        // Get SQ ID
        uint32_t sq_id = 0;
        auto rt_ret = rtStreamGetSqid(reinterpret_cast<rtStream_t>(g_pto_sdma_op_res_info.streams[i].stream_),
                                      &sq_id);
        if (rt_ret != RT_ERROR_NONE) {
            PTO_SDMA_LOG_ERROR("Failed to get SQ ID " << i);
            dlclose(rt_handle);
            return PTO_SDMA_ERROR;
        }

        // Get CQ ID
        uint32_t cq_id = 0;
        uint32_t logic_cq_id = 0;
        rt_ret = rtStreamGetCqid(reinterpret_cast<rtStream_t>(g_pto_sdma_op_res_info.streams[i].stream_),
                                 &cq_id, &logic_cq_id);
        if (rt_ret != RT_ERROR_NONE) {
            PTO_SDMA_LOG_ERROR("Failed to get CQ ID " << i);
            dlclose(rt_handle);
            return PTO_SDMA_ERROR;
        }

        // Get current context
        aclrtContext ctx = nullptr;
        acl_ret = aclrtGetCurrentContext(&ctx);
        if (acl_ret != ACL_SUCCESS) {
            PTO_SDMA_LOG_ERROR("Failed to get context " << i);
            dlclose(rt_handle);
            return PTO_SDMA_ERROR;
        }

        // Store stream info
        g_pto_sdma_op_res_info.streams[i].ctx_ = reinterpret_cast<uint64_t>(ctx);
        g_pto_sdma_op_res_info.streams[i].stream_id = stream_id;
        g_pto_sdma_op_res_info.streams[i].sq_id = sq_id;
        g_pto_sdma_op_res_info.streams[i].cq_id = cq_id;
        g_pto_sdma_op_res_info.streams[i].logic_cq_id = logic_cq_id;
        g_pto_sdma_op_res_info.streams[i].dev_id = static_cast<int32_t>(die_id);

        PTO_SDMA_LOG_DEBUG("Created stream " << i << ": stream=" << g_pto_sdma_op_res_info.streams[i].stream_
                          << ", dev_id=" << g_pto_sdma_op_res_info.streams[i].dev_id
                          << ", stream_id=" << stream_id << ", sq_id=" << sq_id
                          << ", cq_id=" << cq_id << ", logic_cq_id=" << logic_cq_id);
    }

    dlclose(rt_handle);
    PTO_SDMA_LOG_INFO("Successfully created " << PTO_SDMA_MAX_CHAN << " SDMA streams");
    return PTO_SDMA_SUCCESS;
}

/**
 * @brief Run AICPU kernel to initialize device-side SDMA mapping
 * @param streams_addr Address of stream info on device
 * @param sdma_workspace_addr Address of SDMA workspace
 * @param stream AICPU stream to run the kernel on
 * @return 0 on success, non-zero on failure
 */
int run_aicpu_kernel(uint64_t streams_addr, uint64_t sdma_workspace_addr, aclrtStream stream)
{
    // Function pointer types for ACLNN APIs
    using aclnnSdmaMapGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, aclTensor *,
                                                              uint64_t *, aclOpExecutor **);
    using aclnnSdmaMapFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    // Load opapi library
    auto opapi_handle = dlopen("libopapi.so", RTLD_NOW);
    if (opapi_handle == nullptr) {
        PTO_SDMA_LOG_ERROR("Failed to open libopapi.so, error: " << dlerror());
        return PTO_SDMA_ERROR;
    }

    // Get aclnnSdmaMapGetWorkspaceSize function pointer
    auto aclnn_sdma_map_get_workspace_size = reinterpret_cast<aclnnSdmaMapGetWorkspaceSizeFunc>(
        dlsym(opapi_handle, "aclnnSdmaMapGetWorkspaceSize"));
    auto dlsym_err = dlerror();
    if (dlsym_err || aclnn_sdma_map_get_workspace_size == nullptr) {
        PTO_SDMA_LOG_ERROR("Failed to load aclnnSdmaMapGetWorkspaceSize, error: "
                          << (dlsym_err ? dlsym_err : "NULL"));
        dlclose(opapi_handle);
        return PTO_SDMA_ERROR;
    }

    // Get aclnnSdmaMap function pointer
    auto aclnn_sdma_map = reinterpret_cast<aclnnSdmaMapFunc>(dlsym(opapi_handle, "aclnnSdmaMap"));
    dlsym_err = dlerror();
    if (dlsym_err || aclnn_sdma_map == nullptr) {
        PTO_SDMA_LOG_ERROR("Failed to load aclnnSdmaMap, error: " << (dlsym_err ? dlsym_err : "NULL"));
        dlclose(opapi_handle);
        return PTO_SDMA_ERROR;
    }

    // Prepare input/output tensors
    std::vector<int64_t> input_shape = {2};
    std::vector<int64_t> output_shape = {1};
    void *input_device_addr = nullptr;
    void *output_device_addr = nullptr;
    aclTensor *input = nullptr;
    aclTensor *output = nullptr;
    std::vector<uint64_t> input_host_data = {streams_addr, sdma_workspace_addr};
    std::vector<uint64_t> output_host_data = {0};

    // Create input tensor
    auto ret = create_acl_tensor(input_host_data, input_shape, &input_device_addr,
                                 aclDataType::ACL_UINT64, &input);
    if (ret != 0) {
        dlclose(opapi_handle);
        return ret;
    }

    // Create output tensor
    ret = create_acl_tensor(output_host_data, output_shape, &output_device_addr,
                            aclDataType::ACL_UINT64, &output);
    if (ret != 0) {
        if (input_device_addr) aclrtFree(input_device_addr);
        if (input) aclDestroyTensor(input);
        dlclose(opapi_handle);
        return ret;
    }

    // Call ACLNN two-stage interface
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    PTO_SDMA_CHECK_RET(aclnn_sdma_map_get_workspace_size(input, output, &workspace_size, &executor));

    void *workspace = nullptr;
    if (workspace_size > 0) {
        PTO_SDMA_CHECK_RET(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    PTO_SDMA_CHECK_RET(aclnn_sdma_map(workspace, workspace_size, executor, stream));
    PTO_SDMA_CHECK_RET(aclrtSynchronizeStream(stream));

    // Cleanup
    if (workspace) aclrtFree(workspace);
    if (input_device_addr) aclrtFree(input_device_addr);
    if (output_device_addr) aclrtFree(output_device_addr);
    if (input) aclDestroyTensor(input);
    if (output) aclDestroyTensor(output);
    dlclose(opapi_handle);

    return PTO_SDMA_SUCCESS;
}

} // anonymous namespace

// ============================================================================
// Public API Implementation
// ============================================================================

extern "C" {

int pto_sdma_init(struct pto_comm_init_attr_t *attributes)
{
    PTO_SDMA_LOG_INFO("Starting SDMA initialization...");

    // Clear any previous errors
    dlerror();

    // Step 1: Create AICPU stream for kernel execution
    aclrtStream aicpu_stream = nullptr;
    PTO_SDMA_CHECK_RET(aclrtCreateStreamWithConfig(&aicpu_stream, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));
    
    aclrtStreamAttrValue value;
    value.failureMode = 1;  // Stop on error
    PTO_SDMA_CHECK_RET(aclrtSetStreamAttribute(aicpu_stream, ACL_STREAM_ATTR_FAILURE_MODE, &value));
    PTO_SDMA_LOG_INFO("Created AICPU stream");

    // Step 2: Create SDMA streams (40 channels)
    auto ret = create_sdma_streams();
    if (ret != PTO_SDMA_SUCCESS) {
        aclrtDestroyStream(aicpu_stream);
        return ret;
    }
    PTO_SDMA_LOG_INFO("Created SDMA streams");

    // Step 3: Allocate shared workspace memory (16KB for AICPU and AIV communication)
    constexpr size_t workspace_size = 16 * 1024;  // 16KB
    void *sdma_workspace = nullptr;
    PTO_SDMA_CHECK_RET(aclrtMalloc(&sdma_workspace, workspace_size,
        static_cast<aclrtMemMallocPolicy>(ACL_MEM_TYPE_HIGH_BAND_WIDTH | ACL_MEM_MALLOC_HUGE_FIRST)));
    PTO_SDMA_CHECK_RET(aclrtMemset(sdma_workspace, workspace_size, 0, workspace_size));
    
    g_pto_sdma_workspace_addr = reinterpret_cast<uint64_t>(sdma_workspace);
    g_pto_sdma_op_res_info.workspace_addr = g_pto_sdma_workspace_addr;
    PTO_SDMA_LOG_INFO("Allocated SDMA workspace at: 0x" << std::hex << g_pto_sdma_workspace_addr << std::dec);

    // Step 4: Copy resource info to device memory (H2D)
    size_t res_info_size = sizeof(g_pto_sdma_op_res_info);
    PTO_SDMA_CHECK_RET(aclrtMalloc(&g_pto_sdma_op_res_info_device_ptr, res_info_size,
        static_cast<aclrtMemMallocPolicy>(ACL_MEM_TYPE_HIGH_BAND_WIDTH | ACL_MEM_MALLOC_HUGE_FIRST)));
    PTO_SDMA_CHECK_RET(aclrtMemset(g_pto_sdma_op_res_info_device_ptr, res_info_size, 0, res_info_size));
    PTO_SDMA_CHECK_RET(aclrtMemcpy(g_pto_sdma_op_res_info_device_ptr, res_info_size,
                                   &g_pto_sdma_op_res_info, res_info_size, ACL_MEMCPY_HOST_TO_DEVICE));
    PTO_SDMA_LOG_INFO("Copied resource info to device at: " << g_pto_sdma_op_res_info_device_ptr);

    // Step 5: Run AICPU kernel to initialize device-side SDMA mapping
    ret = run_aicpu_kernel(reinterpret_cast<uint64_t>(g_pto_sdma_op_res_info_device_ptr),
                           g_pto_sdma_op_res_info.workspace_addr, aicpu_stream);
    if (ret != PTO_SDMA_SUCCESS) {
        aclrtDestroyStream(aicpu_stream);
        return ret;
    }
    PTO_SDMA_LOG_INFO("AICPU kernel executed successfully");

    // Cleanup AICPU stream
    PTO_SDMA_CHECK_RET(aclrtDestroyStream(aicpu_stream));

    PTO_SDMA_LOG_INFO("SDMA initialization completed successfully");
    return PTO_SDMA_SUCCESS;
}

int pto_sdma_finalize(void)
{
    PTO_SDMA_LOG_INFO("Starting SDMA finalization...");

    // Destroy SDMA streams
    for (int i = 0; i < PTO_SDMA_MAX_CHAN; i++) {
        if (g_pto_sdma_op_res_info.streams[i].stream_ != 0) {
            auto ret = aclrtDestroyStreamForce(
                reinterpret_cast<aclrtStream>(g_pto_sdma_op_res_info.streams[i].stream_));
            if (ret != ACL_SUCCESS) {
                PTO_SDMA_LOG_ERROR("Failed to destroy stream " << i);
            }
            g_pto_sdma_op_res_info.streams[i].stream_ = 0;
        }
    }
    PTO_SDMA_LOG_INFO("Destroyed SDMA streams");

    // Free workspace
    if (g_pto_sdma_workspace_addr != 0) {
        auto ret = aclrtFree(reinterpret_cast<void *>(g_pto_sdma_workspace_addr));
        if (ret != ACL_SUCCESS) {
            PTO_SDMA_LOG_ERROR("Failed to free workspace");
        }
        g_pto_sdma_workspace_addr = 0;
        g_pto_sdma_op_res_info.workspace_addr = 0;
    }
    PTO_SDMA_LOG_INFO("Freed workspace");

    // Free device-side resource info
    if (g_pto_sdma_op_res_info_device_ptr != nullptr) {
        auto ret = aclrtFree(g_pto_sdma_op_res_info_device_ptr);
        if (ret != ACL_SUCCESS) {
            PTO_SDMA_LOG_ERROR("Failed to free device resource info");
        }
        g_pto_sdma_op_res_info_device_ptr = nullptr;
    }
    PTO_SDMA_LOG_INFO("Freed device resource info");

    PTO_SDMA_LOG_INFO("SDMA finalization completed successfully");
    return PTO_SDMA_SUCCESS;
}

} // extern "C"
