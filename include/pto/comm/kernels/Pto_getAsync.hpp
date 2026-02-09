/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_GET_ASYNC_HPP
#define PTO_GET_ASYNC_HPP

#include <acl/acl.h>
#include <pto/pto-inst.hpp>
#include "pto/comm/TGetAsync.hpp"
#include "pto/comm/pto_comm_inst.hpp"

namespace pto {
namespace comm {

// ============================================================================
// Constants
// ============================================================================
constexpr uint32_t kPtoGetAsyncTileBytes = 64U * 1024U;  // 64KB tile size
constexpr uint32_t kPtoGetAsyncAlignment = 64U;          // 64-byte alignment requirement

#define PTO_AIV_ATTR __attribute__((aiv))
#define PTO_GET_ASYNC_DEVICE_ENABLED 1

// ============================================================================
// P2P Access Management Helper Functions (shared with Pto_putAsync.hpp)
// ============================================================================
namespace detail {

// Check if P2P access is supported between two devices
inline bool CanAccessPeer(int32_t srcDeviceId, int32_t dstDeviceId)
{
    int32_t canAccessPeer = 0;
    aclError ret = aclrtDeviceCanAccessPeer(&canAccessPeer, srcDeviceId, dstDeviceId);
    return (ret == ACL_SUCCESS && canAccessPeer == 1);
}

// Enable bidirectional P2P access between two devices
// Must be called before performing inter-device memory copy
inline aclError EnablePeerAccessBidirectional(int32_t deviceId1, int32_t deviceId2)
{
    aclError ret;
    
    // Enable Device1 to access Device2
    ret = aclrtSetDevice(deviceId1);
    if (ret != ACL_SUCCESS) return ret;
    ret = aclrtDeviceEnablePeerAccess(deviceId2, 0);
    if (ret != ACL_SUCCESS) return ret;
    
    // Enable Device2 to access Device1
    ret = aclrtSetDevice(deviceId2);
    if (ret != ACL_SUCCESS) return ret;
    ret = aclrtDeviceEnablePeerAccess(deviceId1, 0);
    if (ret != ACL_SUCCESS) return ret;
    
    return ACL_SUCCESS;
}

} // namespace detail

// ============================================================================
// Device-side Implementation (AIV Kernel)
// ============================================================================
#if PTO_GET_ASYNC_DEVICE_ENABLED
namespace detail {

// Kernel body: Construct GlobalTensor and call TGET_ASYNC
template <typename DType>
PTO_INTERNAL void PtoGetAsyncKernelBody(
    __gm__ DType* dst,
    __gm__ DType* src,
    uint64_t total_elems)
{
    // Define dynamic Shape and Stride for 1D tensor
    using GetShape = Shape<1, 1, 1, 1, DYNAMIC>;
    using GetStride = Stride<1, 1, 1, DYNAMIC, 1>;
    
    GetShape dyn_shape(1, 1, 1, 1, static_cast<int>(total_elems));
    GetStride dyn_stride(1, 1, 1, static_cast<int>(total_elems), 1);
    
    // Construct GlobalTensor for dst and src
    GlobalTensor<DType, GetShape, GetStride> dstGlobal(dst, dyn_shape, dyn_stride);
    GlobalTensor<DType, GetShape, GetStride> srcGlobal(src, dyn_shape, dyn_stride);
    
    // Call TGET_ASYNC from TGetAsync.hpp
    auto get_event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::SDMA>(dstGlobal, srcGlobal);
}

} // namespace detail

// Generic get async kernel: perform async D2D transfer using TGET_ASYNC
__global__ AICORE PTO_AIV_ATTR void PTO_GET_ASYNC_AIV(
    __gm__ uint8_t* dst,
    __gm__ uint8_t* src,
    uint64_t total_bytes)
{
    detail::PtoGetAsyncKernelBody<uint8_t>(dst, src, total_bytes);
}
#endif // PTO_GET_ASYNC_DEVICE_ENABLED

// ============================================================================
// PTO_GET_ASYNC: Host wrapper for async D2D memory copy (GET operation)
//
// GET operation: Read data from remote device to local device
// Data flow: remoteSrc (remote Device) → localDst (local Device)
//
// Supports two execution paths:
// - Host SDMA path (HostSdma=true): Host-initiated aclrtMemcpyAsync
// - AIV SDAM path (HostSdma=false): Launch AIV kernel calling TGET_ASYNC_SDMA_IMPL
//
// Requirements:
// - Source and destination addresses must be 64-byte aligned
// - P2P access must be enabled before calling (use EnablePeerAccessBidirectional)
// - Only supports devices within the same PCIe Switch
// - Only supports same process, same or different threads
//
// @tparam HostSdma    If true, use Host SDMA (aclrtMemcpyAsync); if false, use AIV SDMA kernel
// @tparam AivCores   Number of AIV cores when HostSdma=false (must be > 0)
// @param dst         Destination memory address (local Device)
// @param dst_bytes   Destination memory size in bytes
// @param src         Source memory address (remote Device)
// @param src_bytes   Source memory size in bytes
// @param stream      Async stream for the operation
// @return            aclError error code (ACL_SUCCESS on success)
// ============================================================================
template <bool HostSdma = true, int AivCores = -1>
aclError PTO_GET_ASYNC(
    void* dst,
    size_t dst_bytes,
    const void* src,
    size_t src_bytes,
    aclrtStream stream)
{
    if (src_bytes == 0 || dst_bytes == 0) {
        return ACL_SUCCESS;
    }
    
    const size_t transfer_bytes = (src_bytes < dst_bytes) ? src_bytes : dst_bytes;
    
    if constexpr (HostSdma) {
        // SDMA path: Host-initiated aclrtMemcpyAsync
        // ACL_MEMCPY_DEVICE_TO_DEVICE: Memory copy within Device or between two Devices
        return aclrtMemcpyAsync(
            dst,                           // dst: local destination address
            dst_bytes,                     // destMax: max destination memory size
            src,                           // src: remote source address
            transfer_bytes,                // count: bytes to copy
            ACL_MEMCPY_DEVICE_TO_DEVICE,  // kind: D2D copy
            stream                         // stream: async stream
        );
    } else {
        // AIV path: Launch kernel that calls TGET_ASYNC
        static_assert(AivCores > 0, "AivCores must be > 0 when HostSdma is false");
        PTO_GET_ASYNC_AIV<<<AivCores, nullptr, stream>>>(
            (__gm__ uint8_t*)dst,
            (__gm__ uint8_t*)const_cast<void*>(src),
            transfer_bytes
        );
        return ACL_SUCCESS;
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_GET_ASYNC_HPP
