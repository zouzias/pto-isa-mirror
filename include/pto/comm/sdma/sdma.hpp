/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_SDMA_SDMA_HPP
#define PTO_COMM_SDMA_SDMA_HPP

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/sdma/sdma_types.hpp"
#include "pto/comm/sdma/sdma_device_impl.hpp"
// TODO: Temporarily disabled, will be enabled when init implementation is restored
// #include "pto/comm/sdma/sdma_host_init.h"
#include <cstdint>

// Forward declaration for pto_comm_init_attr_t when sdma_host_init.h is disabled
#ifndef PTO_COMM_SDMA_HOST_INIT_H
struct pto_comm_init_attr_t;
#endif

namespace pto {
namespace comm {
namespace sdma {

// pto_comm_init_attr_t is defined in sdma_host_init.h

// ============================================================================
// SDMA: SDMA engine wrapper class
//
// This class provides a high-level interface for SDMA operations including
// initialization and data transfer operations (TPut, TGet).
//
// Usage:
//   // Initialize SDMA (typically called once at startup, host side)
//   pto::comm::sdma::SDMA::init(attributes);
//
//   // PUT: Write local data to remote PE (device side)
//   auto event = pto::comm::sdma::SDMA::put(remoteDstGlobal, localSrcGlobal);
//   pto::comm::sdma::SDMA::wait(event);
//
//   // GET: Read remote data to local PE (device side)
//   auto event = pto::comm::sdma::SDMA::get(localDstGlobal, remoteSrcGlobal);
//   pto::comm::sdma::SDMA::wait(event);
// ============================================================================

class SDMA {
public:
    // ========================================================================
    // Initialization (Host Side)
    // ========================================================================
    
    // Initialize SDMA engine
    // This function calls the host-side initialization function to set up
    // SDMA resources including streams, workspace, and device-side mapping.
    //
    // Parameters:
    //   - attributes: Initialization attributes (can be nullptr for default)
    // Returns:
    //   - true on success, false on failure
    //
    // Note: This should be called from host-side code during initialization
    static bool init(struct pto_comm_init_attr_t *attributes);
    
    // ========================================================================
    // PUT: Asynchronous remote write operation (Device Side)
    // ========================================================================
    
    // PUT using GlobalTensor shape to determine transfer size
    template <typename GlobalDstData, typename GlobalSrcData>
    PTO_INTERNAL static SdmaEvent put(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
    {
        return put_impl(dstGlobal, srcGlobal);
    }
    
    // PUT with explicit size specification
    template <typename GlobalDstData, typename GlobalSrcData>
    PTO_INTERNAL static SdmaEvent put(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal,
                         uint64_t transfer_size)
    {
        return put_impl(dstGlobal, srcGlobal, transfer_size);
    }
    
    // ========================================================================
    // GET: Asynchronous remote read operation (Device Side)
    // ========================================================================
    
    // GET using GlobalTensor shape to determine transfer size
    // Data flow: srcGlobal (remote GM) -> dstGlobal (local GM)
    template <typename GlobalDstData, typename GlobalSrcData>
    PTO_INTERNAL static SdmaEvent get(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
    {
        return get_impl(dstGlobal, srcGlobal);
    }
    
    // GET with explicit size specification
    template <typename GlobalDstData, typename GlobalSrcData>
    PTO_INTERNAL static SdmaEvent get(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal,
                         uint64_t transfer_size)
    {
        return get_impl(dstGlobal, srcGlobal, transfer_size);
    }
    
    // ========================================================================
    // Synchronization
    // ========================================================================
    
    // Wait for SDMA event completion
    // Note: Current implementation is synchronous, so wait is a no-op
    // (the put/get operations already complete before returning)
    PTO_INTERNAL static void wait(const SdmaEvent &event)
    {
        // Current implementation is synchronous - operations complete
        // before returning from put/get, so nothing to wait for.
        // Future async implementation would poll/wait here.
        (void)event;
    }
    
    // Test if SDMA event is complete (non-blocking)
    // Note: Current implementation is synchronous, always returns true
    PTO_INTERNAL static bool test(const SdmaEvent &event)
    {
        // Current implementation is synchronous - operations complete
        // before returning from put/get, so always complete.
        (void)event;
        return true;
    }

private:
    // Internal implementation of PUT
    template <typename GlobalDstData, typename GlobalSrcData>
    PTO_INTERNAL static SdmaEvent put_impl(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
    {
        // Debug: Check device state before accessing
        __gm__ detail::pto_comm_global_state_t* device_state = detail::pto_comm_get_state();
        if (device_state == nullptr) {
            // Device state is null - shmem not initialized properly
            return SdmaEvent(0);
        }
        
        // Check if SDMA workspace is initialized (non-zero address)
        if (device_state->sdma_workspace_addr == 0) {
            // SDMA workspace not initialized
            return SdmaEvent(0);
        }
        
        __gm__ detail::pto_sdma_op_res_info_t* op_res_info = detail::pto_comm_get_sdma_op_res_info();
        if (op_res_info == nullptr) {
            return SdmaEvent(0);
        }
        
        using T = typename GlobalSrcData::RawDType;
        
        static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>,
            "SDMA::put: src/dst element type mismatch");
        static_assert(GlobalSrcData::layout == GlobalDstData::layout,
            "SDMA::put: src/dst layout mismatch");
        
        constexpr size_t elemSize = sizeof(T);
        static_assert(elemSize == 1 || elemSize == 2 || elemSize == 4 || elemSize == 8,
            "SDMA::put: Element size must be 1, 2, 4, or 8 bytes");
        
        // Calculate transfer size from tensor shape
        uint64_t totalElements = 1;
        totalElements *= static_cast<uint64_t>(srcGlobal.GetShape(GlobalTensorDim::DIM_0));
        totalElements *= static_cast<uint64_t>(srcGlobal.GetShape(GlobalTensorDim::DIM_1));
        totalElements *= static_cast<uint64_t>(srcGlobal.GetShape(GlobalTensorDim::DIM_2));
        totalElements *= static_cast<uint64_t>(srcGlobal.GetShape(GlobalTensorDim::DIM_3));
        totalElements *= static_cast<uint64_t>(srcGlobal.GetShape(GlobalTensorDim::DIM_4));
        uint64_t transferSize = totalElements * sizeof(T);
        
        if (transferSize == 0) {
            return SdmaEvent(0);
        }
        
        // Get source and destination addresses
        __gm__ void* srcAddr = const_cast<__gm__ void*>(
            reinterpret_cast<__gm__ const void*>(srcGlobal.data()));
        __gm__ void* dstAddr = const_cast<__gm__ void*>(
            reinterpret_cast<__gm__ const void*>(dstGlobal.data()));
        
        // Select an SDMA channel
        uint32_t channel_idx = detail::pto_comm_select_sdma_channel(
            static_cast<uint32_t>(reinterpret_cast<uint64_t>(srcAddr) % SDMA_MAX_CHAN));
        
        // Call device-side put implementation
        detail::put<T>(reinterpret_cast<__gm__ T*>(dstAddr), 
                       reinterpret_cast<__gm__ T*>(srcAddr), 
                       transferSize);
        
        // Generate event ID
        uint64_t event_id = (static_cast<uint64_t>(channel_idx) << 32) | 
                            (reinterpret_cast<uint64_t>(srcAddr) & 0xFFFFFFFF);
        
        return SdmaEvent(event_id);
    }
    
    // Internal implementation of PUT with explicit size
    template <typename GlobalDstData, typename GlobalSrcData>
    PTO_INTERNAL static SdmaEvent put_impl(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal,
                              uint64_t transfer_size)
    {
        __gm__ detail::pto_sdma_op_res_info_t* op_res_info = detail::pto_comm_get_sdma_op_res_info();
        if (op_res_info == nullptr) {
            return SdmaEvent(0);
        }
        
        using T = typename GlobalSrcData::RawDType;
        
        static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>,
            "SDMA::put: src/dst element type mismatch");
        static_assert(GlobalSrcData::layout == GlobalDstData::layout,
            "SDMA::put: src/dst layout mismatch");
        
        constexpr size_t elemSize = sizeof(T);
        static_assert(elemSize == 1 || elemSize == 2 || elemSize == 4 || elemSize == 8,
            "SDMA::put: Element size must be 1, 2, 4, or 8 bytes");
        
        if (transfer_size == 0) {
            return SdmaEvent(0);
        }
        
        __gm__ void* srcAddr = const_cast<__gm__ void*>(
            reinterpret_cast<__gm__ const void*>(srcGlobal.data()));
        __gm__ void* dstAddr = const_cast<__gm__ void*>(
            reinterpret_cast<__gm__ const void*>(dstGlobal.data()));
        
        uint32_t channel_idx = detail::pto_comm_select_sdma_channel(
            static_cast<uint32_t>(reinterpret_cast<uint64_t>(srcAddr) % SDMA_MAX_CHAN));
        
        // Call device-side put implementation
        detail::put<T>(reinterpret_cast<__gm__ T*>(dstAddr), 
                       reinterpret_cast<__gm__ T*>(srcAddr), 
                       transfer_size);
        
        uint64_t event_id = (static_cast<uint64_t>(channel_idx) << 32) | 
                            (reinterpret_cast<uint64_t>(srcAddr) & 0xFFFFFFFF);
        
        return SdmaEvent(event_id);
    }
    
    // Internal implementation of GET
    // Data flow: srcGlobal (remote) -> dstGlobal (local)
    template <typename GlobalDstData, typename GlobalSrcData>
    PTO_INTERNAL static SdmaEvent get_impl(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
    {
        // Debug: Check device state before accessing
        __gm__ detail::pto_comm_global_state_t* device_state = detail::pto_comm_get_state();
        if (device_state == nullptr) {
            // Device state is null - shmem not initialized properly
            return SdmaEvent(0);
        }
        
        // Check if SDMA workspace is initialized (non-zero address)
        if (device_state->sdma_workspace_addr == 0) {
            // SDMA workspace not initialized
            return SdmaEvent(0);
        }
        
        __gm__ detail::pto_sdma_op_res_info_t* op_res_info = detail::pto_comm_get_sdma_op_res_info();
        if (op_res_info == nullptr) {
            return SdmaEvent(0);
        }
        
        using T = typename GlobalSrcData::RawDType;
        
        static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>,
            "SDMA::get: src/dst element type mismatch");
        static_assert(GlobalSrcData::layout == GlobalDstData::layout,
            "SDMA::get: src/dst layout mismatch");
        
        constexpr size_t elemSize = sizeof(T);
        static_assert(elemSize == 1 || elemSize == 2 || elemSize == 4 || elemSize == 8,
            "SDMA::get: Element size must be 1, 2, 4, or 8 bytes");
        
        // Calculate transfer size from tensor shape
        uint64_t totalElements = 1;
        totalElements *= static_cast<uint64_t>(srcGlobal.GetShape(GlobalTensorDim::DIM_0));
        totalElements *= static_cast<uint64_t>(srcGlobal.GetShape(GlobalTensorDim::DIM_1));
        totalElements *= static_cast<uint64_t>(srcGlobal.GetShape(GlobalTensorDim::DIM_2));
        totalElements *= static_cast<uint64_t>(srcGlobal.GetShape(GlobalTensorDim::DIM_3));
        totalElements *= static_cast<uint64_t>(srcGlobal.GetShape(GlobalTensorDim::DIM_4));
        uint64_t transferSize = totalElements * sizeof(T);
        
        if (transferSize == 0) {
            return SdmaEvent(0);
        }
        
        // Get source (remote) and destination (local) addresses
        __gm__ void* srcAddr = const_cast<__gm__ void*>(
            reinterpret_cast<__gm__ const void*>(srcGlobal.data()));
        __gm__ void* dstAddr = const_cast<__gm__ void*>(
            reinterpret_cast<__gm__ const void*>(dstGlobal.data()));
        
        // Select an SDMA channel based on destination (local) address
        uint32_t channel_idx = detail::pto_comm_select_sdma_channel(
            static_cast<uint32_t>(reinterpret_cast<uint64_t>(dstAddr) % SDMA_MAX_CHAN));
        
        // Call device-side get implementation
        detail::get<T>(reinterpret_cast<__gm__ T*>(dstAddr), 
                       reinterpret_cast<__gm__ T*>(srcAddr), 
                       transferSize);
        
        // Generate event ID
        uint64_t event_id = (static_cast<uint64_t>(channel_idx) << 32) | 
                            (reinterpret_cast<uint64_t>(dstAddr) & 0xFFFFFFFF);
        
        return SdmaEvent(event_id);
    }
    
    // Internal implementation of GET with explicit size
    template <typename GlobalDstData, typename GlobalSrcData>
    PTO_INTERNAL static SdmaEvent get_impl(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal,
                              uint64_t transfer_size)
    {
        __gm__ detail::pto_sdma_op_res_info_t* op_res_info = detail::pto_comm_get_sdma_op_res_info();
        if (op_res_info == nullptr) {
            return SdmaEvent(0);
        }
        
        using T = typename GlobalSrcData::RawDType;
        
        static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>,
            "SDMA::get: src/dst element type mismatch");
        static_assert(GlobalSrcData::layout == GlobalDstData::layout,
            "SDMA::get: src/dst layout mismatch");
        
        constexpr size_t elemSize = sizeof(T);
        static_assert(elemSize == 1 || elemSize == 2 || elemSize == 4 || elemSize == 8,
            "SDMA::get: Element size must be 1, 2, 4, or 8 bytes");
        
        if (transfer_size == 0) {
            return SdmaEvent(0);
        }
        
        __gm__ void* srcAddr = const_cast<__gm__ void*>(
            reinterpret_cast<__gm__ const void*>(srcGlobal.data()));
        __gm__ void* dstAddr = const_cast<__gm__ void*>(
            reinterpret_cast<__gm__ const void*>(dstGlobal.data()));
        
        uint32_t channel_idx = detail::pto_comm_select_sdma_channel(
            static_cast<uint32_t>(reinterpret_cast<uint64_t>(dstAddr) % SDMA_MAX_CHAN));
        
        // Call device-side get implementation
        detail::get<T>(reinterpret_cast<__gm__ T*>(dstAddr), 
                       reinterpret_cast<__gm__ T*>(srcAddr), 
                       transfer_size);
        
        uint64_t event_id = (static_cast<uint64_t>(channel_idx) << 32) | 
                            (reinterpret_cast<uint64_t>(dstAddr) & 0xFFFFFFFF);
        
        return SdmaEvent(event_id);
    }
    
    // Static flag to track initialization status
    static bool initialized_;
};

// Initialize static member
inline bool SDMA::initialized_ = false;

} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_SDMA_SDMA_HPP
