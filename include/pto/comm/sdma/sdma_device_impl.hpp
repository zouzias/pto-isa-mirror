/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_SDMA_DEVICE_IMPL_HPP
#define PTO_COMM_SDMA_DEVICE_IMPL_HPP

#include "kernel_operator.h"
#include "pto/comm/sdma/sdma_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/comm/comm_types.hpp"
#include <cstdint>

namespace pto {
namespace comm {
namespace sdma {
namespace detail {

// ============================================================================
// Device Memory Address Constants (same as aclshmem)
// ============================================================================

// Device memory layout constants for accessing shared state
constexpr uint64_t PTO_SHM_DEVICE_END_ADDR = 0x180000000000ULL - (1UL << 30UL);
constexpr uint64_t PTO_SHM_DEVICE_PRE_META_SIZE = 128UL;  // 128B per entity
constexpr uint64_t PTO_SHM_DEVICE_GLOBAL_META_SIZE = PTO_SHM_DEVICE_PRE_META_SIZE;  // 128B
constexpr uint64_t PTO_OBJECT_NUM_MAX = 511UL;  // Maximum entity count

constexpr uint64_t PTO_SHM_DEVICE_USER_CONTEXT_PRE_SIZE = 64UL * 1024UL;  // 64K per context
constexpr uint64_t PTO_SHM_DEVICE_META_SIZE = PTO_SHM_DEVICE_PRE_META_SIZE * PTO_OBJECT_NUM_MAX
                                             + PTO_SHM_DEVICE_GLOBAL_META_SIZE;  // 64K total metadata

constexpr uint64_t PTO_SHM_DEVICE_INFO_SIZE = PTO_SHM_DEVICE_USER_CONTEXT_PRE_SIZE * PTO_OBJECT_NUM_MAX
                                             + PTO_SHM_DEVICE_META_SIZE;  // ~32M total

constexpr uint64_t PTO_SHM_DEVICE_META_ADDR = PTO_SHM_DEVICE_END_ADDR - PTO_SHM_DEVICE_INFO_SIZE;
constexpr uint64_t PTO_SHM_DEVICE_USER_CONTEXT_ADDR = PTO_SHM_DEVICE_META_ADDR + PTO_SHM_DEVICE_META_SIZE;

// ============================================================================
// Device-side SDMA Resource Access
// ============================================================================

// Host stream info structure (matches host side definition)
struct pto_host_stream_info_t {
    uint64_t stream_;      // Stream handle
    int32_t dev_id;        // Device ID
    int32_t stream_id;     // Stream ID
    uint32_t sq_id;        // Send Queue ID
    uint32_t cq_id;        // Completion Queue ID
    uint32_t logic_cq_id;  // Logical Completion Queue ID
    uint64_t ctx_;         // Context handle
    uint8_t reserved[12];  // Padding to 48 bytes
};

// SDMA operation resource info structure
struct pto_sdma_op_res_info_t {
    pto_host_stream_info_t streams[40];  // Maximum 40 SDMA channels
    uint64_t workspace_addr;             // Workspace address for SDMA operations
};

// Global state structure (simplified - should match actual PTO global state)
struct pto_comm_global_state_t {
    uint64_t sdma_workspace_addr;    // SDMA workspace address
    uint64_t sdma_flag_addr;         // SDMA flag address for synchronization
    uint64_t sdma_op_res_info_addr;  // Address of SDMA operation resource info
    // ... other state members
};

// Helper function to get extra context address (same as aclshmemi_get_extra_context_addr)
PTO_INTERNAL __gm__ void* pto_comm_get_extra_context_addr(uint32_t shmemId)
{
    if (shmemId >= PTO_OBJECT_NUM_MAX) {
        return nullptr;
    }
    uint64_t ctxAddr = PTO_SHM_DEVICE_USER_CONTEXT_ADDR + shmemId * PTO_SHM_DEVICE_USER_CONTEXT_PRE_SIZE;
    return reinterpret_cast<__gm__ void*>(ctxAddr);
}

// Function to get global state (same implementation as aclshmemi_get_state)
PTO_INTERNAL __gm__ pto_comm_global_state_t* pto_comm_get_state()
{
    return reinterpret_cast<__gm__ pto_comm_global_state_t*>(pto_comm_get_extra_context_addr(0));
}

// Get SDMA operation resource info pointer
PTO_INTERNAL __gm__ pto_sdma_op_res_info_t* pto_comm_get_sdma_op_res_info()
{
    __gm__ pto_comm_global_state_t* state = pto_comm_get_state();
    if (state == nullptr || state->sdma_op_res_info_addr == 0) {
        return nullptr;
    }
    return reinterpret_cast<__gm__ pto_sdma_op_res_info_t*>(state->sdma_op_res_info_addr);
}

// Get SDMA workspace address
PTO_INTERNAL uint64_t pto_comm_get_sdma_workspace_addr()
{
    __gm__ pto_comm_global_state_t* state = pto_comm_get_state();
    if (state == nullptr) {
        return 0;
    }
    return state->sdma_workspace_addr;
}

// Get SDMA flag address
PTO_INTERNAL uint64_t pto_comm_get_sdma_flag_addr()
{
    __gm__ pto_comm_global_state_t* state = pto_comm_get_state();
    if (state == nullptr) {
        return 0;
    }
    return state->sdma_flag_addr;
}

// Select an available SDMA channel based on block index
PTO_INTERNAL uint32_t pto_comm_select_sdma_channel(uint32_t block_idx, uint32_t num_channels = 40)
{
    // Simple round-robin selection
    return block_idx % num_channels;
}

// ============================================================================
// Device-side SDMA Implementation
// ============================================================================

// Helper: Invalidate single cache line
PTO_INTERNAL void dcci_cacheline(__gm__ uint8_t* addr)
{
    using namespace AscendC;
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(addr);

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

// Helper: Copy from GM to GM using AscendC API
template <typename T>
PTO_INTERNAL void copy_gm_to_gm(__gm__ uint8_t *dst, __gm__ uint8_t *src, uint32_t size,
                                 AscendC::TBuf<AscendC::TPosition::VECOUT> &tmp_buf)
{
    AscendC::GlobalTensor<T> gm_src;
    AscendC::GlobalTensor<T> gm_dst;
    gm_src.SetGlobalBuffer((__gm__ T *)src, size);
    gm_dst.SetGlobalBuffer((__gm__ T *)dst, size);
    AscendC::LocalTensor<T> x_local = tmp_buf.template Get<T>();

    uint32_t cp_len = size * sizeof(T);
    AscendC::DataCopyExtParams cp_params{1, cp_len, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad_params{false, 0, 0, 0};
    AscendC::DataCopyPad(x_local, gm_src, cp_params, pad_params);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::DataCopyPad(gm_dst, x_local, cp_params);
    AscendC::PipeBarrier<PIPE_ALL>();
}

// Helper: Set a single value in GM memory using AscendC API
template <typename T>
PTO_INTERNAL void set_value(__gm__ uint8_t* addr, AscendC::TBuf<AscendC::TPosition::VECOUT> &tmp_buf, T x)
{
    AscendC::GlobalTensor<T> gm_dst;
    gm_dst.SetGlobalBuffer((__gm__ T *)addr);
    AscendC::LocalTensor<T> x_local = tmp_buf.template Get<T>();
    x_local.SetValue(0, x);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams cp_out_params{1, sizeof(T), 0, 0, 0};
    AscendC::DataCopyPad(gm_dst, x_local, cp_out_params);
    AscendC::PipeBarrier<PIPE_ALL>();
}

// Helper: Get a single value from GM memory using AscendC API
template <typename T>
PTO_INTERNAL T get_value(__gm__ uint8_t* addr, AscendC::TBuf<AscendC::TPosition::VECOUT> &tmp_buf)
{
    dcci_cacheline(addr);
    T x = *((__gm__ T *)addr);
    return x;
}

// Build STARS SDMA SQE
PTO_INTERNAL void add_one_memcpy_sqe(__gm__ batch_write_channel_info_t* channel_info,
                                     __gm__ uint8_t* src,
                                     __gm__ uint8_t* dst,
                                     uint64_t opcode,
                                     uint32_t length,
                                     uint32_t sq_tail,
                                     uint32_t task_id)
{
    __gm__ batch_write_item_t *sqe = 
        (__gm__ batch_write_item_t *)(channel_info->sq_base);
    sqe += (sq_tail % channel_info->sq_depth);

    sqe->type = RT_STARS_SQE_TYPE_SDMA;
    sqe->blockDim = 0;
    sqe->rtStreamId = channel_info->stream_id;
    sqe->taskId = task_id;
    sqe->kernel_credit = K_CREDIT_TIME_DEFAULT;
    sqe->ptr_mode = 0;
    sqe->opcode = static_cast<uint32_t>(opcode);
    sqe->ie2 = 0;
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns = 1U;
    sqe->dns = 1U;
    sqe->qos = 6;
    sqe->partid = 0U;
    sqe->mpam = 0;
    sqe->length = length;

    uint64_t src_addr = reinterpret_cast<uint64_t>(src);
    uint64_t dst_addr = reinterpret_cast<uint64_t>(dst);

    sqe->srcAddrLow = static_cast<uint32_t>(src_addr & 0xFFFFFFFF);
    sqe->srcAddrHigh = static_cast<uint32_t>((src_addr >> 32) & 0xFFFFFFFF);
    sqe->dstAddrLow = static_cast<uint32_t>(dst_addr & 0xFFFFFFFF);
    sqe->dstAddrHigh = static_cast<uint32_t>((dst_addr >> 32) & 0xFFFFFFFF);
    sqe->linkType = static_cast<uint8_t>(255U);

    AscendC::PipeBarrier<PIPE_ALL>();
}

// Initialize SDMA Configuration
PTO_INTERNAL bool init_sdma_config(__gm__ uint8_t* context_gm,
                                   uint64_t message_len,
                                   uint32_t block_idx,
                                   uint32_t comm_block_dim,
                                   sdma_config_t& config,
                                   AscendC::TBuf<AscendC::TPosition::VECOUT>& tmp_buf)
{
    // Get queue info
    __gm__ batch_write_flag_info_t *flag_info = 
        (__gm__ batch_write_flag_info_t*)context_gm;
    config.queue_num = 1; // Number of queues per core, currently set to 1

    // Check if block_idx is valid, block * queue_num must be < SDMA_MAX_CHAN
    if (block_idx >= comm_block_dim || 
        block_idx >= (SDMA_MAX_CHAN / config.queue_num)) {
        return false;
    }

    uint32_t used_block_dim = AscendC::Std::min<uint32_t>(comm_block_dim, SDMA_MAX_CHAN / config.queue_num);

    // Calculate block parameters
    config.block_bytes = 1024 * 1024; // 1MB per SQE
    config.per_core_bytes = message_len / used_block_dim; // Data to transfer per core

    // Handle remainder
    uint64_t extra_bytes = message_len % used_block_dim;
    if (block_idx < extra_bytes) {
        config.per_core_bytes += 1; // Earlier cores transfer 1 more byte than later cores
    }

    config.iter_num = (config.per_core_bytes + config.block_bytes - 1) / config.block_bytes; // Number of SQEs needed
    if (config.iter_num == 0) {
        return true;
    }

    // Calculate data offset for current core
    uint64_t base_per_core = message_len / used_block_dim;
    if (block_idx < extra_bytes) {
        // First extra cores each have 1 extra byte
        config.comm_block_offset = block_idx * (base_per_core + 1);
    } else {
        // Later cores
        config.comm_block_offset = extra_bytes * (base_per_core + 1) + 
                                   (block_idx - extra_bytes) * base_per_core;
    }

    return true;
}

// Prepare Workspace
PTO_INTERNAL void prepare_workspace(__gm__ uint8_t* workspace,
                                    __gm__ uint8_t* flag_addr,
                                    const sdma_config_t& config,
                                    workspace_layout_t &layout,
                                    uint32_t block_idx,
                                    uint32_t my_pe,
                                    AscendC::TBuf<AscendC::TPosition::VECOUT>& tmp_buf)
{
    // Per-core workspace size for flag data: flag_length + flag receive area flag_length*queue_num
    uint64_t per_core_workspace_size = config.queue_num * SDMA_FLAG_LENGTH;

    // Current core's workspace starting position: placed after the channel
    __gm__ uint8_t* my_workspace = workspace + SDMA_FLAG_LENGTH + 
                                   (block_idx * per_core_workspace_size);

    // Current core's send flag is at the beginning of workspace
    layout.send_workspace = workspace;
    // Current core's recv flags are after the send flag
    layout.recv_workspace = my_workspace;

    // Remote recv workspace calculation
    layout.remote_recv_workspace = flag_addr + 
                                   my_pe * SDMA_MAX_CHAN * SDMA_FLAG_LENGTH + 
                                   block_idx * per_core_workspace_size;

    // Initialize send flag
    set_value<uint32_t>((__gm__ uint8_t*)layout.send_workspace, tmp_buf, config.queue_num);
}

// Initialize SQ Tail Array
PTO_INTERNAL void init_sq_tail_array(__gm__ batch_write_channel_info_t* batch_write_channel_info,
                                     uint32_t queue_num,
                                     uint32_t* sq_tail,
                                     AscendC::TBuf<AscendC::TPosition::VECOUT>& tmp_buf)
{
    for (uint32_t queue_id = 0U; queue_id < queue_num; ++queue_id) {
        __gm__ batch_write_channel_info_t* channel_info = 
            batch_write_channel_info + queue_id;
        // Get sq_tail field (offset 4 bytes)
        sq_tail[queue_id] = get_value<uint32_t>(
            ((__gm__ uint8_t*)channel_info) + 4, tmp_buf);
    }
}

// Submit Data Transfer SQEs
PTO_INTERNAL void submit_data_transfer_sqes(
    __gm__ batch_write_channel_info_t* batch_write_channel_info,
    __gm__ uint8_t* send_buffer,
    __gm__ uint8_t* recv_buffer,
    uint32_t opcode,
    const sdma_config_t& config,
    uint32_t* sq_tail,
    AscendC::TBuf<AscendC::TPosition::VECOUT>& tmp_buf)
{
    for (uint32_t idx = 0U; idx < config.iter_num; ++idx) {
        uint32_t queue_idx = idx % config.queue_num;
        __gm__ batch_write_channel_info_t* channel_info = 
            batch_write_channel_info + queue_idx;

        // Calculate transfer size for current block
        uint32_t transfer_bytes = config.block_bytes;
        if (idx == config.iter_num - 1) {
            // Last block may be partial
            transfer_bytes = config.per_core_bytes - idx * config.block_bytes;
        }

        // Calculate addresses
        __gm__ uint8_t* src_addr = send_buffer + config.comm_block_offset + 
                                   idx * config.block_bytes;
        __gm__ uint8_t* dst_addr = recv_buffer + config.comm_block_offset + 
                                   idx * config.block_bytes;

        add_one_memcpy_sqe(channel_info, src_addr, dst_addr,
                           0, transfer_bytes, sq_tail[queue_idx], 
                           sq_tail[queue_idx] - channel_info->sq_head);

        sq_tail[queue_idx] = (sq_tail[queue_idx] + 1) % SQ_DEPTH;
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// Submit Flag Transfer SQEs
PTO_INTERNAL void submit_flag_transfer_sqes(
    __gm__ batch_write_channel_info_t* batch_write_channel_info,
    const workspace_layout_t &layout,
    const sdma_config_t& config,
    uint32_t* sq_tail,
    AscendC::TBuf<AscendC::TPosition::VECOUT>& tmp_buf)
{
    for (uint32_t queue_id = 0U; queue_id < config.queue_num; ++queue_id) {
        __gm__ batch_write_channel_info_t* channel_info = 
            batch_write_channel_info + queue_id;

        add_one_memcpy_sqe(channel_info,
                           layout.send_workspace,      // Source: current core's send flag
                           layout.remote_recv_workspace + queue_id * SDMA_FLAG_LENGTH, // Dest: remote window's position for current queue
                           0, 8, sq_tail[queue_id], 
                           sq_tail[queue_id] - channel_info->sq_head);

        sq_tail[queue_id] = (sq_tail[queue_id] + 1) % SQ_DEPTH;
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// Flush Cache and Ring Doorbell
PTO_INTERNAL void flush_cache_and_ring_doorbell(
    __gm__ batch_write_channel_info_t* batch_write_channel_info,
    const sdma_config_t& config,
    uint32_t* sq_tail,
    AscendC::TBuf<AscendC::TPosition::VECOUT>& tmp_buf)
{
    auto item_size = config.iter_num * sizeof(batch_write_item_t);
    for (uint8_t queue_id = 0; queue_id < config.queue_num; queue_id++) {
        __gm__ batch_write_channel_info_t* channel_info = 
            batch_write_channel_info + queue_id;

        // Flush entire data cache to ensure all SQEs are written to HBM
        AscendC::GlobalTensor<uint8_t> write_info;
        write_info.SetGlobalBuffer((__gm__ uint8_t *)(channel_info->sq_base), item_size);
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_OUT>(write_info);

        // Ring doorbell for each channel
        set_value<uint32_t>((__gm__ uint8_t*)(channel_info->sq_reg_base) + 8, 
                           tmp_buf, sq_tail[queue_id]); // 8: position of third uint32
    }
}

// Poll for Completion
PTO_INTERNAL bool poll_for_completion(
    __gm__ batch_write_channel_info_t* batch_write_channel_info,
    const workspace_layout_t &layout,
    const sdma_config_t& config,
    uint32_t* sq_tail,
    AscendC::TBuf<AscendC::TPosition::VECOUT>& tmp_buf)
{
    const uint32_t max_times = 1000000;
    for (uint8_t queue_id = 0; queue_id < config.queue_num; queue_id++) {
        __gm__ batch_write_channel_info_t* channel_info = 
            batch_write_channel_info + queue_id;

        auto local_recv_workspace = layout.recv_workspace + queue_id * SDMA_FLAG_LENGTH;
        auto remote_recv_workspace = layout.remote_recv_workspace + queue_id * SDMA_FLAG_LENGTH;

        uint32_t send_value = 0;
        uint32_t times = 0;

        // Poll until flag is received or timeout
        while (send_value == 0 && times < max_times) {
            copy_gm_to_gm<uint32_t>(local_recv_workspace, remote_recv_workspace, 1, tmp_buf);
            send_value = get_value<uint32_t>(local_recv_workspace, tmp_buf);
            times++;
        }

        // Clean up status area data
        set_value<uint32_t>(remote_recv_workspace, tmp_buf, 0);
        set_value<uint32_t>(local_recv_workspace, tmp_buf, 0);

        // Update tail value in channel info
        set_value<uint32_t>(((__gm__ uint8_t*)channel_info) + 4, tmp_buf, sq_tail[queue_id]);
    }

    return true;
}

// Debug status codes for sdma_post_send
enum class SdmaDebugStatus : uint32_t {
    NOT_STARTED = 0,
    DEVICE_STATE_NULL = 1,
    CONTEXT_GM_NULL = 2,
    FLAG_ADDR_NULL = 3,
    UB_INIT_DONE = 10,
    CONFIG_INIT_FAILED = 11,
    CONFIG_ITER_ZERO = 12,
    CONFIG_INIT_DONE = 20,
    WORKSPACE_PREPARED = 30,
    SQ_TAIL_INIT_DONE = 40,
    DATA_SQES_SUBMITTED = 50,
    FLAG_SQES_SUBMITTED = 60,
    DOORBELL_RUNG = 70,
    POLL_STARTED = 80,
    POLL_COMPLETED = 90,
    ALL_DONE = 100
};

// Helper: Write debug status to a debug buffer in GM
// debug_buffer should be pre-allocated in the test kernel
PTO_INTERNAL void write_debug_status(__gm__ uint32_t* debug_buffer, 
                                     uint32_t block_idx,
                                     SdmaDebugStatus status,
                                     AscendC::TBuf<AscendC::TPosition::VECOUT>& tmp_buf)
{
    if (debug_buffer != nullptr) {
        set_value<uint32_t>((__gm__ uint8_t*)(debug_buffer + block_idx), 
                           tmp_buf, static_cast<uint32_t>(status));
    }
}

// Main SDMA Post Send Function
// debug_buffer: optional debug status buffer (pass nullptr to disable debug)
PTO_INTERNAL void sdma_post_send_debug(__gm__ uint8_t* recv_buffer,
                                        __gm__ uint8_t* send_buffer,
                                        uint64_t opcode,
                                        uint64_t message_len,
                                        __gm__ uint32_t* debug_buffer)
{
    const auto block_idx = AscendC::GetBlockIdx();
    
    // Initialize temporary UB buffer early for debug writes
    AscendC::TBuf<AscendC::TPosition::VECOUT> tmp_buf;
    GetTPipePtr()->InitBuffer(tmp_buf, UB_ALIGN_SIZE * 2);
    
    __gm__ detail::pto_comm_global_state_t* device_state = detail::pto_comm_get_state();
    if (device_state == nullptr) {
        write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::DEVICE_STATE_NULL, tmp_buf);
        return;
    }

    __gm__ uint8_t* context_gm = reinterpret_cast<__gm__ uint8_t*>(
        device_state->sdma_workspace_addr);
    if (context_gm == nullptr) {
        write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::CONTEXT_GM_NULL, tmp_buf);
        return;
    }

    __gm__ uint8_t* flag_addr = reinterpret_cast<__gm__ uint8_t*>(
        device_state->sdma_flag_addr);
    if (flag_addr == nullptr) {
        write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::FLAG_ADDR_NULL, tmp_buf);
        return;
    }

    write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::UB_INIT_DONE, tmp_buf);

    // 2. Get current core info
    const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

    // 3. Initialize configuration parameters
    sdma_config_t config;
    if (!init_sdma_config(context_gm, message_len, block_idx, comm_block_dim, 
                          config, tmp_buf)) {
        write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::CONFIG_INIT_FAILED, tmp_buf);
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }
    if (config.iter_num == 0) {
        write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::CONFIG_ITER_ZERO, tmp_buf);
        return; // No transfer task, exit directly
    }
    
    write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::CONFIG_INIT_DONE, tmp_buf);

    // 4. Get channel info
    __gm__ batch_write_channel_info_t* batch_write_channel_base =
        (__gm__ batch_write_channel_info_t *)(context_gm + 
                                               sizeof(batch_write_flag_info_t));
    // Channel info for current block
    __gm__ batch_write_channel_info_t* batch_write_channel_info = 
        batch_write_channel_base + block_idx * config.queue_num;

    // 5.1 Calculate workspace
    __gm__ uint8_t* workspace = context_gm + 
                                 sizeof(batch_write_flag_info_t) + 
                                 SDMA_MAX_CHAN * sizeof(batch_write_channel_info_t);
    
    // 5.2 Prepare workspace
    // TODO: Get my_pe from device state (currently using block_idx as placeholder)
    uint32_t my_pe = block_idx;
    workspace_layout_t workspace_layout;
    prepare_workspace(workspace, flag_addr, config, workspace_layout, 
                      block_idx, my_pe, tmp_buf);
    
    write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::WORKSPACE_PREPARED, tmp_buf);

    // 6. Initialize sq_tail array
    uint32_t sq_tail[64] = {0};  // Assume max 64 queues
    init_sq_tail_array(batch_write_channel_info, config.queue_num, sq_tail, tmp_buf);
    
    write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::SQ_TAIL_INIT_DONE, tmp_buf);

    // 8. Submit data transfer SQEs
    submit_data_transfer_sqes(batch_write_channel_info, send_buffer, recv_buffer,
                              static_cast<uint32_t>(opcode), config, sq_tail, tmp_buf);
    
    write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::DATA_SQES_SUBMITTED, tmp_buf);

    // 9. Submit flag transfer SQEs
    submit_flag_transfer_sqes(batch_write_channel_info, workspace_layout, config, 
                               sq_tail, tmp_buf);
    
    write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::FLAG_SQES_SUBMITTED, tmp_buf);

    // 10. Flush cache and ring doorbell
    flush_cache_and_ring_doorbell(batch_write_channel_info, config, sq_tail, tmp_buf);
    
    write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::DOORBELL_RUNG, tmp_buf);

    // 11. Poll for completion
    write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::POLL_STARTED, tmp_buf);
    
    if (!poll_for_completion(batch_write_channel_info, workspace_layout, config, 
                             sq_tail, tmp_buf)) {
        // Transfer failed - status already indicates POLL_STARTED
    }
    
    write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::POLL_COMPLETED, tmp_buf);

    AscendC::PipeBarrier<PIPE_ALL>();
    
    write_debug_status(debug_buffer, block_idx, SdmaDebugStatus::ALL_DONE, tmp_buf);
}

// Main SDMA Post Send Function
PTO_INTERNAL void sdma_post_send(__gm__ uint8_t* recv_buffer,
                                  __gm__ uint8_t* send_buffer,
                                  uint64_t opcode,
                                  uint64_t message_len)
{
    __gm__ detail::pto_comm_global_state_t* device_state = detail::pto_comm_get_state();
    if (device_state == nullptr) {
        return;
    }

    __gm__ uint8_t* context_gm = reinterpret_cast<__gm__ uint8_t*>(
        device_state->sdma_workspace_addr);
    if (context_gm == nullptr) {
        return;
    }

    __gm__ uint8_t* flag_addr = reinterpret_cast<__gm__ uint8_t*>(
        device_state->sdma_flag_addr);
    if (flag_addr == nullptr) {
        return;
    }

    // 1. Initialize UB buffer
    AscendC::TBuf<AscendC::TPosition::VECOUT> tmp_buf;
    GetTPipePtr()->InitBuffer(tmp_buf, UB_ALIGN_SIZE * 2);

    // 2. Get current core info
    const auto block_idx = AscendC::GetBlockIdx();
    const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

    // 3. Initialize configuration parameters
    sdma_config_t config;
    if (!init_sdma_config(context_gm, message_len, block_idx, comm_block_dim, 
                          config, tmp_buf)) {
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }
    if (config.iter_num == 0) {
        return; // No transfer task, exit directly
    }

    // 4. Get channel info
    __gm__ batch_write_channel_info_t* batch_write_channel_base =
        (__gm__ batch_write_channel_info_t *)(context_gm + 
                                               sizeof(batch_write_flag_info_t));
    // Channel info for current block
    __gm__ batch_write_channel_info_t* batch_write_channel_info = 
        batch_write_channel_base + block_idx * config.queue_num;

    // 5.1 Calculate workspace
    __gm__ uint8_t* workspace = context_gm + 
                                 sizeof(batch_write_flag_info_t) + 
                                 SDMA_MAX_CHAN * sizeof(batch_write_channel_info_t);
    
    // 5.2 Prepare workspace
    // TODO: Get my_pe from device state (currently using block_idx as placeholder)
    uint32_t my_pe = block_idx;
    workspace_layout_t workspace_layout;
    prepare_workspace(workspace, flag_addr, config, workspace_layout, 
                      block_idx, my_pe, tmp_buf);

    // 6. Initialize sq_tail array
    uint32_t sq_tail[64] = {0};  // Assume max 64 queues
    init_sq_tail_array(batch_write_channel_info, config.queue_num, sq_tail, tmp_buf);

    // 8. Submit data transfer SQEs
    submit_data_transfer_sqes(batch_write_channel_info, send_buffer, recv_buffer,
                              static_cast<uint32_t>(opcode), config, sq_tail, tmp_buf);

    // 9. Submit flag transfer SQEs
    submit_flag_transfer_sqes(batch_write_channel_info, workspace_layout, config, 
                               sq_tail, tmp_buf);

    // 10. Flush cache and ring doorbell
    flush_cache_and_ring_doorbell(batch_write_channel_info, config, sq_tail, tmp_buf);

    // 11. Poll for completion
    if (!poll_for_completion(batch_write_channel_info, workspace_layout, config, 
                             sq_tail, tmp_buf)) {
        // Transfer failed
    }

    AscendC::PipeBarrier<PIPE_ALL>();
}

// SDMA Write Function
template <typename T>
PTO_INTERNAL void sdma_write(__gm__ T* dst, __gm__ T* src, uint64_t messageLen)
{
    sdma_post_send((__gm__ uint8_t*)dst, (__gm__ uint8_t*)src, 0, messageLen);
}

// Get Remote PE Address
PTO_INTERNAL __gm__ void* sdma_ptr(__gm__ void *ptr, int pe)
{
    __gm__ detail::pto_comm_global_state_t* device_state = detail::pto_comm_get_state();
    if (device_state == nullptr) {
        return nullptr;
    }
    // TODO: Implement address translation
    return ptr;
}

// PUT: Device-side put implementation
// Data flow: src (local) -> dst (remote)
template <typename T>
PTO_INTERNAL void put(__gm__ T* dst, __gm__ T* src, uint64_t transfer_size)
{
    sdma_write((__gm__ uint8_t*)dst, (__gm__ uint8_t*)src, transfer_size);
}

// GET: Device-side get implementation
// Data flow: src (remote) -> dst (local)
// Note: GET is essentially a read from remote memory to local memory.
// The SDMA hardware performs the same memcpy operation, just with reversed semantics.
template <typename T>
PTO_INTERNAL void get(__gm__ T* dst, __gm__ T* src, uint64_t transfer_size)
{
    // For GET, we read from remote (src) to local (dst)
    // The underlying SDMA operation is the same as PUT, just with different semantics
    sdma_write((__gm__ uint8_t*)dst, (__gm__ uint8_t*)src, transfer_size);
}

} // namespace detail
} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_SDMA_DEVICE_IMPL_HPP
