/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_SDMA_ASYNC_INTRIN_HPP
#define PTO_COMM_SDMA_ASYNC_INTRIN_HPP

#include "kernel_operator.h"
#include "pto/comm/sdma/sdma_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/pto-inst.hpp"
#include <cstdint>

namespace pto {
namespace comm {
namespace sdma {
namespace detail {

// ============================================================================
// Temporary Buffer Tile Type Definition
// ============================================================================
using TmpBufTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, 512, pto::BLayout::RowMajor, -1, -1>;

// ============================================================================
// Device Memory Address Constants (same as aclshmem)
// ============================================================================
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
// Device-side SDMA Resource Access (standalone re-implementation)
// ============================================================================
struct pto_host_stream_info_t {
    uint64_t stream_;
    int32_t dev_id;
    int32_t stream_id;
    uint32_t sq_id;
    uint32_t cq_id;
    uint32_t logic_cq_id;
    uint64_t ctx_;
    uint8_t reserved[12];
};

struct pto_sdma_op_res_info_t {
    pto_host_stream_info_t streams[40];
    uint64_t workspace_addr;
};

struct pto_comm_global_state_t {
    uint64_t sdma_workspace_addr;
    uint64_t sdma_flag_addr;
    uint64_t sdma_op_res_info_addr;
};

PTO_INTERNAL __gm__ void* pto_comm_get_extra_context_addr(uint32_t shmemId)
{
    if (shmemId >= PTO_OBJECT_NUM_MAX) {
        return nullptr;
    }
    uint64_t ctxAddr = PTO_SHM_DEVICE_USER_CONTEXT_ADDR + shmemId * PTO_SHM_DEVICE_USER_CONTEXT_PRE_SIZE;
    return reinterpret_cast<__gm__ void*>(ctxAddr);
}

PTO_INTERNAL __gm__ pto_comm_global_state_t* pto_comm_get_state()
{
    return reinterpret_cast<__gm__ pto_comm_global_state_t*>(pto_comm_get_extra_context_addr(0));
}

PTO_INTERNAL __gm__ pto_sdma_op_res_info_t* pto_comm_get_sdma_op_res_info()
{
    __gm__ pto_comm_global_state_t* state = pto_comm_get_state();
    if (state == nullptr || state->sdma_op_res_info_addr == 0) {
        return nullptr;
    }
    return reinterpret_cast<__gm__ pto_sdma_op_res_info_t*>(state->sdma_op_res_info_addr);
}

PTO_INTERNAL uint64_t pto_comm_get_sdma_workspace_addr()
{
    __gm__ pto_comm_global_state_t* state = pto_comm_get_state();
    if (state == nullptr) {
        return 0;
    }
    return state->sdma_workspace_addr;
}

PTO_INTERNAL uint64_t pto_comm_get_sdma_flag_addr()
{
    __gm__ pto_comm_global_state_t* state = pto_comm_get_state();
    if (state == nullptr) {
        return 0;
    }
    return state->sdma_flag_addr;
}

PTO_INTERNAL uint32_t pto_comm_select_sdma_channel(uint32_t block_idx, uint32_t num_channels = 40)
{
    return block_idx % num_channels;
}

// ============================================================================
// Device-side SDMA Implementation (standalone re-implementation)
// ============================================================================
PTO_INTERNAL void dcci_cacheline(__gm__ uint8_t* addr)
{
    using namespace AscendC;
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(addr);

    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

template <typename T>
PTO_INTERNAL void copy_gm_to_gm(__gm__ uint8_t *dst, __gm__ uint8_t *src, uint32_t size,
                               TmpBufTile& tmp_tile)
{
    __ubuf__ uint8_t* ub_ptr = tmp_tile.data();
    __gm__ uint8_t* gm_src_ptr = src;
    __gm__ uint8_t* gm_dst_ptr = dst;

    uint32_t copy_bytes = size * sizeof(T);

    for (uint32_t i = 0; i < copy_bytes; ++i) {
        ub_ptr[i] = gm_src_ptr[i];
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    for (uint32_t i = 0; i < copy_bytes; ++i) {
        gm_dst_ptr[i] = ub_ptr[i];
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename T>
PTO_INTERNAL void set_value(__gm__ uint8_t* addr, TmpBufTile& tmp_tile, T x)
{
    __gm__ T* gm_ptr = reinterpret_cast<__gm__ T*>(addr);
    gm_ptr[0] = x;
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename T>
PTO_INTERNAL T get_value(__gm__ uint8_t* addr, TmpBufTile& tmp_tile)
{
    dcci_cacheline(addr);
    T x = *((__gm__ T *)addr);
    return x;
}

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

PTO_INTERNAL bool init_sdma_config(__gm__ uint8_t* context_gm,
                                  uint64_t message_len,
                                  uint32_t block_idx,
                                  uint32_t comm_block_dim,
                                  sdma_config_t& config,
                                  TmpBufTile& tmp_tile)
{
    __gm__ batch_write_flag_info_t *flag_info =
        (__gm__ batch_write_flag_info_t*)context_gm;
    (void)flag_info;
    config.queue_num = 1;

    if (block_idx >= comm_block_dim ||
        block_idx >= (SDMA_MAX_CHAN / config.queue_num)) {
        return false;
    }

    uint32_t used_block_dim = AscendC::Std::min<uint32_t>(comm_block_dim, SDMA_MAX_CHAN / config.queue_num);

    config.block_bytes = 1024 * 1024;
    config.per_core_bytes = message_len / used_block_dim;

    uint64_t extra_bytes = message_len % used_block_dim;
    if (block_idx < extra_bytes) {
        config.per_core_bytes += 1;
    }

    config.iter_num = (config.per_core_bytes + config.block_bytes - 1) / config.block_bytes;
    if (config.iter_num == 0) {
        return true;
    }

    uint64_t base_per_core = message_len / used_block_dim;
    if (block_idx < extra_bytes) {
        config.comm_block_offset = block_idx * (base_per_core + 1);
    } else {
        config.comm_block_offset = extra_bytes * (base_per_core + 1) +
                                   (block_idx - extra_bytes) * base_per_core;
    }

    return true;
}

PTO_INTERNAL void prepare_workspace(__gm__ uint8_t* workspace,
                                   __gm__ uint8_t* flag_addr,
                                   const sdma_config_t& config,
                                   workspace_layout_t &layout,
                                   uint32_t block_idx,
                                   uint32_t my_pe,
                                   TmpBufTile& tmp_tile)
{
    uint64_t per_core_workspace_size = config.queue_num * SDMA_FLAG_LENGTH;

    __gm__ uint8_t* my_workspace = workspace + SDMA_FLAG_LENGTH +
                                   (block_idx * per_core_workspace_size);

    layout.send_workspace = workspace;
    layout.recv_workspace = my_workspace;

    layout.remote_recv_workspace = flag_addr +
                                   my_pe * SDMA_MAX_CHAN * SDMA_FLAG_LENGTH +
                                   block_idx * per_core_workspace_size;

    set_value<uint32_t>((__gm__ uint8_t*)layout.send_workspace, tmp_tile, config.queue_num);
}

PTO_INTERNAL void init_sq_tail_array(__gm__ batch_write_channel_info_t* batch_write_channel_info,
                                    uint32_t queue_num,
                                    uint32_t* sq_tail,
                                    TmpBufTile& tmp_tile)
{
    for (uint32_t queue_id = 0U; queue_id < queue_num; ++queue_id) {
        __gm__ batch_write_channel_info_t* channel_info =
            batch_write_channel_info + queue_id;
        sq_tail[queue_id] = get_value<uint32_t>(
            ((__gm__ uint8_t*)channel_info) + 4, tmp_tile);
    }
}

PTO_INTERNAL void submit_data_transfer_sqes(
    __gm__ batch_write_channel_info_t* batch_write_channel_info,
    __gm__ uint8_t* send_buffer,
    __gm__ uint8_t* recv_buffer,
    uint32_t opcode,
    const sdma_config_t& config,
    uint32_t* sq_tail,
    TmpBufTile& tmp_tile)
{
    for (uint32_t idx = 0U; idx < config.iter_num; ++idx) {
        uint32_t queue_idx = idx % config.queue_num;
        __gm__ batch_write_channel_info_t* channel_info =
            batch_write_channel_info + queue_idx;

        uint32_t transfer_bytes = config.block_bytes;
        if (idx == config.iter_num - 1) {
            transfer_bytes = config.per_core_bytes - idx * config.block_bytes;
        }

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

PTO_INTERNAL void submit_flag_transfer_sqes(
    __gm__ batch_write_channel_info_t* batch_write_channel_info,
    const workspace_layout_t &layout,
    const sdma_config_t& config,
    uint32_t* sq_tail,
    TmpBufTile& tmp_tile)
{
    for (uint32_t queue_id = 0U; queue_id < config.queue_num; ++queue_id) {
        __gm__ batch_write_channel_info_t* channel_info =
            batch_write_channel_info + queue_id;

        add_one_memcpy_sqe(channel_info,
                           layout.send_workspace,
                           layout.remote_recv_workspace + queue_id * SDMA_FLAG_LENGTH,
                           0, 8, sq_tail[queue_id],
                           sq_tail[queue_id] - channel_info->sq_head);

        sq_tail[queue_id] = (sq_tail[queue_id] + 1) % SQ_DEPTH;
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

PTO_INTERNAL void flush_cache_and_ring_doorbell(
    __gm__ batch_write_channel_info_t* batch_write_channel_info,
    const sdma_config_t& config,
    uint32_t* sq_tail,
    TmpBufTile& tmp_tile)
{
    auto item_size = config.iter_num * sizeof(batch_write_item_t);
    for (uint8_t queue_id = 0; queue_id < config.queue_num; queue_id++) {
        __gm__ batch_write_channel_info_t* channel_info =
            batch_write_channel_info + queue_id;

        AscendC::GlobalTensor<uint8_t> write_info;
        write_info.SetGlobalBuffer((__gm__ uint8_t *)(channel_info->sq_base), item_size);
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_OUT>(write_info);

        set_value<uint32_t>((__gm__ uint8_t*)(channel_info->sq_reg_base) + 8,
                           tmp_tile, sq_tail[queue_id]);
    }
}

PTO_INTERNAL bool poll_for_completion(
    __gm__ batch_write_channel_info_t* batch_write_channel_info,
    const workspace_layout_t &layout,
    const sdma_config_t& config,
    uint32_t* sq_tail,
    TmpBufTile& tmp_tile)
{
    const uint32_t max_times = 1000000;
    for (uint8_t queue_id = 0; queue_id < config.queue_num; queue_id++) {
        __gm__ batch_write_channel_info_t* channel_info =
            batch_write_channel_info + queue_id;

        auto local_recv_workspace = layout.recv_workspace + queue_id * SDMA_FLAG_LENGTH;
        auto remote_recv_workspace = layout.remote_recv_workspace + queue_id * SDMA_FLAG_LENGTH;

        uint32_t send_value = 0;
        uint32_t times = 0;

        while (send_value == 0 && times < max_times) {
            copy_gm_to_gm<uint32_t>(local_recv_workspace, remote_recv_workspace, 1, tmp_tile);
            send_value = get_value<uint32_t>(local_recv_workspace, tmp_tile);
            times++;
        }

        set_value<uint32_t>(remote_recv_workspace, tmp_tile, 0);
        set_value<uint32_t>(local_recv_workspace, tmp_tile, 0);

        set_value<uint32_t>(((__gm__ uint8_t*)channel_info) + 4, tmp_tile, sq_tail[queue_id]);
    }

    return true;
}

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

    TmpBufTile tmp_tile(1, 512);
    TASSIGN(tmp_tile, 0x0);

    const auto block_idx = AscendC::GetBlockIdx();
    const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

    sdma_config_t config;
    if (!init_sdma_config(context_gm, message_len, block_idx, comm_block_dim,
                          config, tmp_tile)) {
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }
    if (config.iter_num == 0) {
        return;
    }

    __gm__ batch_write_channel_info_t* batch_write_channel_base =
        (__gm__ batch_write_channel_info_t *)(context_gm +
                                              sizeof(batch_write_flag_info_t));
    __gm__ batch_write_channel_info_t* batch_write_channel_info =
        batch_write_channel_base + block_idx * config.queue_num;

    __gm__ uint8_t* workspace = context_gm +
                                sizeof(batch_write_flag_info_t) +
                                SDMA_MAX_CHAN * sizeof(batch_write_channel_info_t);

    uint32_t my_pe = block_idx;
    workspace_layout_t workspace_layout;
    prepare_workspace(workspace, flag_addr, config, workspace_layout,
                     block_idx, my_pe, tmp_tile);

    uint32_t sq_tail[64] = {0};
    init_sq_tail_array(batch_write_channel_info, config.queue_num, sq_tail, tmp_tile);

    submit_data_transfer_sqes(batch_write_channel_info, send_buffer, recv_buffer,
                              static_cast<uint32_t>(opcode), config, sq_tail, tmp_tile);

    submit_flag_transfer_sqes(batch_write_channel_info, workspace_layout, config,
                              sq_tail, tmp_tile);

    flush_cache_and_ring_doorbell(batch_write_channel_info, config, sq_tail, tmp_tile);

    (void)poll_for_completion(batch_write_channel_info, workspace_layout, config,
                              sq_tail, tmp_tile);

    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename T>
PTO_INTERNAL void sdma_write(__gm__ T* dst, __gm__ T* src, uint64_t messageLen)
{
    sdma_post_send((__gm__ uint8_t*)dst, (__gm__ uint8_t*)src, 0, messageLen);
}

} // namespace detail

// ============================================================================
// Async SDMA intrinsics (standalone re-implementation)
// ============================================================================
template <typename T>
PTO_INTERNAL uint64_t __sdma_put_async(__gm__ T* dst, __gm__ T* src, uint64_t transfer_size)
{
    if (transfer_size == 0) {
        return 0;
    }
    detail::sdma_write(dst, src, transfer_size);
    uint32_t channel_idx = detail::pto_comm_select_sdma_channel(
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(src) % SDMA_MAX_CHAN));
    return (static_cast<uint64_t>(channel_idx) << 32) |
           (reinterpret_cast<uint64_t>(src) & 0xFFFFFFFF);
}

template <typename T>
PTO_INTERNAL uint64_t __sdma_get_async(__gm__ T* dst, __gm__ T* src, uint64_t transfer_size)
{
    if (transfer_size == 0) {
        return 0;
    }
    detail::sdma_write(dst, src, transfer_size);
    uint32_t channel_idx = detail::pto_comm_select_sdma_channel(
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(dst) % SDMA_MAX_CHAN));
    return (static_cast<uint64_t>(channel_idx) << 32) |
           (reinterpret_cast<uint64_t>(dst) & 0xFFFFFFFF);
}

} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_SDMA_ASYNC_INTRIN_HPP
