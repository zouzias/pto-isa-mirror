/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_SDMA_TYPES_HPP
#define PTO_COMM_SDMA_TYPES_HPP

#include <cstdint>

namespace pto {
namespace comm {
namespace sdma {

// SDMA SQE type constants
constexpr uint64_t RT_STARS_SQE_TYPE_SDMA = 11ULL;
constexpr uint64_t K_CREDIT_TIME_DEFAULT = 240ULL;
constexpr uint32_t SQ_DEPTH = 2048U;
constexpr uint32_t SDMA_FLAG_LENGTH = 128U;
constexpr uint32_t UB_ALIGN_SIZE = 256U;
constexpr uint32_t SDMA_MAX_CHAN = 40U;
constexpr uint32_t SDMA_EVENT_RECORD_BYTES = 16U;
constexpr uint32_t SDMA_EVENT_SLOT_COUNT =
    SDMA_FLAG_LENGTH / SDMA_EVENT_RECORD_BYTES;

// ============================================================================
// SDMA Configuration Structure
// ============================================================================
struct sdma_config_t {
    uint64_t block_bytes;        // Block size per SQE (typically 1MB)
    uint64_t per_core_bytes;     // Total bytes to transfer per core
    uint64_t comm_block_offset;  // Offset for current core's data
    uint32_t queue_num;          // Number of queues per core
    uint32_t iter_num;           // Number of iterations (SQEs) needed
};

// ============================================================================
// Workspace Layout Structure
// ============================================================================
struct workspace_layout_t {
    __gm__ uint8_t* send_workspace;        // Local send flag workspace
    __gm__ uint8_t* recv_workspace;        // Local receive flag workspace
};

// ============================================================================
// Batch Write Flag Info Structure
// ============================================================================
struct batch_write_flag_info_t {
    uint32_t flag;
    uint32_t totalQueueNum;
    // uint8_t reserved[56]; // Padding to 64 bytes
};

// ============================================================================
// Batch Write Channel Info Structure
// ============================================================================
struct batch_write_channel_info_t {
    uint32_t sq_head;        // Send Queue head
    uint32_t sq_tail;        // Send Queue tail
    uint64_t sq_base;        // SQ buffer base address
    uint64_t sq_reg_base;    // SQ register base address
    uint32_t sq_depth;       // SQ depth
    uint32_t sq_id;          // SQ ID
    uint32_t cq_id;          // CQ ID
    uint32_t stream_id;      // Stream ID
    // uint8_t reserved[24]; // Padding to 64 bytes
};

// ============================================================================
// Batch Write Item (SQE) Structure
// ============================================================================
struct batch_write_item_t {
    uint8_t type : 6;
    uint16_t res1 : 10;
    uint16_t blockDim;
    uint16_t rtStreamId;
    uint16_t taskId;

    uint32_t res3;

    uint16_t res4;
    uint8_t kernel_credit;
    uint8_t ptr_mode : 1;
    uint8_t res5 : 7;

    uint32_t opcode : 8;
    uint32_t ie2 : 1;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t qos : 4;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t partid : 8;
    uint32_t mpam : 1;
    uint32_t res6 : 4;

    uint16_t src_streamid;
    uint16_t src_sub_streamid;

    uint16_t dst_streamid;
    uint16_t dst_sub_streamid;

    uint32_t length;

    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;
    uint32_t dstAddrLow;
    uint32_t dstAddrHigh;

    uint8_t linkType;
    uint8_t reserved[3];
    uint32_t reslast[3];
};

// ============================================================================
// SDMA Async Event Record (single slot)
// ============================================================================
struct sdma_event_record_t {
    uint32_t flag;          // Set by flag SQE (non-zero indicates completion)
    uint32_t sq_tail;       // Tail value to commit on completion
    uint64_t channel_info;  // __gm__ batch_write_channel_info_t* (stored as uint64_t)
};

} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_SDMA_TYPES_HPP
