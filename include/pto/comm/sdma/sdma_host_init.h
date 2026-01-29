/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_SDMA_HOST_INIT_H
#define PTO_COMM_SDMA_HOST_INIT_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================
#define PTO_SDMA_MAX_CHAN 40
#define PTO_SDMA_SUCCESS 0
#define PTO_SDMA_ERROR (-1)

// ============================================================================
// Initialization Attributes Structure
// ============================================================================

// Optional initialization attributes
struct pto_comm_init_optional_attr_t {
    int32_t version;                 // Version number
    int32_t data_op_engine_type;     // Data operation engine type
    int64_t shm_init_timeout;        // Initialization timeout (ms)
    int64_t control_operation_timeout; // Control operation timeout (ms)
    int64_t data_operation_timeout;  // Data operation timeout (ms)
    int32_t sockFd;                  // Socket file descriptor (for bootstrap)
};

// Main initialization attributes structure
struct pto_comm_init_attr_t {
    int32_t my_pe;                   // Local PE (Processing Element) rank
    int32_t n_pes;                   // Total number of PEs
    int64_t local_mem_size;          // Local memory size to allocate
    void *comm_args;                 // Communication arguments (for bootstrap)
    char ip_port[256];               // IP:port string for communication
    struct pto_comm_init_optional_attr_t option_attr; // Optional attributes
};

// ============================================================================
// SDMA Operation Resource Information
// ============================================================================

// Host stream information structure
struct pto_host_stream_info_t {
    uint64_t stream_;      // Stream handle
    int32_t dev_id;        // Device ID (die_id)
    int32_t stream_id;     // Stream ID
    uint32_t sq_id;        // Send Queue ID
    uint32_t cq_id;        // Completion Queue ID
    uint32_t logic_cq_id;  // Logical Completion Queue ID
    uint64_t ctx_;         // Context handle
    uint8_t reserved[12];  // Padding to 48 bytes
};

// SDMA operation resource information structure
// This structure holds SDMA stream information and workspace addresses
struct pto_sdma_op_res_info_t {
    struct pto_host_stream_info_t streams[PTO_SDMA_MAX_CHAN];
    uint64_t workspace_addr;   // Workspace address for SDMA operations
};

// ============================================================================
// Global Variables (Host Side)
// ============================================================================

// Global SDMA operation resource info (host side)
extern pto_sdma_op_res_info_t g_pto_sdma_op_res_info;
extern void *g_pto_sdma_op_res_info_device_ptr;
extern uint64_t g_pto_sdma_workspace_addr;

// ============================================================================
// SDMA Initialization Function
// ============================================================================

// SDMA initialization function (host side)
// This function initializes SDMA resources including streams, workspace, and device-side mapping.
//
// It performs the following steps:
// 1. Creates an AICPU stream for kernel execution
// 2. Creates 40 SDMA streams for data transfer
// 3. Allocates shared workspace memory (16KB) for AICPU and AIV communication
// 4. Copies resource info to device memory
// 5. Runs AICPU kernel to initialize device-side SDMA mapping
//
// Parameters:
//   - attributes: Initialization attributes (can be nullptr for default initialization)
//                 If provided, n_pes is used for flag allocation
// Returns:
//   - 0 (PTO_SDMA_SUCCESS) on success
//   - Non-zero error code on failure
int pto_sdma_init(struct pto_comm_init_attr_t *attributes);

// SDMA finalization function (host side)
// This function releases all SDMA resources allocated during initialization.
//
// Returns:
//   - 0 (PTO_SDMA_SUCCESS) on success
//   - Non-zero error code on failure
int pto_sdma_finalize(void);

#ifdef __cplusplus
}
#endif

#endif // PTO_COMM_SDMA_HOST_INIT_H
