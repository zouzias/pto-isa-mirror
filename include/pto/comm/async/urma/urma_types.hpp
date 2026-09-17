/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_TYPES_HPP
#define PTO_COMM_ASYNC_URMA_TYPES_HPP

#include <cstdint>

#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {
namespace urma {

// ============================================================================
// Constants
// ============================================================================
constexpr uint32_t kUrmaMaxPollTimes = 1000000;
constexpr uint64_t kCacheLineSize = 64;
constexpr size_t kUrmaEidBytes = 16;

constexpr uint64_t kUrmaMaxWqeTransferBytes = 256ULL * 1024ULL * 1024ULL;

constexpr uint64_t kUrmaMinSliceBytes = kUrmaMaxWqeTransferBytes;
constexpr uint64_t kUrmaSliceAlignBytes = 512ULL;

constexpr uint32_t kUrmaAutoAivCount = UINT32_MAX;

constexpr uint32_t kUrmaSqeSizeBytes = 48;
constexpr uint32_t kUrmaSgeSizeBytes = 16;
constexpr uint32_t kUrmaSqeRmtEidLOffset = 16;
constexpr uint32_t kUrmaSqeRmtEidHOffset = 24;
constexpr uint32_t kUrmaSqeRmtAddrLOffset = 40;
constexpr uint32_t kUrmaSqeRmtAddrHOffset = 44;

constexpr uint32_t kUrmaNotifySetSlotCount = 8U;
constexpr uint32_t kUrmaNotifyResourceCachelineBytes = 64U;

// One cache-line-aligned region is owned by one (peer, QP). SET sources are
// reused in batches after draining the queue. FAA writes its ignored old value
// to a fixed sink in the same cache line.
struct alignas(kUrmaNotifyResourceCachelineBytes) UrmaNotifyResourceRegion {
    int32_t setValues[kUrmaNotifySetSlotCount];
    uint32_t nextSetSlot;
    uint32_t setRingStarted;
    int32_t faaResult;
    uint8_t reserved
        [kUrmaNotifyResourceCachelineBytes - sizeof(setValues) - sizeof(nextSetSlot) - sizeof(setRingStarted) -
         sizeof(faaResult)];
};

static_assert(
    sizeof(UrmaNotifyResourceRegion) == kUrmaNotifyResourceCachelineBytes,
    "URMA notify resource region layout mismatch");
static_assert(
    alignof(UrmaNotifyResourceRegion) == kUrmaNotifyResourceCachelineBytes,
    "URMA notify resource region must be cache-line aligned");

// ============================================================================
// UrmaOpcode — URMA operation codes (binary-compatible with hcomm UB ABI)
// ============================================================================
enum class UrmaOpcode : uint32_t {
    SEND = 0,
    SEND_WITH_IMM,
    SEND_WITH_INV,
    WRITE,
    WRITE_WITH_IMM,
    WRITE_WITH_NOTIFY,
    READ,
    CAS,
    ATOMIC_SWAP,
    ATOMIC_STORE,
    ATOMIC_LOAD,
    FAA = 0xb,
    NOP = 0x11
};

// ============================================================================
// UrmaLayout — how the SQ/CQ context table is indexed
//
// PER_PEER:    one jetty per peer, jetty i only reaches peer i. Selecting a peer
//              is the same act as selecting a queue, so a ctx row is a peer, and
//              concurrency is capped at one AIV per peer.
// SHARED_POOL: a pool of jetties, each reaching every peer. Peer and queue are
//              chosen independently, so a ctx row is a jetty. The pool is carved
//              into one exclusive run of jettiesPerCore jetties per AIV, so a
//              session derives its run from get_block_idx() and no caller ever
//              names a jetty.
// ============================================================================
enum class UrmaLayout : uint32_t { PER_PEER = 0, SHARED_POOL = 1 };

// ============================================================================
// UrmaInfo — device-side workspace root
// ============================================================================
struct UrmaInfo {
    uint32_t rowsPerPeer; // SQ/CQ ctx rows occupied per peer; >= 1
    uint32_t rankCount;
    uint64_t sqPtr;
    uint64_t rqPtr;
    uint64_t scqPtr;
    uint64_t rcqPtr;
    uint64_t memPtr;
    uint64_t notifyPoolPtr;
    UrmaLayout layout;
    uint32_t jettyCount;
    uint32_t jettiesPerCore;
};

// ============================================================================
// UrmaMemInfo — per-peer memory authentication info
// ============================================================================
struct UrmaMemInfo {
    bool tokenValueValid;
    uint32_t rmtJettyType : 2;
    uint8_t targetHint;
    uint32_t tpn;
    uint32_t tid;
    uint32_t rmtTokenValue;
    uint32_t len;
    uint64_t addr;
    uint64_t eidAddr;
    uint32_t localTokenId;
    uint32_t notifyTokenId;
};

// ============================================================================
// UrmaDbMode — doorbell mode for URMA queues
// ============================================================================
enum class UrmaDbMode : int32_t { INVALID_DB = -1, HW_DB = 0, SW_DB };

// ============================================================================
// UrmaWQCtx — send/receive work queue context
// ============================================================================
struct UrmaWQCtx {
    uint32_t wqn;
    uint64_t bufAddr;
    uint32_t wqeShiftSize;
    uint32_t depth;
    uint64_t headAddr;
    uint64_t tailAddr;
    UrmaDbMode dbMode;
    uint64_t dbAddr;
    uint32_t sl;
    // Software WQE sequence for CQ accounting; initialized from the CQ tail.
    uint32_t submittedWqeCount;
};

static_assert(sizeof(UrmaWQCtx) == 64U, "URMA WQ context layout mismatch");

// ============================================================================
// UrmaCqCtx — completion queue context
// ============================================================================
struct UrmaCqCtx {
    uint32_t cqn;
    uint64_t bufAddr;
    uint32_t cqeShiftSize;
    uint32_t depth;
    uint64_t headAddr;
    uint64_t tailAddr;
    UrmaDbMode dbMode;
    uint64_t dbAddr;
};

// ============================================================================
// UrmaSqeCtx — 48-byte SQE (ABI-compatible with hcomm UB JFS WQE layout)
// ============================================================================
struct UrmaSqeCtx {
    /* byte 0 - 4 */
    uint32_t sqeBbIdx : 16;
    uint32_t flag : 8;
    uint32_t rsv0 : 3;
    uint32_t nf : 1;
    uint32_t tokenEn : 1;
    uint32_t rmtJettyType : 2;
    uint32_t owner : 1;
    /* byte 4 - 8 */
    uint32_t targetHint : 8;
    uint32_t opcode : 8;
    uint32_t rsv1 : 6;
    uint32_t inlineMsgLen : 10;
    /* byte 8 - 12 */
    uint32_t tpId : 24;
    uint32_t sgeNum : 8;
    /* byte 12 - 16 */
    uint32_t rmtJettyOrSegId : 20;
    uint32_t rsv2 : 12;
    /* byte 16 - 32 */
    uint64_t rmtEidL;
    uint64_t rmtEidH;
    /* byte 32 - 36 */
    uint32_t rmtTokenValue;
    /* byte 36 - 40 */
    uint32_t udfType : 8;
    uint32_t reduceDataType : 4;
    uint32_t reduceOpcode : 4;
    uint32_t rsv3 : 16;
    /* byte 40 - 48 */
    uint32_t rmtAddrLOrTokenId;
    uint32_t rmtAddrHOrTokenValue;
};

// ============================================================================
// UrmaSgeCtx — scatter-gather entry
// ============================================================================
struct UrmaSgeCtx {
    uint32_t len;
    uint32_t tokenId;
    uint64_t va;
};

// ============================================================================
// UrmaJfcCqeCtx — 64-byte CQE (ABI-compatible with hcomm UB JFC CQE layout)
// ============================================================================
struct UrmaJfcCqeCtx {
    /* DW0 */
    uint32_t sR : 1;
    uint32_t isJetty : 1;
    uint32_t owner : 1;
    uint32_t inlineEn : 1;
    uint32_t opcode : 3;
    uint32_t fd : 1;
    uint32_t rsv : 8;
    uint32_t substatus : 8;
    uint32_t status : 8;
    /* DW1 */
    uint32_t entryIdx : 16;
    uint32_t localNumL : 16;
    /* DW2 */
    uint32_t localNumH : 4;
    uint32_t rmtIdx : 20;
    uint32_t rsv1 : 8;
    /* DW3 */
    uint32_t tpn : 24;
    uint32_t rsv2 : 8;
    /* DW4 */
    uint32_t byteCnt;
    /* DW5 ~ DW6 */
    uint32_t userDataL;
    uint32_t userDataH;
    /* DW7 ~ DW10 */
    uint32_t rmtEid[4];
    /* DW11 ~ DW12 */
    uint32_t dataL;
    uint32_t dataH;
    /* DW13 ~ DW15 */
    uint32_t inlineData[3];
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_TYPES_HPP
