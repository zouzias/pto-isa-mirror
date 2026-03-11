/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TRADIXSELECT_HPP
#define TRADIXSELECT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/npu/a5/TCmps.hpp>
#include <pto/npu/a5/TGather.hpp>
#include <pto/npu/a5/Tci.hpp>

namespace pto {

// ============================================================================
// TRadixSelect - Radix Selection Kernel for TopK
// ============================================================================
//
// Implements radix selection algorithm for finding the K-th element threshold
// and producing index-only output. Uses vectorized operations throughout.
//
// ALGORITHM OVERVIEW:
// 1. Initialize: All elements are "active" (in selection set)
// 2. For each byte position (MSD first for largest K):
//    a. Extract byte at current position from all active elements
//    b. Compute cumulative histogram of these bytes (user-provided TCumHist8)
//    c. Find bucket using vectorized approach:
//       - mask = TCMPS(cumHist, totalActive - k, CmpMode::GT)  // cumHist > threshold
//       - idxTile = TCI(0..255)
//       - bucket = TGATHER(idxTile, mask)[0]  // First index where cumHist > threshold
//    d. Update k = k - cumHist[bucket-1] (elements before bucket)
//    e. Keep only elements with byte value == bucket as active
// 3. After all bytes processed, remaining active element indices are output
//
// ASSUMPTIONS:
// - User provides TCumHist8 function with signature:
//   void TCumHist8(CumHistTile& cumHist, ByteTile& src, bool msd);
//   
//   Where:
//   - src: VecTile input of uint8_t byte values (flattened from data)
//   - cumHist: VecTile output [1, 256] of uint16_t cumulative histogram
//   - msd: true if processing MSD (descending count), false for LSD
//
// OUTPUT:
// - Index-only: produces indices of top-K elements, not values
// - For sorted output, apply additional sorting to output indices
//
// ============================================================================

constexpr uint32_t RADIX_BUCKETS = 256;  // 2^8 buckets per digit (byte)

// ============================================================================
// TRadixFindBucketVec: Vectorized bucket finding
// ============================================================================
// Given cumulative histogram and target k, find bucket index using:
//   1. mask = TCMPS(cumHist, inputSize - k, CmpMode::GT)
//   2. idxTile initialized to [0, 1, 2, ..., 255]
//   3. Apply mask-based gather/squeeze to get first matching index
//
// For "largest K" selection (MSD first):
//   cumHist[b] = count of active elements with byte value >= b
//   We want first bucket where cumHist[b] >= k
//   Equivalently: first b where cumHist[b] > (k-1), i.e., TCMPS GT (k-1)
//
// Returns: bucket index [0, 255]
// Updates: newK for next iteration
// ============================================================================
template <typename CumHistTile, typename MaskTile, typename IdxTile>
__tf__ PTO_INTERNAL OP_NAME(TRADIX_FIND_BUCKET)
    OP_TYPE(reduce) uint16_t TRadixFindBucketVec(
        typename CumHistTile::TileDType __in__ cumHist,
        typename MaskTile::TileDType __out__ cmpMask,
        typename IdxTile::TileDType __inout__ idxTile,
        uint32_t k, uint32_t activeCount, uint32_t& newK, bool largest)
{
    using HistT = typename CumHistTile::DType;  // uint16_t
    using IdxT = typename IdxTile::DType;       // uint16_t or uint32_t

    __ubuf__ HistT *histPtr = (__ubuf__ HistT *)__cce_get_tile_ptr(cumHist);
    __ubuf__ uint8_t *maskPtr = (__ubuf__ uint8_t *)__cce_get_tile_ptr(cmpMask);
    __ubuf__ IdxT *idxPtr = (__ubuf__ IdxT *)__cce_get_tile_ptr(idxTile);

    // Step 1: TCMPS - compare cumHist against threshold
    // For largest: find first bucket where cumHist[b] >= k
    //   Equivalent to cumHist[b] > (k-1) when k >= 1
    // For smallest: find first bucket where cumHist[b] >= k
    uint16_t threshold = static_cast<uint16_t>(k - 1);

    __VEC_SCOPE__
    {
        // Load cumulative histogram (256 elements, 2 repeats for uint16_t)
        constexpr uint32_t elemsPerRepeat = REPEAT_BYTE / sizeof(HistT);  // 128 for uint16_t
        constexpr uint32_t numRepeats = CeilDivision(RADIX_BUCKETS, elemsPerRepeat);  // 2

        RegTensor<HistT> vregHist;
        MaskReg preg, cmpReg;

        for (uint16_t r = 0; r < numRepeats; ++r) {
            uint32_t offset = r * elemsPerRepeat;
            uint32_t remaining = RADIX_BUCKETS - offset;
            uint32_t count = (remaining < elemsPerRepeat) ? remaining : elemsPerRepeat;

            preg = CreatePredicate<HistT>(count);

            // Load histogram values
            vlds(vregHist, histPtr, offset, NORM);

            // Compare: cumHist > threshold (find where cumHist >= k)
            vcmps_gt(cmpReg, vregHist, threshold, preg);

            // Store comparison mask
            uint32_t maskBytesPerRepeat = elemsPerRepeat / 8;  // 16 bytes per repeat
            psts(cmpReg, (__ubuf__ uint32_t *)(maskPtr + r * maskBytesPerRepeat), 0, PK);
        }
    }

    // Step 2: Find first set bit in mask (first bucket where cumHist > threshold)
    // This gives us the bucket containing the K-th element
    //
    // Linear scan through mask bytes to find first 1 bit
    // TODO: Could be vectorized with squeeze/compress if available
    uint16_t bucket = RADIX_BUCKETS - 1;  // Default to last bucket
    uint32_t maskBytes = RADIX_BUCKETS / 8;  // 32 bytes for 256 bits

    for (uint32_t byteIdx = 0; byteIdx < maskBytes; ++byteIdx) {
        uint8_t maskByte = maskPtr[byteIdx];
        if (maskByte != 0) {
            // Find first set bit in this byte
            for (uint8_t bit = 0; bit < 8; ++bit) {
                if (maskByte & (1u << bit)) {
                    bucket = static_cast<uint16_t>(byteIdx * 8 + bit);
                    goto found;
                }
            }
        }
    }
found:

    // Step 3: Compute newK = k - cumHist[bucket-1]
    // cumHist[bucket-1] = count of elements with byte value < bucket
    if (bucket > 0) {
        newK = k - static_cast<uint32_t>(histPtr[bucket - 1]);
    } else {
        newK = k;  // bucket 0: all elements before this bucket = 0
    }

    return bucket;
}

// ============================================================================
// TRadixFindBucketVecSqueeze: Alternative using VCI + masked gather
// ============================================================================
// More vectorized approach using:
//   1. Initialize index tile [0, 1, 2, ..., 255]
//   2. TCMPS to create mask
//   3. TGATHER with mask to "squeeze" indices
//   4. First element of squeezed output is the bucket
//
// NOTE: Requires TSqueezeVCI or similar compress operation to be efficient.
//       If not available, falls back to TRadixFindBucketVec above.
// ============================================================================

// Forward declaration - user should implement TSqueezeVCI if available
// template <typename DstIdxTile, typename SrcIdxTile, typename MaskTile>
// PTO_INTERNAL uint32_t TSqueezeVCI(DstIdxTile& dst, SrcIdxTile& src, MaskTile& mask);

// ============================================================================
// TRadixExtractByte: Extract byte at position from data elements
// ============================================================================
template <typename DataTile, typename ByteTile, unsigned elemPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TRADIX_EXTRACT_BYTE)
    OP_TYPE(element_wise) void TRadixExtractByte(
        typename ByteTile::TileDType __out__ dst,
        typename DataTile::TileDType __in__ src,
        uint32_t bytePos, uint32_t validElems)
{
    using DataT = typename DataTile::DType;
    using ByteT = typename ByteTile::DType;

    __ubuf__ DataT *srcPtr = (__ubuf__ DataT *)__cce_get_tile_ptr(src);
    __ubuf__ ByteT *dstPtr = (__ubuf__ ByteT *)__cce_get_tile_ptr(dst);

    constexpr uint32_t dataElems = CCE_VL / sizeof(DataT);
    uint32_t shiftBits = bytePos * 8;
    uint16_t repeatTimes = CeilDivision(validElems, dataElems);

    __VEC_SCOPE__
    {
        RegTensor<DataT> vregData, vregShifted;
        MaskReg preg;

        constexpr auto distValue = std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<ByteT, DistVST::DIST_NORM>())>();

        for (uint16_t j = 0; j < repeatTimes; ++j) {
            uint32_t offset = j * dataElems;
            uint32_t count = (validElems > offset + dataElems) ? dataElems : (validElems - offset);
            preg = CreatePredicate<DataT>(count);

            vlds(vregData, srcPtr, offset, NORM);

            if constexpr (sizeof(DataT) == 1) {
                // No shift needed for uint8_t
                vsts(vregData, dstPtr, offset, distValue, preg);
            } else {
                // Shift right and mask to get target byte
                RegTensor<DataT> vregShiftAmt, vregMask;
                vbr(vregShiftAmt, static_cast<DataT>(shiftBits));
                vbr(vregMask, static_cast<DataT>(0xFF));

                if constexpr (sizeof(DataT) == 2) {
                    vshr((RegTensor<uint16_t>&)vregShifted, (RegTensor<uint16_t>&)vregData,
                         (RegTensor<int16_t>&)vregShiftAmt, preg, MODE_ZEROING);
                    vand((RegTensor<uint16_t>&)vregShifted, (RegTensor<uint16_t>&)vregShifted,
                         (RegTensor<uint16_t>&)vregMask, preg);
                } else {
                    vshr((RegTensor<uint32_t>&)vregShifted, (RegTensor<uint32_t>&)vregData,
                         (RegTensor<int32_t>&)vregShiftAmt, preg, MODE_ZEROING);
                    vand((RegTensor<uint32_t>&)vregShifted, (RegTensor<uint32_t>&)vregShifted,
                         (RegTensor<uint32_t>&)vregMask, preg);
                }

                // Store as uint8_t (truncates to lowest byte)
                vsts((RegTensor<uint8_t>&)vregShifted, dstPtr, offset, distValue, preg);
            }
        }
    }
}

// ============================================================================
// TRadixUpdateActiveIndices: Filter indices to keep only matching bucket
// ============================================================================
// Given current active indices and data, keep only indices where
// data[idx] has byte value == bucket at position bytePos
//
// Input:  activeIdx[0..activeCount-1] - current active element indices
// Output: activeIdx[0..newCount-1] - filtered indices (compacted)
// Returns: newCount
// ============================================================================
template <typename DataTile, typename IdxTile, typename ByteTile, typename MaskTile, unsigned elemPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TRADIX_UPDATE_ACTIVE)
    OP_TYPE(element_wise) uint32_t TRadixUpdateActiveIndices(
        typename IdxTile::TileDType __inout__ activeIdx,
        typename DataTile::TileDType __in__ data,
        typename ByteTile::TileDType __out__ byteTmp,
        typename MaskTile::TileDType __out__ maskTmp,
        uint32_t bucket, uint32_t bytePos, uint32_t activeCount)
{
    using DataT = typename DataTile::DType;
    using IdxT = typename IdxTile::DType;

    __ubuf__ DataT *dataPtr = (__ubuf__ DataT *)__cce_get_tile_ptr(data);
    __ubuf__ IdxT *idxPtr = (__ubuf__ IdxT *)__cce_get_tile_ptr(activeIdx);
    __ubuf__ uint8_t *bytePtr = (__ubuf__ uint8_t *)__cce_get_tile_ptr(byteTmp);
    __ubuf__ uint8_t *maskPtr = (__ubuf__ uint8_t *)__cce_get_tile_ptr(maskTmp);

    constexpr uint32_t idxElems = CCE_VL / sizeof(IdxT);
    uint32_t shiftBits = bytePos * 8;
    uint16_t repeatTimes = CeilDivision(activeCount, idxElems);

    // Step 1: For each active index, gather data and extract byte
    // Step 2: Compare byte with target bucket
    // Step 3: Compact indices where comparison is true

    __VEC_SCOPE__
    {
        RegTensor<IdxT> vregIdx;
        RegTensor<DataT> vregData, vregShifted, vregByte;
        RegTensor<DataT> vregTarget, vregMaskVal;
        MaskReg preg, cmpReg;

        vbr(vregTarget, static_cast<DataT>(bucket));
        vbr(vregMaskVal, static_cast<DataT>(0xFF));

        for (uint16_t r = 0; r < repeatTimes; ++r) {
            uint32_t offset = r * idxElems;
            uint32_t count = (activeCount > offset + idxElems) ? idxElems : (activeCount - offset);
            preg = CreatePredicate<IdxT>(count);

            // Load active indices
            vlds(vregIdx, idxPtr, offset, NORM);

            // Gather data values at those indices
            if constexpr (sizeof(IdxT) == 2) {
                vgather2(vregData, dataPtr, (vector_u16&)vregIdx, preg);
            } else {
                vgather2(vregData, dataPtr, (vector_u32&)vregIdx, preg);
            }

            // Extract byte at bytePos
            if constexpr (sizeof(DataT) == 1) {
                vregByte = vregData;
            } else {
                RegTensor<DataT> vregShiftAmt;
                vbr(vregShiftAmt, static_cast<DataT>(shiftBits));
                if constexpr (sizeof(DataT) == 2) {
                    vshr((RegTensor<uint16_t>&)vregShifted, (RegTensor<uint16_t>&)vregData,
                         (RegTensor<int16_t>&)vregShiftAmt, preg, MODE_ZEROING);
                } else {
                    vshr((RegTensor<uint32_t>&)vregShifted, (RegTensor<uint32_t>&)vregData,
                         (RegTensor<int32_t>&)vregShiftAmt, preg, MODE_ZEROING);
                }
                vand((RegTensor<uint32_t>&)vregByte, (RegTensor<uint32_t>&)vregShifted,
                     (RegTensor<uint32_t>&)vregMaskVal, preg);
            }

            // Compare: byte == bucket
            vcmp_eq(cmpReg, vregByte, vregTarget, preg);

            // Store comparison mask
            uint32_t maskBytesPerRepeat = idxElems / 8;
            psts(cmpReg, (__ubuf__ uint32_t *)(maskPtr + r * maskBytesPerRepeat), 0, PK);
        }
    }

    // Step 3: Compact indices using mask
    // This is the "squeeze" operation - collect indices where mask bit is set
    // TODO: Replace with vectorized TSqueezeVCI if available
    uint32_t newCount = 0;
    for (uint32_t i = 0; i < activeCount; ++i) {
        uint32_t byteIdx = i / 8;
        uint8_t bitIdx = i % 8;
        if (maskPtr[byteIdx] & (1u << bitIdx)) {
            if (newCount != i) {
                idxPtr[newCount] = idxPtr[i];
            }
            newCount++;
        }
    }

    return newCount;
}

// ============================================================================
// TRadixInitIndices: Initialize index tile to [0, 1, 2, ..., n-1]
// ============================================================================
template <typename IdxTile>
PTO_INTERNAL void TRadixInitIndices(IdxTile& idxTile, uint32_t count)
{
    using IdxT = typename IdxTile::DType;
    __ubuf__ IdxT *idxPtr = (__ubuf__ IdxT *)__cce_get_tile_ptr(idxTile.data());

    // Use TCI pattern for vectorized initialization
    // Scalar fallback for now
    for (uint32_t i = 0; i < count; ++i) {
        idxPtr[i] = static_cast<IdxT>(i);
    }
}

// ============================================================================
// TRADIXSELECT_IMPL: Main radix selection implementation (index output)
// ============================================================================
//
// Template Parameters:
//   DataTile:    Input data tile (uint8_t/uint16_t/uint32_t)
//   IdxTile:     Output index tile (uint16_t or uint32_t)
//   ByteTile:    Temporary for byte extraction (uint8_t)
//   CumHistTile: Temporary for cumulative histogram (uint16_t, [1,256])
//   MaskTile:    Temporary for comparison masks (uint8_t)
//   CumHistFunc: User-provided cumulative histogram function
//
// Parameters:
//   data:        Input data tile
//   outIdx:      Output index tile (top-K indices)
//   byteTmp:     Temporary buffer for extracted bytes
//   cumHistTmp:  Temporary buffer for cumulative histogram [1, 256]
//   maskTmp:     Temporary buffer for comparison masks
//   bucketIdxTmp: Temporary buffer for bucket indices [1, 256]
//   k:           Select K-th element (1 = largest/smallest)
//   largest:     true for K largest, false for K smallest
//   cumHistFunc: Function to compute cumulative histogram
//                Signature: void(CumHistTile& dst, ByteTile& src, bool msd)
//
// Output:
//   outIdx: Indices of top-K elements (first k entries are valid)
//   Returns: Number of valid indices (== k on success)
//
// ============================================================================
template <typename DataTile, typename IdxTile, typename ByteTile, typename CumHistTile,
          typename MaskTile, typename CumHistFunc>
PTO_INTERNAL uint32_t TRADIXSELECT_IMPL(
    DataTile& data, IdxTile& outIdx,
    ByteTile& byteTmp, CumHistTile& cumHistTmp, MaskTile& maskTmp, IdxTile& bucketIdxTmp,
    uint32_t k, bool largest, CumHistFunc cumHistFunc)
{
    using DataT = typename DataTile::DType;
    using IdxT = typename IdxTile::DType;

    // Static validation
    static_assert(std::is_same_v<DataT, uint8_t> || std::is_same_v<DataT, uint16_t> ||
                      std::is_same_v<DataT, uint32_t>,
                  "TRADIXSELECT: Data must be uint8_t, uint16_t, or uint32_t.");
    static_assert(std::is_same_v<IdxT, uint16_t> || std::is_same_v<IdxT, uint32_t>,
                  "TRADIXSELECT: Index type must be uint16_t or uint32_t.");
    static_assert(std::is_same_v<typename ByteTile::DType, uint8_t>,
                  "TRADIXSELECT: Byte tile must be uint8_t.");
    static_assert(std::is_same_v<typename CumHistTile::DType, uint16_t>,
                  "TRADIXSELECT: Cumulative histogram must be uint16_t.");
    static_assert(CumHistTile::Cols >= RADIX_BUCKETS,
                  "TRADIXSELECT: Cumulative histogram must have >= 256 columns.");

    constexpr uint32_t numBytes = sizeof(DataT);
    constexpr uint32_t elemPerRepeat = REPEAT_BYTE / sizeof(DataT);

    uint32_t totalElems = data.GetValidRow() * data.GetValidCol();

    // Runtime validation
    PTO_ASSERT(k > 0 && k <= totalElems,
               "TRADIXSELECT: k must be in range [1, totalElems].");

    // Initialize active indices to [0, 1, 2, ..., totalElems-1]
    TRadixInitIndices(outIdx, totalElems);
    uint32_t activeCount = totalElems;
    uint32_t currentK = k;

    // Process bytes from MSD to LSD (for largest) or LSD to MSD (for smallest)
    for (uint32_t byteIdx = 0; byteIdx < numBytes && activeCount > 0; ++byteIdx) {
        uint32_t bytePos = largest ? (numBytes - 1 - byteIdx) : byteIdx;

        // Step 1: Extract bytes from active elements
        // For efficiency, we extract bytes from data[activeIdx[i]] for all active indices
        // This requires gather operation
        TRadixExtractByte<DataTile, ByteTile, elemPerRepeat>(
            byteTmp.data(), data.data(), bytePos, totalElems);

        // Step 2: Compute cumulative histogram
        // User function signature: void cumHistFunc(CumHistTile& dst, ByteTile& src, bool msd)
        cumHistFunc(cumHistTmp, byteTmp, largest);

        // Step 3: Find bucket containing K-th element (vectorized)
        uint32_t newK;
        uint16_t bucket = TRadixFindBucketVec<CumHistTile, MaskTile, IdxTile>(
            cumHistTmp.data(), maskTmp.data(), bucketIdxTmp.data(),
            currentK, activeCount, newK, largest);

        // Step 4: Update active indices to keep only elements in selected bucket
        activeCount = TRadixUpdateActiveIndices<DataTile, IdxTile, ByteTile, MaskTile, elemPerRepeat>(
            outIdx.data(), data.data(), byteTmp.data(), maskTmp.data(),
            bucket, bytePos, activeCount);

        currentK = newK;

        // Early termination check
        if (activeCount <= k) {
            // Found all top-K candidates
            break;
        }
    }

    // Output: outIdx[0..min(activeCount, k)-1] contains top-K indices
    return (activeCount < k) ? activeCount : k;
}

// ============================================================================
// Convenience wrapper without bucketIdxTmp (uses internal buffer)
// ============================================================================
template <typename DataTile, typename IdxTile, typename ByteTile, typename CumHistTile,
          typename MaskTile, typename CumHistFunc>
PTO_INTERNAL uint32_t TRADIXSELECT_TOPK_IMPL(
    DataTile& data, IdxTile& outIdx,
    ByteTile& byteTmp, CumHistTile& cumHistTmp, MaskTile& maskTmp,
    uint32_t k, bool largest, CumHistFunc cumHistFunc)
{
    // Temporary for bucket index finding [1, 256] uint16_t
    // Note: This should be allocated by caller for efficiency
    // Using cumHistTmp space is possible if we're careful about ordering

    // For now, delegate to main implementation
    // Caller should provide bucketIdxTmp separately
    IdxTile bucketIdxTmp;  // Placeholder - caller should allocate
    return TRADIXSELECT_IMPL(data, outIdx, byteTmp, cumHistTmp, maskTmp, bucketIdxTmp,
                             k, largest, cumHistFunc);
}

} // namespace pto

#endif // TRADIXSELECT_HPP
