/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_FFN_COMBINE_V8_DEVICE_DEBUG_DECL_H
#define DISPATCH_FFN_COMBINE_V8_DEVICE_DEBUG_DECL_H

#include "profile_debug_config.h"

namespace dispatch_ffn_combine_v8 {

constexpr uint64_t kDispatchLayoutDebugMagic = 0x5635444953504c59ULL;       // V5DISPLY
constexpr uint64_t kFrontCheckLayoutDebugMagic = 0x5635464e45574c59ULL;     // V5FNEWLY
constexpr uint64_t kDispatchGatherLayoutDebugMagic = 0x56354450324c5954ULL; // V5DP2LYT
constexpr uint64_t kFrontLayoutDebugMagic = 0x56354652324c5954ULL;          // V5FR2LYT
constexpr uint64_t kDispatchGroupDoneDebugMagic = 0x5635444752444f4eULL;    // V5GRDON
constexpr uint64_t kDispatchCopyChunkDebugMagic = 0x5635444350594348ULL;    // V5DCPYCH
constexpr uint64_t kGmm1LayoutDebugMagic = 0x5635474d4d314c59ULL;           // V5GMM1LY
constexpr uint64_t kGmm1SyncDebugMagic = 0x5635474d4d315359ULL;             // V5GMM1SY
constexpr uint64_t kGmm1TaskDebugMagic = 0x5635474d4d315453ULL;             // V5GMM1TS
constexpr uint64_t kGmm1DoneDebugMagic = 0x5635474d4d31444eULL;             // V5GMM1DN
constexpr uint64_t kSwigluLayoutDebugMagic = 0x5635535749474c59ULL;         // V5SWIGLY
constexpr uint64_t kSwigluC2VDebugMagic = 0x5635535743325644ULL;            // V5SWC2VD
constexpr uint64_t kSwigluSegmentDebugMagic = 0x5635535749475347ULL;        // V5SWIGSG
constexpr uint64_t kSwigluTaskDebugMagic = 0x563553575441534bULL;           // V5SWTASK
constexpr uint64_t kSwigluDoneDebugMagic = 0x56355357444f4e45ULL;           // V5SWDONE
constexpr uint64_t kSwigluFinalSyncDebugMagic = 0x56355357464e5359ULL;      // V5SWFNSY
constexpr uint64_t kGmm2LayoutDebugMagic = 0x5635474d4d324c59ULL;           // V5GMM2LY
constexpr uint64_t kGmm2V2CDebugMagic = 0x5635474d32563243ULL;              // V5GM2V2C
constexpr uint64_t kGmm2SegmentDebugMagic = 0x5635474d3253474dULL;          // V5GM2SGM
constexpr uint64_t kGmm2TaskDebugMagic = 0x5635474d3254534bULL;             // V5GM2TSK
constexpr uint64_t kGmm2DoneDebugMagic = 0x5635474d32444f4eULL;             // V5GM2DON
constexpr uint64_t kGmm2FinalSyncDebugMagic = 0x5635474d32465359ULL;        // V5GM2FSY
constexpr uint64_t kCombineLayoutDebugMagic = 0x5635434f4d424c59ULL;        // V5COMBLY
constexpr uint64_t kCombineReadyDebugMagic = 0x5635434f4d524459ULL;         // V5COMRDY
constexpr uint64_t kCombineMetaDebugMagic = 0x5635434f4d4d4554ULL;          // V5COMMET
constexpr uint64_t kCombineLoadDebugMagic = 0x5635434f4d4c4f44ULL;          // V5COMLOD
constexpr uint64_t kCombineDequantDebugMagic = 0x5635434f4d445154ULL;       // V5COMDQT
constexpr uint64_t kCombineStoreDebugMagic = 0x5635434f4d535452ULL;         // V5COMSTR
constexpr uint64_t kCombineFinalizeDebugMagic = 0x5635434f4d46494eULL;      // V5COMFIN
constexpr uint64_t kCombineTaskDebugMagic = 0x5635434f4d54534bULL;          // V5COMTSK
constexpr uint64_t kCombineDoneDebugMagic = 0x5635434f4d444f4eULL;          // V5COMDON
constexpr uint64_t kUnpermuteLayoutDebugMagic = 0x5635554e504c5954ULL;      // V5UNPLYT
constexpr uint64_t kUnpermuteTaskDebugMagic = 0x5635554e5054534bULL;        // V5UNPTSK
constexpr uint64_t kUnpermuteMetaDebugMagic = 0x5635554e504d4554ULL;        // V5UNPMET
constexpr uint64_t kUnpermuteAccumDebugMagic = 0x5635554e50414343ULL;       // V5UNPACC
constexpr uint64_t kUnpermuteOutputDebugMagic = 0x5635554e504f5554ULL;      // V5UNPOUT
constexpr uint64_t kUnpermuteDoneDebugMagic = 0x5635554e50444f4eULL;        // V5UNPDON

constexpr uint32_t kFrontProducerClassMultiCore = 1U;
constexpr uint32_t kFrontProducerClassSingleCore = 2U;
constexpr uint32_t kFrontProducerClassFullLoad = 3U;
constexpr uint32_t kCombineScaleReadDirectScalar = 1U;
constexpr uint32_t kCombineDequantOpCounts = 0x00010101U;

struct DeviceDebug {
    template <typename Stage>
    AICORE inline static void FrontLogInitDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static void FrontLogBuildHistogramEnter(const Stage &op, uint32_t tokenStart, uint32_t tokenEnd);
    template <typename Stage>
    AICORE inline static void FrontLogBuildHistogramStored(const Stage &op, uint32_t expertStart);
    template <typename Stage>
    AICORE inline static void FrontLogProcessDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static bool FrontDebugEnabled(const Stage &op);
    template <typename Stage>
    AICORE inline static uint32_t FrontStopStep(const Stage &op);
    template <typename Stage>
    AICORE inline static bool FrontDebugWriteEnabled(const Stage &op);
    template <typename Stage>
    AICORE inline static uint32_t FrontEffectiveStopStep(const Stage &op);
    template <typename Stage>
    AICORE inline static void FrontWriteBoundaryDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static void FrontSingleCoreWriteLayoutDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static void FrontFullLoadWriteLayoutDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static void FrontMultiCoreWriteLayoutDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static bool DispatchGatherDebugEnabled(const Stage &op);
    template <typename Stage>
    AICORE inline static uint32_t DispatchGatherStopStep(const Stage &op);
    template <typename Stage>
    AICORE inline static bool DispatchGatherDebugControlEnabled(const Stage &op);
    template <typename Stage>
    AICORE inline static void DispatchGatherWriteLayoutDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t DispatchMetadataDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t DispatchMetadataDebugBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t DispatchTaskStatsDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t DispatchTaskStatsDebugBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t DispatchGroupDoneDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t DispatchGroupDoneDebugBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static bool DispatchDebugEnabled(const Stage &op);
    template <typename Stage>
    AICORE inline static void DispatchWriteLayoutDebugHeader(const Stage &op);
    template <typename Stage>
    AICORE inline static void DispatchWriteGroupMetadataDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static void DispatchWriteTaskSplitDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static void DispatchNotifyGmm1GroupReady(const Stage &op, uint32_t groupIdx, uint32_t groupBase,
                                                           uint32_t currentM);
    template <typename Stage>
    AICORE inline static bool Gmm1DebugEnabled(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm1SyncDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm1SyncDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm1SyncDebugBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm1TaskDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm1TaskDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm1TaskDebugBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm1DoneDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm1DoneDebugBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static void Gmm1WriteLayoutDebugHeader(const Stage &op);
    template <typename Stage>
    AICORE inline static void Gmm1WriteSyncDebug(const Stage &op, uint32_t groupIdx, uint32_t groupBase,
                                                 uint32_t currentM);
    template <typename Stage>
    AICORE inline static void Gmm1WriteTaskDebug(const Stage &op, uint32_t groupIdx, uint32_t groupBase,
                                                 uint32_t currentMRaw, uint32_t currentM, uint32_t expertTokenNums,
                                                 uint32_t startCoreIdx, uint32_t coreLoops);
    template <typename Stage>
    AICORE inline static void Gmm1WriteDoneDebug(const Stage &op, uint32_t segmentIdx, uint32_t segmentStartExpert,
                                                 uint32_t segmentEndExpert, uint32_t segmentRowBase,
                                                 uint32_t segmentRows);
    template <typename Stage>
    AICORE inline static bool SwigluDebugEnabled(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t SwigluC2VDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t SwigluC2VDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t SwigluSegmentDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t SwigluSegmentDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t SwigluTaskDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t SwigluTaskDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t SwigluDoneDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t SwigluDoneDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t SwigluFinalSyncDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t SwigluFinalSyncDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static void SwigluWriteLayoutDebugHeader(const Stage &op);
    template <typename Stage>
    AICORE inline static void SwigluWriteC2VDebug(const Stage &op, uint32_t segmentIdx);
    template <typename Stage>
    AICORE inline static void SwigluWriteSegmentDebug(const Stage &op, uint32_t segmentIdx, uint32_t segmentStartExpert,
                                                      uint32_t segmentEndExpert, uint32_t segmentRowBase,
                                                      uint32_t segmentRows, uint32_t cumsumRows,
                                                      uint32_t expertTokenRows);
    template <typename Stage>
    AICORE inline static void SwigluWriteTaskDebug(const Stage &op, uint32_t segmentIdx, uint32_t segmentStartExpert,
                                                   uint32_t segmentEndExpert, uint32_t segmentRowBase,
                                                   uint32_t segmentRows, uint32_t localRowStart, uint32_t localRows,
                                                   uint32_t rowSplitBase, uint32_t rowSplitRem);
    template <typename Stage>
    AICORE inline static void SwigluWriteDoneDebug(const Stage &op, uint32_t segmentIdx, uint32_t segmentRowBase,
                                                   uint32_t segmentRows, uint32_t localRowStart, uint32_t localRows);
    template <typename Stage>
    AICORE inline static void SwigluWriteFinalSyncDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static bool Gmm2DebugEnabled(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm2V2CDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm2V2CDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm2SegmentDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm2SegmentDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm2TaskDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm2TaskDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm2DoneDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm2DoneDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm2FinalSyncDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t Gmm2FinalSyncDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static void Gmm2WriteLayoutDebugHeader(const Stage &op);
    template <typename Stage>
    AICORE inline static void Gmm2WriteV2CDebug(const Stage &op, uint32_t segmentIdx);
    template <typename Stage>
    AICORE inline static void Gmm2WriteSegmentDebug(const Stage &op, uint32_t segmentIdx, uint32_t segmentStartExpert,
                                                    uint32_t segmentEndExpert, uint32_t segmentRowBase,
                                                    uint32_t segmentRows, uint32_t cumsumRows,
                                                    uint32_t expertTokenRows);
    template <typename Stage>
    AICORE inline static void Gmm2WriteTaskDebug(const Stage &op, uint32_t segmentIdx, uint32_t groupIdx,
                                                 uint32_t groupBase, uint32_t currentMRaw, uint32_t currentM,
                                                 uint32_t expertTokenNums, uint32_t startCoreIdx, uint32_t coreLoops);
    template <typename Stage>
    AICORE inline static void Gmm2WriteDoneDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static void Gmm2WriteFinalSyncDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static bool CombineDebugEnabled(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineReadyDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineReadyDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineMetaDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineMetaDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineLoadDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineLoadDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineDequantDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineDequantDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineStoreDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineStoreDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineFinalizeDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineFinalizeDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineTaskDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineTaskDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineDoneDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t CombineDoneDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static void CombineWriteLayoutDebugHeader(const Stage &op);
    template <typename Stage>
    AICORE inline static void CombineWriteReadyDebug(const Stage &op, uint32_t groupIdx, bool aivSyncAfterWait,
                                                     uint64_t waitStartSyscnt, uint64_t waitEndSyscnt,
                                                     uint64_t syncAllStartSyscnt, uint64_t syncAllEndSyscnt);
    template <typename Stage>
    AICORE inline static void CombineWriteMetaDebug(const Stage &op, uint32_t groupIdx, uint32_t srcRank,
                                                    uint32_t groupBase, uint32_t currentM);
    template <typename Stage>
    AICORE inline static void CombineWriteDirectLargeMetaDebug(const Stage &op, uint32_t groupIdx, uint32_t srcRank,
                                                               uint32_t groupBase, uint32_t currentM);
    template <typename Stage>
    AICORE inline static void CombineWriteLoadDebug(const Stage &op, uint32_t groupIdx, uint32_t srcRank,
                                                    uint32_t srcRow, uint32_t dstRow, uint32_t rows, float scaleValue,
                                                    float cFirst, float cLast);
    template <typename Stage>
    AICORE inline static void CombineWriteDequantDebug(const Stage &op, uint32_t groupIdx, uint32_t srcRank,
                                                       uint32_t srcRow, uint32_t rows, float scaleValue,
                                                       float fp32BeforeFirst, float fp32AfterFirst, float fp32AfterLast,
                                                       float dFirst, float dLast);
    template <typename Stage>
    AICORE inline static void CombineWriteStoreDebug(const Stage &op, uint32_t groupIdx, uint32_t srcRank,
                                                     uint32_t srcRow, uint32_t dstRow, uint32_t rows,
                                                     bool remoteBaseValid, float storedFirst, float storedLast);
    template <typename Stage>
    AICORE inline static void CombineWriteFinalizeDebug(const Stage &op, uint64_t finalizeWaitStartSyscnt,
                                                        uint64_t finalizeWaitEndSyscnt, uint64_t syncAllStartSyscnt,
                                                        uint64_t syncAllEndSyscnt, uint64_t resetStartSyscnt,
                                                        uint64_t resetEndSyscnt, uint64_t crossRankSyncStartSyscnt,
                                                        uint64_t crossRankSyncEndSyscnt, uint32_t resetElems,
                                                        bool resetWriter);
    template <typename Stage>
    AICORE inline static void CombineWriteTaskDebug(const Stage &op, uint32_t groupIdx, uint32_t groupBase,
                                                    uint32_t currentM);
    template <typename Stage>
    AICORE inline static void CombineWriteDoneDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static bool UnpermuteDebugEnabled(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t UnpermuteTaskDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t UnpermuteTaskDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t UnpermuteMetaDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t UnpermuteMetaDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t UnpermuteAccumDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t UnpermuteAccumDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t UnpermuteOutputDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t UnpermuteOutputDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t UnpermuteDoneDebugOffset(const Stage &op);
    template <typename Stage>
    AICORE inline static uint64_t UnpermuteDoneDebugTableBytes(const Stage &op);
    template <typename Stage>
    AICORE inline static void UnpermuteWriteLayoutDebugHeader(const Stage &op);
    template <typename Stage>
    AICORE inline static void UnpermuteWriteTaskDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static void UnpermuteWriteMetaDebug(Stage &op, uint32_t token, uint32_t topkIdx, int32_t expandedRow,
                                                      float prob, uint32_t batchStart, uint32_t localToken);
    template <typename Stage>
    AICORE inline static void UnpermuteWriteAccumDebug(Stage &op, uint32_t token, uint32_t col, uint32_t cols,
                                                       uint32_t validTopk, uint32_t topkProcessed,
                                                       int32_t firstExpandedRow, float firstProb);
    template <typename Stage>
    AICORE inline static void UnpermuteWriteOutputDebug(Stage &op, uint32_t token, uint32_t col, uint32_t cols,
                                                        uint32_t validTopk, uint32_t topkProcessed);
    template <typename Stage>
    AICORE inline static void UnpermuteWriteDoneDebug(const Stage &op);
    template <typename Stage>
    AICORE inline static void DispatchFfnLogProcessDebug(const Stage &op);
};

} // namespace dispatch_ffn_combine_v8

#endif // DISPATCH_FFN_COMBINE_V8_DEVICE_DEBUG_DECL_H

#ifdef DISPATCH_FFN_COMBINE_V8_DEVICE_DEBUG_IMPLEMENTATION
#ifndef DISPATCH_FFN_COMBINE_V8_DEVICE_DEBUG_IMPL_H
#define DISPATCH_FFN_COMBINE_V8_DEVICE_DEBUG_IMPL_H

namespace dispatch_ffn_combine_v8 {

#if defined(__DAV_VEC__)
template <typename Stage>
AICORE inline void DeviceDebug::FrontLogInitDebug(const Stage &op)
{
#ifdef _DEBUG
    if ASCEND_IS_AIV {
        cce::printf(
            "V8 FrontReorder::Init AIV block=%d sub=%d coreIdx=%d coreNum=%d stage=%d M=%d topK=%d "
            "expertNum=%d aligned=%d\n",
            int(get_block_idx()), int(get_subblockid()), int(op.coreIdx_), int(op.coreNum_), int(op.stageNum_),
            int(op.problemM_), int(op.topK_), int(op.expertNum_), int(op.expertNumAligned_));
    } else {
        cce::printf("V8 FrontReorder::Init AIC block=%d coreIdx=%d coreNum=%d stage=%d\n", int(get_block_idx()),
                    int(op.coreIdx_), int(op.coreNum_), int(op.stageNum_));
    }
#endif
}

template <typename Stage>
AICORE inline void DeviceDebug::FrontLogBuildHistogramEnter(const Stage &op, uint32_t tokenStart, uint32_t tokenEnd)
{
#ifdef _DEBUG
    cce::printf("V8 BuildHistogram enter block=%d sub=%d coreIdx=%d tokenStart=%d tokenEnd=%d expertElems=%d\n",
                int(get_block_idx()), int(get_subblockid()), int(op.coreIdx_), int(tokenStart), int(tokenEnd),
                int((tokenEnd - tokenStart) * op.topK_));
#else
    (void)tokenStart;
    (void)tokenEnd;
#endif
}

template <typename Stage>
AICORE inline void DeviceDebug::FrontLogBuildHistogramStored(const Stage &op, uint32_t expertStart)
{
#ifdef _DEBUG
    cce::printf("V8 BuildHistogram stored block=%d sub=%d coreIdx=%d expertStart=%d count0=%d count1=%d\n",
                int(get_block_idx()), int(get_subblockid()), int(op.coreIdx_), int(expertStart),
                int(PtoGetValue<int32_t, kCountAlign>(op.CountUb(), 0)),
                int(PtoGetValue<int32_t, kCountAlign>(op.CountUb(), 1)));
#else
    (void)expertStart;
#endif
}

template <typename Stage>
AICORE inline void DeviceDebug::FrontLogProcessDebug(const Stage &op)
{
#ifdef _DEBUG
    if ASCEND_IS_AIV {
        cce::printf("V8 FrontReorder::Process AIV block=%d sub=%d\n", int(get_block_idx()), int(get_subblockid()));
    } else {
        cce::printf("V8 FrontReorder::Process AIC block=%d\n", int(get_block_idx()));
    }
#endif
}

template <typename Stage>
AICORE inline bool DeviceDebug::FrontDebugEnabled(const Stage &op)
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DEVICE_DEBUG
    return op.tilingData_->frontDebugTiling.frontDebugMode != 0U;
#else
    (void)op;
    return false;
#endif
}

template <typename Stage>
AICORE inline uint32_t DeviceDebug::FrontStopStep(const Stage &op)
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DEVICE_DEBUG
    return op.tilingData_->frontDebugTiling.frontStopStep;
#else
    (void)op;
    return 0U;
#endif
}

template <typename Stage>
AICORE inline bool DeviceDebug::FrontDebugWriteEnabled(const Stage &op)
{
    return DeviceDebug::FrontDebugEnabled(op);
}

template <typename Stage>
AICORE inline uint32_t DeviceDebug::FrontEffectiveStopStep(const Stage &op)
{
    const uint32_t stopStep = DeviceDebug::FrontStopStep(op);
    return stopStep == 0U ? 8U : stopStep;
}

template <typename Stage>
AICORE inline void DeviceDebug::FrontWriteBoundaryDebug(const Stage &op)
{
    if (op.coreIdx_ != 0U || !DeviceDebug::FrontDebugWriteEnabled(op)) {
        return;
    }
    const auto &front = op.tilingData_->frontReorderTiling;
    const auto &frontDebug = op.tilingData_->frontDebugTiling;
    if (frontDebug.frontDebugBytes < sizeof(DispatchFFNCombineFrontLayoutDebug)) {
        return;
    }

    __gm__ DispatchFFNCombineFrontLayoutDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineFrontLayoutDebug *>(op.workspaceGM_ + frontDebug.frontDebugOffset);
    const uint32_t peerRanks = op.rankSize_ == 0U ? 0U : op.rankSize_ - 1U;
    debug->countMarkerBias = static_cast<uint32_t>(kFrontCountMarkerBias);
    debug->tokenPerExpertRowBytes = op.expertNumAligned_ * sizeof(int32_t);
    debug->countPublishedRanks = peerRanks;
    debug->countWaitedRanks = peerRanks;
    debug->countMarkerRestored = 1U;
    debug->countPreSumBuilt = 1U;
    debug->countOldNotifyWaitUsed = 0U;
    debug->cumsumCore0Only = 1U;
    debug->cumsumRows = op.rankSize_;
    debug->cumsumLocalExpertCount = op.expertPerRank_;
    debug->cumsumFormulaReference = 1U;
    debug->cumsumExpertTokenNumsWritten = 1U;
    debug->offsetAPublished = 1U;
    debug->tokenPerExpertPublished = 1U;
    debug->preSumPublished = 1U;
    debug->cumsumPublished = 1U;
    debug->expertTokenNumsPublished = 1U;
    debug->dispatchContractReady = 1U;
    debug->marker = 1U;
}

template <typename Stage>
AICORE inline void DeviceDebug::FrontSingleCoreWriteLayoutDebug(const Stage &op)
{
    const auto &front = op.tilingData_->frontReorderTiling;
    const auto &frontDebug = op.tilingData_->frontDebugTiling;
    if (op.coreIdx_ != 0U || frontDebug.frontDebugBytes < sizeof(DispatchFFNCombineFrontLayoutDebug)) {
        return;
    }

    __gm__ DispatchFFNCombineFrontLayoutDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineFrontLayoutDebug *>(op.workspaceGM_ + frontDebug.frontDebugOffset);
    debug->magic = kFrontLayoutDebugMagic;
    debug->workspaceBase = reinterpret_cast<uint64_t>(op.workspaceGM_);
    debug->expandedRowIdxOffset = front.expandedRowIdxOffset;
    debug->frontExpandedExpertOffset = front.frontExpandedExpertOffset;
    debug->frontExpandDstToSrcOffset = front.frontExpandDstToSrcOffset;
    debug->frontSortWs0Offset = front.frontSortWs0Offset;
    debug->frontSortWs1Offset = front.frontSortWs1Offset;
    debug->frontQuantTmpOffset = front.frontQuantTmpOffset;
    debug->frontQuantTmpBytes = front.frontQuantTmpBytes;
    debug->frontDebugOffset = frontDebug.frontDebugOffset;
    debug->frontDebugBytes = frontDebug.frontDebugBytes;
    debug->frontWorkspaceBytes = front.frontWorkspaceBytes;
    debug->peerOffsetA = static_cast<uint64_t>(op.peerMemoryLayout_.offsetA);
    debug->peerOffsetPeerTokenPerExpert = static_cast<uint64_t>(op.peerMemoryLayout_.offsetPeerTokenPerExpert);
    debug->rank = op.rank_;
    debug->rankSize = op.rankSize_;
    debug->coreIdx = op.coreIdx_;
    debug->coreNum = op.coreNum_;
    debug->stageNum = front.stageNum;
    debug->problemM = op.problemM_;
    debug->problemK = op.problemK_;
    debug->topK = op.topK_;
    debug->expertPerRank = op.expertPerRank_;
    debug->expertNum = op.expertNum_;
    debug->expertNumAligned = op.expertNumAligned_;
    debug->frontCase = front.frontCase;
    debug->frontMode = frontDebug.frontMode;
    debug->frontDebugMode = frontDebug.frontDebugMode;
    debug->frontStopStep = frontDebug.frontStopStep;
    debug->frontSortDebugStep = frontDebug.frontSortDebugStep;
    debug->routeElems = op.routeElems_;
    debug->alignedRouteElems = front.alignedRouteElems;
    debug->sortNeedCoreNum = front.sortNeedCoreNum;
    debug->sortPerCoreElems = front.sortPerCoreElems;
    debug->sortLastCoreElems = front.sortLastCoreElems;
    debug->sortPerCoreLoops = front.sortPerCoreLoops;
    debug->sortPerCorePerLoopElems = front.sortPerCorePerLoopElems;
    debug->sortPerCoreLastLoopElems = front.sortPerCoreLastLoopElems;
    debug->sortLastCoreLoops = front.sortLastCoreLoops;
    debug->sortLastCorePerLoopElems = front.sortLastCorePerLoopElems;
    debug->sortLastCoreLastLoopElems = front.sortLastCoreLastLoopElems;
    debug->sortVmsMiddleNeedCoreNum = front.sortVmsMiddleNeedCoreNum;
    debug->sortOutLoopMaxElems = front.sortOutLoopMaxElems;
    debug->activeCopyCores = op.ActiveCopyCores();
    debug->oneCoreVmsActive = 0U;
    debug->middleMergeActive = 0U;
    debug->sortOutSrcWsIndex = 0U;
    debug->noOldNotifyWait = 1U;
    debug->producerClass = kFrontProducerClassSingleCore;
    debug->sortOwnerCore = kFrontSingleCoreSortOwnerCore;
    debug->activeAivNum = op.coreNum_;
    debug->totalLength = op.problemM_ * op.topK_;
    debug->sortLoopMaxElement = op.sortLoopMaxElement_;
    debug->oneCoreCondition = debug->totalLength <= op.sortLoopMaxElement_ ? 1U : 0U;
    debug->routeElemsAlias = op.routeElems_;
    const uint32_t singleSortNum = op.SingleSortNum();
    constexpr uint32_t kFrontSortBufferCount = 4U;
    constexpr uint32_t kFrontSortPayloadAndKey = 2U;
    const uint64_t singleSortRequiredUbBytes =
        static_cast<uint64_t>(singleSortNum) * sizeof(int32_t) * kFrontSortPayloadAndKey * kFrontSortBufferCount;
    debug->tileLength = op.SingleSortTileLength();
    debug->sortNum = singleSortNum;
    debug->singleSortUbFits = singleSortRequiredUbBytes <= AtlasA5::UB_SIZE ? 1U : 0U;
    debug->requiredUbBytesSingleSort = singleSortRequiredUbBytes;
    debug->marker = 1U;
}

template <typename Stage>
AICORE inline void DeviceDebug::FrontFullLoadWriteLayoutDebug(const Stage &op)
{
    if (op.coreIdx_ != 0U || !DeviceDebug::FrontDebugWriteEnabled(op)) {
        return;
    }
    const auto &front = op.tilingData_->frontReorderTiling;
    const auto &frontDebug = op.tilingData_->frontDebugTiling;
    if (frontDebug.frontDebugBytes < sizeof(DispatchFFNCombineFrontLayoutDebug)) {
        return;
    }

    const auto fullLoad = op.BuildFullLoadRouteTiling();
    __gm__ DispatchFFNCombineFrontLayoutDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineFrontLayoutDebug *>(op.workspaceGM_ + frontDebug.frontDebugOffset);
    debug->magic = kFrontLayoutDebugMagic;
    debug->workspaceBase = reinterpret_cast<uint64_t>(op.workspaceGM_);
    debug->expandedRowIdxOffset = front.expandedRowIdxOffset;
    debug->frontExpandedExpertOffset = front.frontExpandedExpertOffset;
    debug->frontExpandDstToSrcOffset = front.frontExpandDstToSrcOffset;
    debug->frontSortWs0Offset = front.frontSortWs0Offset;
    debug->frontSortWs1Offset = front.frontSortWs1Offset;
    debug->frontQuantTmpOffset = front.frontQuantTmpOffset;
    debug->frontQuantTmpBytes = front.frontQuantTmpBytes;
    debug->frontDebugOffset = frontDebug.frontDebugOffset;
    debug->frontDebugBytes = frontDebug.frontDebugBytes;
    debug->frontWorkspaceBytes = front.frontWorkspaceBytes;
    debug->peerOffsetA = static_cast<uint64_t>(op.peerMemoryLayout_.offsetA);
    debug->peerOffsetPeerTokenPerExpert = static_cast<uint64_t>(op.peerMemoryLayout_.offsetPeerTokenPerExpert);
    debug->rank = op.rank_;
    debug->rankSize = op.rankSize_;
    debug->coreIdx = op.coreIdx_;
    debug->coreNum = op.coreNum_;
    debug->stageNum = front.stageNum;
    debug->problemM = op.problemM_;
    debug->problemK = op.problemK_;
    debug->topK = op.topK_;
    debug->expertPerRank = op.expertPerRank_;
    debug->expertNum = op.expertNum_;
    debug->expertNumAligned = op.expertNumAligned_;
    debug->frontCase = front.frontCase;
    debug->frontMode = frontDebug.frontMode;
    debug->frontDebugMode = frontDebug.frontDebugMode;
    debug->frontStopStep = frontDebug.frontStopStep;
    debug->frontSortDebugStep = frontDebug.frontSortDebugStep;
    debug->routeElems = op.routeElems_;
    debug->alignedRouteElems = front.alignedRouteElems;
    debug->sortNeedCoreNum = front.sortNeedCoreNum;
    debug->sortPerCoreElems = front.sortPerCoreElems;
    debug->sortLastCoreElems = front.sortLastCoreElems;
    debug->sortPerCoreLoops = front.sortPerCoreLoops;
    debug->sortPerCorePerLoopElems = front.sortPerCorePerLoopElems;
    debug->sortPerCoreLastLoopElems = front.sortPerCoreLastLoopElems;
    debug->sortLastCoreLoops = front.sortLastCoreLoops;
    debug->sortLastCorePerLoopElems = front.sortLastCorePerLoopElems;
    debug->sortLastCoreLastLoopElems = front.sortLastCoreLastLoopElems;
    debug->sortVmsMiddleNeedCoreNum = front.sortVmsMiddleNeedCoreNum;
    debug->sortOutLoopMaxElems = front.sortOutLoopMaxElems;
    debug->activeCopyCores = op.ActiveCopyCores();
    debug->oneCoreVmsActive = 0U;
    debug->middleMergeActive = 0U;
    debug->sortOutSrcWsIndex = 0U;
    debug->noOldNotifyWait = 1U;
    debug->producerClass = kFrontProducerClassFullLoad;
    debug->sortOwnerCore = 0xFFFFFFFFU;
    debug->activeAivNum = fullLoad.needCoreNum;
    debug->totalLength = op.problemM_ * op.topK_;
    debug->sortLoopMaxElement = op.sortLoopMaxElement_;
    debug->oneCoreCondition = 0U;
    debug->routeElemsAlias = op.routeElems_;
    debug->tileLength = op.FullLoadTileLength();
    debug->sortNum = op.FullLoadSortNum();
    debug->singleSortUbFits = 0U;
    debug->requiredUbBytesSingleSort = 0U;
    const uint64_t fullLoadRequiredUbBytes = op.FullLoadRequiredUbBytes();
    debug->fullLoadRequiredUbBytes = fullLoadRequiredUbBytes;
    debug->fullLoadRemainUbBytes =
        AtlasA5::UB_SIZE > fullLoadRequiredUbBytes ? AtlasA5::UB_SIZE - fullLoadRequiredUbBytes : 0U;
    debug->fullLoadCondition = op.FullLoadCondition();
    debug->fullLoadNeedCoreNum = fullLoad.needCoreNum;
    debug->fullLoadPerCoreRows = fullLoad.perCoreRows;
    debug->fullLoadLastCoreRows = fullLoad.lastCoreRows;
    debug->fullLoadActivateRows = fullLoad.activateRows;
    debug->fullLoadCoreRows = fullLoad.coreRows;
    debug->fullLoadColsScale = FrontPackedRowStride(op.problemK_);
    debug->fullLoadRequiredUbFits = fullLoadRequiredUbBytes <= AtlasA5::UB_SIZE ? 1U : 0U;
    debug->fullLoadSortInputLoaded = 0U;
    debug->fullLoadPayloadArangeTotalLength = 0U;
    debug->fullLoadAllActiveCoresSort = 0U;
    debug->fullLoadSecondSortUsed = 0U;
    debug->fullLoadExpandedRowIdxLocalReady = 0U;
    debug->fullLoadOldSmallPathBusinessUsed = 0U;
    debug->fullLoadCopyOutIdxBlock0Only = 0U;
    debug->fullLoadExpandedRowIdxQueueReEnqueued = 0U;
    debug->fullLoadNonOwnerCopyOutIdxStores = 0U;
    debug->fullLoadCountOwnerCore = 0U;
    debug->fullLoadCopyOutEmptyCores = 0U;
    debug->fullLoadExpertTokensFlag = 0U;
    debug->marker = 1U;
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

template <typename Stage>
AICORE inline void DeviceDebug::FrontMultiCoreWriteLayoutDebug(const Stage &op)
{
    const auto &front = op.tilingData_->frontReorderTiling;
    const auto &frontDebug = op.tilingData_->frontDebugTiling;
    if (op.coreIdx_ != 0U || frontDebug.frontDebugBytes < sizeof(DispatchFFNCombineFrontLayoutDebug)) {
        return;
    }

    __gm__ DispatchFFNCombineFrontLayoutDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineFrontLayoutDebug *>(op.workspaceGM_ + frontDebug.frontDebugOffset);
    debug->magic = kFrontLayoutDebugMagic;
    debug->workspaceBase = reinterpret_cast<uint64_t>(op.workspaceGM_);
    debug->expandedRowIdxOffset = front.expandedRowIdxOffset;
    debug->frontExpandedExpertOffset = front.frontExpandedExpertOffset;
    debug->frontExpandDstToSrcOffset = front.frontExpandDstToSrcOffset;
    debug->frontSortWs0Offset = front.frontSortWs0Offset;
    debug->frontSortWs1Offset = front.frontSortWs1Offset;
    debug->frontQuantTmpOffset = front.frontQuantTmpOffset;
    debug->frontQuantTmpBytes = front.frontQuantTmpBytes;
    debug->frontDebugOffset = frontDebug.frontDebugOffset;
    debug->frontDebugBytes = frontDebug.frontDebugBytes;
    debug->frontWorkspaceBytes = front.frontWorkspaceBytes;
    debug->peerOffsetA = static_cast<uint64_t>(op.peerMemoryLayout_.offsetA);
    debug->peerOffsetPeerTokenPerExpert = static_cast<uint64_t>(op.peerMemoryLayout_.offsetPeerTokenPerExpert);
    debug->rank = op.rank_;
    debug->rankSize = op.rankSize_;
    debug->coreIdx = op.coreIdx_;
    debug->coreNum = op.coreNum_;
    debug->stageNum = op.stageNum_;
    debug->problemM = op.problemM_;
    debug->problemK = op.problemK_;
    debug->topK = op.topK_;
    debug->expertPerRank = op.expertPerRank_;
    debug->expertNum = op.expertNum_;
    debug->expertNumAligned = op.expertNumAligned_;
    debug->frontCase = op.frontCase_;
    debug->frontMode = frontDebug.frontMode;
    debug->frontDebugMode = frontDebug.frontDebugMode;
    debug->frontStopStep = frontDebug.frontStopStep;
    debug->frontSortDebugStep = frontDebug.frontSortDebugStep;
    debug->routeElems = op.routeElems_;
    debug->alignedRouteElems = op.alignedRouteElems_;
    debug->sortNeedCoreNum = op.sortNeedCoreNum_;
    debug->sortPerCoreElems = op.sortPerCoreElems_;
    debug->sortLastCoreElems = op.sortLastCoreElems_;
    debug->sortPerCoreLoops = op.sortPerCoreLoops_;
    debug->sortPerCorePerLoopElems = op.sortPerCorePerLoopElems_;
    debug->sortPerCoreLastLoopElems = op.sortPerCoreLastLoopElems_;
    debug->sortLastCoreLoops = op.sortLastCoreLoops_;
    debug->sortLastCorePerLoopElems = op.sortLastCorePerLoopElems_;
    debug->sortLastCoreLastLoopElems = op.sortLastCoreLastLoopElems_;
    debug->sortVmsMiddleNeedCoreNum = front.sortVmsMiddleNeedCoreNum;
    debug->sortOutLoopMaxElems = op.sortOutLoopMaxElems_;
    debug->activeCopyCores = op.ActiveCopyCores();
    debug->oneCoreVmsActive = op.sortPerCoreLoops_ > 1U ? 1U : 0U;
    debug->middleMergeActive = op.sortNeedCoreNum_ > 4U ? 1U : 0U;
    debug->sortOutSrcWsIndex = op.BuildFinalSortState().srcWsIndex;
    debug->noOldNotifyWait = 1U;
    debug->marker = 1U;
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

template <typename Stage>
AICORE inline bool DeviceDebug::DispatchGatherDebugEnabled(const Stage &op)
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DEVICE_DEBUG
    return op.tilingData_->dispatchTiling.dispatchGatherDebugMode != 0U;
#else
    (void)op;
    return false;
#endif
}

template <typename Stage>
AICORE inline uint32_t DeviceDebug::DispatchGatherStopStep(const Stage &op)
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DEVICE_DEBUG
    return op.tilingData_->dispatchTiling.dispatchGatherStopStep;
#else
    (void)op;
    return 0U;
#endif
}

template <typename Stage>
AICORE inline bool DeviceDebug::DispatchGatherDebugControlEnabled(const Stage &op)
{
    return DeviceDebug::DispatchGatherDebugEnabled(op) || DeviceDebug::DispatchGatherStopStep(op) != 0U;
}

template <typename Stage>
AICORE inline void DeviceDebug::DispatchGatherWriteLayoutDebug(const Stage &op)
{
    const auto &dispatch = op.tilingData_->dispatchTiling;
    if (op.coreIdx_ != 0U || dispatch.dispatchGatherDebugBytes < sizeof(DispatchFFNCombineDispatchGatherLayoutDebug)) {
        return;
    }

    __gm__ DispatchFFNCombineDispatchGatherLayoutDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineDispatchGatherLayoutDebug *>(op.workspaceGM_ +
                                                                               dispatch.dispatchGatherDebugOffset);
    debug->magic = kDispatchGatherLayoutDebugMagic;
    debug->workspaceBase = reinterpret_cast<uint64_t>(op.workspaceGM_);
    debug->gmAOffset = dispatch.gmAOffset;
    debug->perTokenScaleOffset = dispatch.perTokenScaleOffset;
    debug->dispatchGatherScratchOffset = dispatch.dispatchGatherScratchOffset;
    debug->dispatchGatherScratchBytes = dispatch.dispatchGatherScratchBytes;
    debug->dispatchGatherScratchBytesPerAiv = dispatch.dispatchGatherScratchBytesPerAiv;
    debug->dispatchGatherScratchCoreOffset = op.DispatchGatherScratchCoreOffset();
    debug->dispatchGatherDebugOffset = dispatch.dispatchGatherDebugOffset;
    debug->dispatchGatherDebugBytes = dispatch.dispatchGatherDebugBytes;
    debug->dispatchGatherTileBytes = dispatch.dispatchGatherTileBytes;
    debug->cumsumMMOffset = op.tilingData_->frontReorderTiling.cumsumMMOffset;
    debug->preSumBeforeRankOffset = op.tilingData_->frontReorderTiling.preSumBeforeRankOffset;
    debug->peerOffsetA = static_cast<uint64_t>(op.peerMemoryLayout_.offsetA);
    debug->peerOffsetPeerTokenPerExpert = static_cast<uint64_t>(op.peerMemoryLayout_.offsetPeerTokenPerExpert);
    debug->rank = op.rank_;
    debug->rankSize = op.rankSize_;
    debug->coreIdx = op.coreIdx_;
    debug->coreNum = op.coreNum_;
    debug->stageNum = op.stageNum_;
    debug->problemK = op.problemK_;
    debug->maxOutputSize = op.maxOutputSize_;
    debug->packedStride = op.PackedRowStride();
    debug->activeCopyCores = op.ActiveCopyCores();
    debug->dispatchGatherMode = dispatch.dispatchGatherMode;
    debug->dispatchGatherDebugMode = dispatch.dispatchGatherDebugMode;
    debug->dispatchGatherStopStep = dispatch.dispatchGatherStopStep;
    debug->marker = 1U;
    for (uint32_t idx = 0U; idx < 16U; ++idx) {
        debug->reserved1[idx] = 0U;
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::DispatchMetadataDebugOffset(const Stage &op)
{
    return op.tilingData_->dispatchTiling.dispatchDebugOffset + sizeof(DispatchFFNCombineDispatchLayoutDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::DispatchMetadataDebugBytes(const Stage &op)
{
    const uint64_t debugBytes = op.tilingData_->dispatchTiling.dispatchDebugBytes;
    if (debugBytes <= sizeof(DispatchFFNCombineDispatchLayoutDebug)) {
        return 0;
    }
    return debugBytes - sizeof(DispatchFFNCombineDispatchLayoutDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::DispatchTaskStatsDebugOffset(const Stage &op)
{
    const uint64_t metadataBytes =
        static_cast<uint64_t>(op.expertPerRank_) * op.rankSize_ * sizeof(DispatchFFNCombineDispatchMetadataDebug);
    return DeviceDebug::DispatchMetadataDebugOffset(op) + metadataBytes;
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::DispatchTaskStatsDebugBytes(const Stage &op)
{
    const uint64_t taskOffset = DeviceDebug::DispatchTaskStatsDebugOffset(op);
    const uint64_t debugEnd =
        op.tilingData_->dispatchTiling.dispatchDebugOffset + op.tilingData_->dispatchTiling.dispatchDebugBytes;
    if (debugEnd <= taskOffset) {
        return 0;
    }
    return debugEnd - taskOffset;
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::DispatchGroupDoneDebugOffset(const Stage &op)
{
    const uint64_t taskBytes =
        static_cast<uint64_t>(op.expertPerRank_) * op.coreNum_ * sizeof(DispatchFFNCombineDispatchTaskStatsDebug);
    return DeviceDebug::DispatchTaskStatsDebugOffset(op) + taskBytes;
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::DispatchGroupDoneDebugBytes(const Stage &op)
{
    const uint64_t groupDoneOffset = DeviceDebug::DispatchGroupDoneDebugOffset(op);
    const uint64_t debugEnd =
        op.tilingData_->dispatchTiling.dispatchDebugOffset + op.tilingData_->dispatchTiling.dispatchDebugBytes;
    if (debugEnd <= groupDoneOffset) {
        return 0;
    }
    return debugEnd - groupDoneOffset;
}

template <typename Stage>
AICORE inline bool DeviceDebug::DispatchDebugEnabled(const Stage &op)
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DEVICE_DEBUG
    return op.tilingData_->dispatchTiling.dispatchDebugMode != 0U;
#else
    (void)op;
    return false;
#endif
}

template <typename Stage>
AICORE inline void DeviceDebug::DispatchWriteLayoutDebugHeader(const Stage &op)
{
    if (op.coreIdx_ != 0 ||
        op.tilingData_->dispatchTiling.dispatchDebugBytes < sizeof(DispatchFFNCombineDispatchLayoutDebug)) {
        return;
    }

    __gm__ DispatchFFNCombineDispatchLayoutDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineDispatchLayoutDebug *>(
            op.workspaceGM_ + op.tilingData_->dispatchTiling.dispatchDebugOffset);
    debug->magic = kDispatchLayoutDebugMagic;
    debug->workspaceBase = reinterpret_cast<uint64_t>(op.workspaceGM_);
    debug->gmAOffset = op.tilingData_->dispatchTiling.gmAOffset;
    debug->perTokenScaleOffset = op.tilingData_->dispatchTiling.perTokenScaleOffset;
    debug->dispatchScratchOffset = op.tilingData_->dispatchTiling.dispatchScratchOffset;
    debug->dispatchScratchBytes = op.tilingData_->dispatchTiling.dispatchScratchBytes;
    debug->dispatchScratchBytesPerAiv = op.tilingData_->dispatchTiling.dispatchScratchBytesPerAiv;
    debug->dispatchScratchCoreOffset = op.DispatchScratchCoreOffset();
    debug->dispatchDebugOffset = op.tilingData_->dispatchTiling.dispatchDebugOffset;
    debug->dispatchDebugBytes = op.tilingData_->dispatchTiling.dispatchDebugBytes;
    debug->dispatchTileBytes = op.tilingData_->dispatchTiling.dispatchTileBytes;
    debug->cumsumMMOffset = op.tilingData_->frontReorderTiling.cumsumMMOffset;
    debug->preSumBeforeRankOffset = op.tilingData_->frontReorderTiling.preSumBeforeRankOffset;
    debug->frontWorkspaceBytes = op.tilingData_->frontReorderTiling.frontWorkspaceBytes;
    debug->peerOffsetA = static_cast<uint64_t>(op.peerMemoryLayout_.offsetA);
    debug->peerOffsetPeerTokenPerExpert = static_cast<uint64_t>(op.peerMemoryLayout_.offsetPeerTokenPerExpert);
    debug->rank = op.rank_;
    debug->rankSize = op.rankSize_;
    debug->coreIdx = op.coreIdx_;
    debug->coreNum = op.coreNum_;
    debug->stageNum = op.stageNum_;
    debug->problemK = op.problemK_;
    debug->maxOutputSize = op.maxOutputSize_;
    debug->packedStride = op.PackedRowStride();
    pipe_barrier(PIPE_ALL);
}

template <typename Stage>
AICORE inline void DeviceDebug::DispatchWriteGroupMetadataDebug(const Stage &op)
{
    const uint64_t metadataCount = static_cast<uint64_t>(op.expertPerRank_) * op.rankSize_;
    const uint64_t requiredBytes = metadataCount * sizeof(DispatchFFNCombineDispatchMetadataDebug);
    if (op.coreIdx_ != 0 || DeviceDebug::DispatchMetadataDebugBytes(op) < requiredBytes) {
        return;
    }

    __gm__ DispatchFFNCombineDispatchMetadataDebug *metadata =
        reinterpret_cast<__gm__ DispatchFFNCombineDispatchMetadataDebug *>(
            op.workspaceGM_ + DeviceDebug::DispatchMetadataDebugOffset(op));

    uint32_t groupBase = 0;
    for (uint32_t groupIdx = 0; groupIdx < op.expertPerRank_; ++groupIdx) {
        const uint32_t currentM = op.CurrentM(groupIdx);
        for (uint32_t srcRank = 0; srcRank < op.rankSize_; ++srcRank) {
            const uint64_t metaIdx = static_cast<uint64_t>(groupIdx) * op.rankSize_ + srcRank;
            const uint32_t rawRows = op.RawRowsForLocalGroup(srcRank, groupIdx);
            const uint32_t srcRowBase = op.SourceRowBase(srcRank, groupIdx);
            const uint32_t cumsumBeforeSrc = op.CumsumBeforeSource(srcRank, groupIdx);
            const uint32_t dstRowBase = groupBase + cumsumBeforeSrc;
            uint32_t rows = rawRows;
            if (dstRowBase >= op.maxOutputSize_) {
                rows = 0;
            } else if (dstRowBase + rows > op.maxOutputSize_) {
                rows = op.maxOutputSize_ - dstRowBase;
            }

            metadata[metaIdx].groupIdx = groupIdx;
            metadata[metaIdx].srcRank = srcRank;
            metadata[metaIdx].groupBase = groupBase;
            metadata[metaIdx].currentM = currentM;
            metadata[metaIdx].rawRows = rawRows;
            metadata[metaIdx].rows = rows;
            metadata[metaIdx].srcRowBase = srcRowBase;
            metadata[metaIdx].dstRowBase = dstRowBase;
        }
        groupBase += currentM;
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

template <typename Stage>
AICORE inline void DeviceDebug::DispatchWriteTaskSplitDebug(const Stage &op)
{
    const uint64_t statsCount = static_cast<uint64_t>(op.expertPerRank_) * op.coreNum_;
    const uint64_t requiredBytes = statsCount * sizeof(DispatchFFNCombineDispatchTaskStatsDebug);
    if (op.coreIdx_ != 0 || DeviceDebug::DispatchTaskStatsDebugBytes(op) < requiredBytes) {
        return;
    }

    __gm__ DispatchFFNCombineDispatchTaskStatsDebug *stats =
        reinterpret_cast<__gm__ DispatchFFNCombineDispatchTaskStatsDebug *>(
            op.workspaceGM_ + DeviceDebug::DispatchTaskStatsDebugOffset(op));

    uint32_t groupBase = 0;
    for (uint32_t groupIdx = 0; groupIdx < op.expertPerRank_; ++groupIdx) {
        const uint32_t currentM = op.CurrentM(groupIdx);
        const uint32_t rowBlockRows = op.DecideDispatchRowBlockRows(groupIdx, groupBase, currentM);
        const uint32_t maxBlocksPerGroup = op.DispatchMaxBlocksPerGroup();
        for (uint32_t core = 0; core < op.coreNum_; ++core) {
            uint32_t assignedBlockCount = 0;
            uint32_t coreRowCount = 0;
            uint32_t firstSrcRank = kDispatchInvalidTask;
            uint32_t firstRows = 0;
            uint32_t firstSrcRowBase = 0;
            uint32_t firstDstRowBase = 0;
            uint32_t lastSrcRank = kDispatchInvalidTask;
            uint32_t lastRows = 0;
            uint32_t lastSrcRowBase = 0;
            uint32_t lastDstRowBase = 0;

            uint32_t linearBlock = 0U;
            for (uint32_t srcRank = 0; srcRank < op.rankSize_; ++srcRank) {
                const uint32_t rows = op.SourceRows(srcRank, groupIdx, groupBase);
                const uint32_t srcRowBase = op.SourceRowBase(srcRank, groupIdx);
                const uint32_t dstRowBase = op.DstRowBase(srcRank, groupIdx, groupBase);
                for (uint32_t blockRow = 0; blockRow < rows; blockRow += rowBlockRows) {
                    const uint32_t blockRows = (rows - blockRow > rowBlockRows) ? rowBlockRows : (rows - blockRow);
                    if (linearBlock % op.coreNum_ == core) {
                        if (firstSrcRank == kDispatchInvalidTask) {
                            firstSrcRank = srcRank;
                            firstRows = blockRows;
                            firstSrcRowBase = srcRowBase + blockRow;
                            firstDstRowBase = dstRowBase + blockRow;
                        }
                        lastSrcRank = srcRank;
                        lastRows = blockRows;
                        lastSrcRowBase = srcRowBase + blockRow;
                        lastDstRowBase = dstRowBase + blockRow;
                        ++assignedBlockCount;
                        coreRowCount += blockRows;
                    }
                    ++linearBlock;
                }
            }

            const uint64_t statIdx = static_cast<uint64_t>(groupIdx) * op.coreNum_ + core;
            stats[statIdx].groupIdx = groupIdx;
            stats[statIdx].coreIdx = core;
            stats[statIdx].rankSize = op.rankSize_;
            stats[statIdx].assignedBlockCount = assignedBlockCount;
            stats[statIdx].coreRowCount = coreRowCount;
            stats[statIdx].groupBase = groupBase;
            stats[statIdx].currentM = currentM;
            stats[statIdx].firstSrcRank = firstSrcRank;
            stats[statIdx].firstRows = firstRows;
            stats[statIdx].firstSrcRowBase = firstSrcRowBase;
            stats[statIdx].firstDstRowBase = firstDstRowBase;
            stats[statIdx].lastSrcRank = lastSrcRank;
            stats[statIdx].lastRows = lastRows;
            stats[statIdx].lastSrcRowBase = lastSrcRowBase;
            stats[statIdx].lastDstRowBase = lastDstRowBase;
            stats[statIdx].rowBlockRows = rowBlockRows;
            stats[statIdx].maxBlocksPerGroup = maxBlocksPerGroup;
        }
        groupBase += currentM;
    }
    pipe_barrier(PIPE_ALL);
}

template <typename Stage>
AICORE inline void DeviceDebug::DispatchNotifyGmm1GroupReady(const Stage &op, uint32_t groupIdx, uint32_t groupBase,
                                                             uint32_t currentM)
{
    const uint64_t requiredBytes =
        static_cast<uint64_t>(op.expertPerRank_) * sizeof(DispatchFFNCombineDispatchGroupDoneDebug);
    if (op.coreIdx_ != 0 || DeviceDebug::DispatchGroupDoneDebugBytes(op) < requiredBytes) {
        return;
    }

    __gm__ DispatchFFNCombineDispatchGroupDoneDebug *groupDone =
        reinterpret_cast<__gm__ DispatchFFNCombineDispatchGroupDoneDebug *>(
            op.workspaceGM_ + DeviceDebug::DispatchGroupDoneDebugOffset(op));
    groupDone[groupIdx].magic = kDispatchGroupDoneDebugMagic;
    groupDone[groupIdx].groupIdx = groupIdx;
    groupDone[groupIdx].rank = op.rank_;
    groupDone[groupIdx].rankSize = op.rankSize_;
    groupDone[groupIdx].groupBase = groupBase;
    groupDone[groupIdx].currentM = currentM;
    groupDone[groupIdx].marker = 1U;
    groupDone[groupIdx].reserved0 = 0U;
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}
#endif

#if defined(__DAV_CUBE__)
template <typename Stage>
AICORE inline bool DeviceDebug::Gmm1DebugEnabled(const Stage &op)
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DEVICE_DEBUG
    return op.tilingData_->gmm1Tiling.gmm1DebugMode != 0U;
#else
    (void)op;
    return false;
#endif
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm1SyncDebugOffset(const Stage &op)
{
    return op.tilingData_->gmm1Tiling.gmm1DebugOffset + sizeof(DispatchFFNCombineGmm1LayoutDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm1SyncDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.expertPerRank_) * op.coreNum_ * sizeof(DispatchFFNCombineGmm1SyncDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm1SyncDebugBytes(const Stage &op)
{
    const uint64_t debugBytes = op.tilingData_->gmm1Tiling.gmm1DebugBytes;
    if (debugBytes <= sizeof(DispatchFFNCombineGmm1LayoutDebug)) {
        return 0;
    }
    return debugBytes - sizeof(DispatchFFNCombineGmm1LayoutDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm1TaskDebugOffset(const Stage &op)
{
    return DeviceDebug::Gmm1SyncDebugOffset(op) + DeviceDebug::Gmm1SyncDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm1TaskDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.expertPerRank_) * op.coreNum_ * sizeof(DispatchFFNCombineGmm1TaskDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm1TaskDebugBytes(const Stage &op)
{
    const uint64_t debugEnd = op.tilingData_->gmm1Tiling.gmm1DebugOffset + op.tilingData_->gmm1Tiling.gmm1DebugBytes;
    const uint64_t taskOffset = DeviceDebug::Gmm1TaskDebugOffset(op);
    if (debugEnd <= taskOffset) {
        return 0;
    }
    return debugEnd - taskOffset;
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm1DoneDebugOffset(const Stage &op)
{
    return DeviceDebug::Gmm1TaskDebugOffset(op) + DeviceDebug::Gmm1TaskDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm1DoneDebugBytes(const Stage &op)
{
    const uint64_t debugEnd = op.tilingData_->gmm1Tiling.gmm1DebugOffset + op.tilingData_->gmm1Tiling.gmm1DebugBytes;
    const uint64_t doneOffset = DeviceDebug::Gmm1DoneDebugOffset(op);
    if (debugEnd <= doneOffset) {
        return 0;
    }
    return debugEnd - doneOffset;
}

template <typename Stage>
AICORE inline void DeviceDebug::Gmm1WriteLayoutDebugHeader(const Stage &op)
{
    if (op.coreIdx_ != 0 || op.tilingData_->gmm1Tiling.gmm1DebugBytes < sizeof(DispatchFFNCombineGmm1LayoutDebug)) {
        return;
    }

    __gm__ DispatchFFNCombineGmm1LayoutDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineGmm1LayoutDebug *>(
        op.workspaceGM_ + op.tilingData_->gmm1Tiling.gmm1DebugOffset);
    debug->magic = kGmm1LayoutDebugMagic;
    debug->workspaceBase = reinterpret_cast<uint64_t>(op.workspaceGM_);
    debug->gmAOffset = op.tilingData_->dispatchTiling.gmAOffset;
    debug->gmCOffset = op.tilingData_->gmm1Tiling.gmCOffset;
    debug->perTokenScaleOffset = op.tilingData_->dispatchTiling.perTokenScaleOffset;
    debug->cumsumMMOffset = op.tilingData_->frontReorderTiling.cumsumMMOffset;
    debug->expertTokenNumsBase = reinterpret_cast<uint64_t>(op.expertTokenNumsPtr_);
    debug->weight1Base = reinterpret_cast<uint64_t>(op.weight1Ptr_);
    debug->scale1Base = reinterpret_cast<uint64_t>(op.scale1Ptr_);
    debug->gmm1DebugOffset = op.tilingData_->gmm1Tiling.gmm1DebugOffset;
    debug->gmm1DebugBytes = op.tilingData_->gmm1Tiling.gmm1DebugBytes;
    debug->rank = op.rank_;
    debug->rankSize = op.rankSize_;
    debug->coreIdx = op.coreIdx_;
    debug->coreNum = op.coreNum_;
    debug->stageNum = op.stageNum_;
    debug->problemK = op.problemK_;
    debug->problemN = op.problemN_;
    debug->maxOutputSize = op.maxOutputSize_;
    debug->expertPerRank = op.expertPerRank_;
    debug->l1TileM = op.tilingData_->gmm1Tiling.l1TileM;
    debug->l1TileN = op.tilingData_->gmm1Tiling.l1TileN;
    debug->l1TileK = op.tilingData_->gmm1Tiling.l1TileK;
    debug->l0TileM = op.tilingData_->gmm1Tiling.l0TileM;
    debug->l0TileN = op.tilingData_->gmm1Tiling.l0TileN;
    debug->l0TileK = op.tilingData_->gmm1Tiling.l0TileK;
    debug->debugMode = op.tilingData_->gmm1Tiling.gmm1DebugMode;
    pipe_barrier(PIPE_ALL);
}

template <typename Stage>
AICORE inline void DeviceDebug::Gmm1WriteSyncDebug(const Stage &op, uint32_t groupIdx, uint32_t groupBase,
                                                   uint32_t currentM)
{
    const uint64_t requiredBytes =
        static_cast<uint64_t>(op.expertPerRank_) * op.coreNum_ * sizeof(DispatchFFNCombineGmm1SyncDebug);
    if (DeviceDebug::Gmm1SyncDebugBytes(op) < requiredBytes) {
        return;
    }
    __gm__ DispatchFFNCombineGmm1SyncDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineGmm1SyncDebug *>(
        op.workspaceGM_ + DeviceDebug::Gmm1SyncDebugOffset(op));
    const uint64_t debugIdx = static_cast<uint64_t>(groupIdx) * op.coreNum_ + op.coreIdx_;
    volatile __gm__ DispatchFFNCombineGmm1SyncDebug *entry = debug + debugIdx;
    entry->magic = kGmm1SyncDebugMagic;
    entry->groupIdx = groupIdx;
    entry->coreIdx = op.coreIdx_;
    entry->flagId = op.DispatchV2CFlagId(groupIdx);
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->groupBase = groupBase;
    entry->currentM = currentM;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + debugIdx), sizeof(DispatchFFNCombineGmm1SyncDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::Gmm1WriteTaskDebug(const Stage &op, uint32_t groupIdx, uint32_t groupBase,
                                                   uint32_t currentMRaw, uint32_t currentM, uint32_t expertTokenNums,
                                                   uint32_t startCoreIdx, uint32_t coreLoops)
{
    const uint64_t requiredBytes =
        static_cast<uint64_t>(op.expertPerRank_) * op.coreNum_ * sizeof(DispatchFFNCombineGmm1TaskDebug);
    if (DeviceDebug::Gmm1TaskDebugBytes(op) < requiredBytes) {
        return;
    }
    __gm__ DispatchFFNCombineGmm1TaskDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineGmm1TaskDebug *>(
        op.workspaceGM_ + DeviceDebug::Gmm1TaskDebugOffset(op));
    const uint64_t debugIdx = static_cast<uint64_t>(groupIdx) * op.coreNum_ + op.coreIdx_;
    volatile __gm__ DispatchFFNCombineGmm1TaskDebug *entry = debug + debugIdx;

    const uint32_t tileM = op.TileM(currentM);
    const uint32_t tileN = op.TileN();
    const uint32_t startLoopIdx = op.StartLoopIdx(startCoreIdx);
    const uint32_t assignedTileCount = op.AssignedTileCount(startLoopIdx, coreLoops);
    uint32_t firstLoop = kGmm1InvalidTask;
    uint32_t lastLoop = kGmm1InvalidTask;
    uint32_t firstBlockM = kGmm1InvalidTask;
    uint32_t firstBlockN = kGmm1InvalidTask;
    uint32_t lastBlockM = kGmm1InvalidTask;
    uint32_t lastBlockN = kGmm1InvalidTask;
    uint32_t firstActualM = 0;
    uint32_t firstActualN = 0;
    uint32_t lastActualM = 0;
    uint32_t lastActualN = 0;

    if (assignedTileCount > 0U) {
        firstLoop = startLoopIdx;
        lastLoop = startLoopIdx + (assignedTileCount - 1U) * op.coreNum_;
        op.GetBlockCoordMN(firstLoop, tileM, tileN, firstBlockM, firstBlockN);
        op.GetBlockCoordMN(lastLoop, tileM, tileN, lastBlockM, lastBlockN);
        op.GetActualBlockShapeMN(firstBlockM, firstBlockN, tileM, tileN, currentM, firstActualM, firstActualN);
        op.GetActualBlockShapeMN(lastBlockM, lastBlockN, tileM, tileN, currentM, lastActualM, lastActualN);
    }

    entry->magic = kGmm1TaskDebugMagic;
    entry->groupIdx = groupIdx;
    entry->coreIdx = op.coreIdx_;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->groupBase = groupBase;
    entry->currentMRaw = currentMRaw;
    entry->currentM = currentM;
    entry->expertTokenNums = expertTokenNums;
    entry->tileM = tileM;
    entry->tileN = tileN;
    entry->coreLoops = coreLoops;
    entry->startCoreIdx = startCoreIdx;
    entry->startLoopIdx = startLoopIdx;
    entry->assignedTileCount = assignedTileCount;
    entry->firstLoop = firstLoop;
    entry->lastLoop = lastLoop;
    entry->firstBlockM = firstBlockM;
    entry->firstBlockN = firstBlockN;
    entry->lastBlockM = lastBlockM;
    entry->lastBlockN = lastBlockN;
    entry->firstActualM = firstActualM;
    entry->firstActualN = firstActualN;
    entry->lastActualM = lastActualM;
    entry->lastActualN = lastActualN;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + debugIdx), sizeof(DispatchFFNCombineGmm1TaskDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::Gmm1WriteDoneDebug(const Stage &op, uint32_t segmentIdx, uint32_t segmentStartExpert,
                                                   uint32_t segmentEndExpert, uint32_t segmentRowBase,
                                                   uint32_t segmentRows)
{
    const uint64_t requiredBytes =
        static_cast<uint64_t>(op.SwigluSegmentNum()) * op.coreNum_ * sizeof(DispatchFFNCombineGmm1DoneDebug);
    if (DeviceDebug::Gmm1DoneDebugBytes(op) < requiredBytes) {
        return;
    }
    __gm__ DispatchFFNCombineGmm1DoneDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineGmm1DoneDebug *>(
        op.workspaceGM_ + DeviceDebug::Gmm1DoneDebugOffset(op));
    const uint64_t debugIdx = static_cast<uint64_t>(segmentIdx) * op.coreNum_ + op.coreIdx_;
    volatile __gm__ DispatchFFNCombineGmm1DoneDebug *entry = debug + debugIdx;
    entry->magic = kGmm1DoneDebugMagic;
    entry->segmentIdx = segmentIdx;
    entry->coreIdx = op.coreIdx_;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->segmentStartExpert = segmentStartExpert;
    entry->segmentEndExpert = segmentEndExpert;
    entry->segmentRowBase = segmentRowBase;
    entry->segmentRows = segmentRows;
    entry->c2vFlagId = op.C2VFlagId(segmentIdx);
    entry->c2vEnabled = op.stageNum_ >= 11U ? 1U : 0U;
    entry->marker = 1U;
    entry->segmentPolicy = kGmm1SegmentPolicyReferenceEpilogue;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + debugIdx), sizeof(DispatchFFNCombineGmm1DoneDebug));
}
#endif

#if defined(__DAV_VEC__)
template <typename Stage>
AICORE inline bool DeviceDebug::SwigluDebugEnabled(const Stage &op)
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DEVICE_DEBUG
    return op.tilingData_->swigluTiling.swigluDebugMode != 0U;
#else
    (void)op;
    return false;
#endif
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::SwigluC2VDebugOffset(const Stage &op)
{
    return op.tilingData_->swigluTiling.swigluDebugOffset + sizeof(DispatchFFNCombineSwigluLayoutDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::SwigluC2VDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.SwigluSegmentNum()) * op.coreNum_ * sizeof(DispatchFFNCombineSwigluC2VDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::SwigluSegmentDebugOffset(const Stage &op)
{
    return DeviceDebug::SwigluC2VDebugOffset(op) + DeviceDebug::SwigluC2VDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::SwigluSegmentDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.SwigluSegmentNum()) * sizeof(DispatchFFNCombineSwigluSegmentDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::SwigluTaskDebugOffset(const Stage &op)
{
    return DeviceDebug::SwigluSegmentDebugOffset(op) + DeviceDebug::SwigluSegmentDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::SwigluTaskDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.SwigluSegmentNum()) * op.coreNum_ * sizeof(DispatchFFNCombineSwigluTaskDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::SwigluDoneDebugOffset(const Stage &op)
{
    return DeviceDebug::SwigluTaskDebugOffset(op) + DeviceDebug::SwigluTaskDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::SwigluDoneDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.SwigluSegmentNum()) * op.coreNum_ * sizeof(DispatchFFNCombineSwigluDoneDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::SwigluFinalSyncDebugOffset(const Stage &op)
{
    return DeviceDebug::SwigluDoneDebugOffset(op) + DeviceDebug::SwigluDoneDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::SwigluFinalSyncDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.coreNum_) * sizeof(DispatchFFNCombineSwigluFinalSyncDebug);
}

template <typename Stage>
AICORE inline void DeviceDebug::SwigluWriteLayoutDebugHeader(const Stage &op)
{
    if (op.coreIdx_ != 0 ||
        op.tilingData_->swigluTiling.swigluDebugBytes < sizeof(DispatchFFNCombineSwigluLayoutDebug)) {
        return;
    }

    __gm__ DispatchFFNCombineSwigluLayoutDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineSwigluLayoutDebug *>(
        op.workspaceGM_ + op.tilingData_->swigluTiling.swigluDebugOffset);
    debug->magic = kSwigluLayoutDebugMagic;
    debug->workspaceBase = reinterpret_cast<uint64_t>(op.workspaceGM_);
    debug->gmCOffset = op.tilingData_->gmm1Tiling.gmCOffset;
    debug->perTokenScaleOffset = op.tilingData_->dispatchTiling.perTokenScaleOffset;
    debug->gmPermutedTokenOffset = op.tilingData_->swigluTiling.gmPermutedTokenOffset;
    debug->perTokenScale2Offset = op.tilingData_->swigluTiling.perTokenScale2Offset;
    debug->cumsumMMOffset = op.tilingData_->frontReorderTiling.cumsumMMOffset;
    debug->expertTokenNumsBase = reinterpret_cast<uint64_t>(op.expertTokenNumsPtr_);
    debug->swigluDebugOffset = op.tilingData_->swigluTiling.swigluDebugOffset;
    debug->swigluDebugBytes = op.tilingData_->swigluTiling.swigluDebugBytes;
    debug->swigluSegmentMetaOffset = op.tilingData_->swigluTiling.swigluSegmentMetaOffset;
    debug->swigluSegmentMetaBytes = op.tilingData_->swigluTiling.swigluSegmentMetaBytes;
    debug->gmCBytes = static_cast<uint64_t>(op.maxOutputSize_) * op.problemN_ * sizeof(half);
    debug->gmPermutedTokenBytes = static_cast<uint64_t>(op.maxOutputSize_) * op.outputN_ * sizeof(int8_t);
    debug->perTokenScaleBytes = static_cast<uint64_t>(op.maxOutputSize_) * sizeof(float);
    debug->perTokenScale2Bytes = static_cast<uint64_t>(op.maxOutputSize_) * sizeof(float);
    debug->frontWorkspaceBytes = op.tilingData_->frontReorderTiling.frontWorkspaceBytes;
    debug->dispatchScratchOffset = op.tilingData_->dispatchTiling.dispatchScratchOffset;
    debug->gmm1DebugOffset = op.tilingData_->gmm1Tiling.gmm1DebugOffset;
    debug->rank = op.rank_;
    debug->rankSize = op.rankSize_;
    debug->coreIdx = op.coreIdx_;
    debug->coreNum = op.coreNum_;
    debug->stageNum = op.stageNum_;
    debug->problemN = op.problemN_;
    debug->outputN = op.outputN_;
    debug->maxOutputSize = op.maxOutputSize_;
    debug->expertPerRank = op.expertPerRank_;
    debug->segmentNum = op.tilingData_->swigluTiling.swigluSegmentNum;
    debug->epilogueGranularity = op.tilingData_->swigluTiling.swigluEpilogueGranularity;
    debug->tileElems = op.tilingData_->swigluTiling.swigluTileElems;
    debug->ubStages = op.tilingData_->swigluTiling.swigluUbStages;
    debug->debugMode = op.tilingData_->swigluTiling.swigluDebugMode;
    debug->pipelineMode = kSwigluPipelineModeInputOutputSplit;
    debug->scale2BufferNum = kSwigluScaleChunkBuffers;
    debug->metadataMode = op.tilingData_->swigluTiling.swigluMetadataMode;
    debug->gmCRowBytes = op.problemN_ * sizeof(half);
    debug->gmPermutedTokenRowBytes = op.outputN_ * sizeof(int8_t);
    debug->perTokenScaleBytesPerRow = sizeof(float);
    debug->perTokenScale2BytesPerRow = sizeof(float);
    debug->layoutVersion = kSwigluLayoutVersion;
    debug->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug), sizeof(DispatchFFNCombineSwigluLayoutDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::SwigluWriteC2VDebug(const Stage &op, uint32_t segmentIdx)
{
    const uint64_t requiredBytes = DeviceDebug::SwigluC2VDebugOffset(op) -
                                   op.tilingData_->swigluTiling.swigluDebugOffset +
                                   DeviceDebug::SwigluC2VDebugTableBytes(op);
    if (op.tilingData_->swigluTiling.swigluDebugBytes < requiredBytes) {
        return;
    }

    __gm__ DispatchFFNCombineSwigluC2VDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineSwigluC2VDebug *>(
        op.workspaceGM_ + DeviceDebug::SwigluC2VDebugOffset(op));
    const uint64_t debugIdx = static_cast<uint64_t>(segmentIdx) * op.coreNum_ + op.coreIdx_;
    volatile __gm__ DispatchFFNCombineSwigluC2VDebug *entry = debug + debugIdx;
    entry->magic = kSwigluC2VDebugMagic;
    entry->segmentIdx = segmentIdx;
    entry->coreIdx = op.coreIdx_;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->stageNum = op.stageNum_;
    entry->waitSource = kSwigluWaitSourceC2VOnly;
    entry->c2vFlagId = op.C2VFlagId(segmentIdx);
    entry->c2vEnabled = 1U;
    entry->gmm1DoneConsumed = 0U;
    entry->aivSyncAfterWait = 1U;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + debugIdx), sizeof(DispatchFFNCombineSwigluC2VDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::SwigluWriteSegmentDebug(const Stage &op, uint32_t segmentIdx,
                                                        uint32_t segmentStartExpert, uint32_t segmentEndExpert,
                                                        uint32_t segmentRowBase, uint32_t segmentRows,
                                                        uint32_t cumsumRows, uint32_t expertTokenRows)
{
    const uint64_t requiredBytes = DeviceDebug::SwigluSegmentDebugOffset(op) -
                                   op.tilingData_->swigluTiling.swigluDebugOffset +
                                   DeviceDebug::SwigluSegmentDebugTableBytes(op);
    if (op.coreIdx_ != 0 || op.tilingData_->swigluTiling.swigluDebugBytes < requiredBytes) {
        return;
    }

    __gm__ DispatchFFNCombineSwigluSegmentDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineSwigluSegmentDebug *>(op.workspaceGM_ +
                                                                        DeviceDebug::SwigluSegmentDebugOffset(op));
    volatile __gm__ DispatchFFNCombineSwigluSegmentDebug *entry = debug + segmentIdx;
    entry->magic = kSwigluSegmentDebugMagic;
    entry->segmentIdx = segmentIdx;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->segmentStartExpert = segmentStartExpert;
    entry->segmentEndExpert = segmentEndExpert;
    entry->segmentRowBase = segmentRowBase;
    entry->segmentRows = segmentRows;
    entry->cumsumRows = cumsumRows;
    entry->expertTokenRows = expertTokenRows;
    entry->maxOutputSize = op.maxOutputSize_;
    entry->segmentNum = op.SwigluSegmentNum();
    entry->epilogueGranularity = op.SwigluEpilogueGranularity();
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + segmentIdx), sizeof(DispatchFFNCombineSwigluSegmentDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::SwigluWriteTaskDebug(const Stage &op, uint32_t segmentIdx, uint32_t segmentStartExpert,
                                                     uint32_t segmentEndExpert, uint32_t segmentRowBase,
                                                     uint32_t segmentRows, uint32_t localRowStart, uint32_t localRows,
                                                     uint32_t rowSplitBase, uint32_t rowSplitRem)
{
    const uint64_t requiredBytes = DeviceDebug::SwigluTaskDebugOffset(op) -
                                   op.tilingData_->swigluTiling.swigluDebugOffset +
                                   DeviceDebug::SwigluTaskDebugTableBytes(op);
    if (op.tilingData_->swigluTiling.swigluDebugBytes < requiredBytes) {
        return;
    }

    __gm__ DispatchFFNCombineSwigluTaskDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineSwigluTaskDebug *>(
        op.workspaceGM_ + DeviceDebug::SwigluTaskDebugOffset(op));
    const uint64_t debugIdx = static_cast<uint64_t>(segmentIdx) * op.coreNum_ + op.coreIdx_;
    volatile __gm__ DispatchFFNCombineSwigluTaskDebug *entry = debug + debugIdx;
    entry->magic = kSwigluTaskDebugMagic;
    entry->segmentIdx = segmentIdx;
    entry->coreIdx = op.coreIdx_;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->segmentStartExpert = segmentStartExpert;
    entry->segmentEndExpert = segmentEndExpert;
    entry->segmentRowBase = segmentRowBase;
    entry->segmentRows = segmentRows;
    entry->localRowStart = localRowStart;
    entry->localRows = localRows;
    entry->rowSplitBase = rowSplitBase;
    entry->rowSplitRem = rowSplitRem;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + debugIdx), sizeof(DispatchFFNCombineSwigluTaskDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::SwigluWriteDoneDebug(const Stage &op, uint32_t segmentIdx, uint32_t segmentRowBase,
                                                     uint32_t segmentRows, uint32_t localRowStart, uint32_t localRows)
{
    const uint64_t requiredBytes = DeviceDebug::SwigluDoneDebugOffset(op) -
                                   op.tilingData_->swigluTiling.swigluDebugOffset +
                                   DeviceDebug::SwigluDoneDebugTableBytes(op);
    if (op.tilingData_->swigluTiling.swigluDebugBytes < requiredBytes) {
        return;
    }

    __gm__ DispatchFFNCombineSwigluDoneDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineSwigluDoneDebug *>(
        op.workspaceGM_ + DeviceDebug::SwigluDoneDebugOffset(op));
    const uint64_t debugIdx = static_cast<uint64_t>(segmentIdx) * op.coreNum_ + op.coreIdx_;
    volatile __gm__ DispatchFFNCombineSwigluDoneDebug *entry = debug + debugIdx;
    entry->magic = kSwigluDoneDebugMagic;
    entry->segmentIdx = segmentIdx;
    entry->coreIdx = op.coreIdx_;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->stageNum = op.stageNum_;
    entry->segmentRowBase = segmentRowBase;
    entry->segmentRows = segmentRows;
    entry->localRowStart = localRowStart;
    entry->localRows = localRows;
    entry->aivSyncBeforeDone = 1U;
    entry->v2cFlagId = op.V2CFlagId(segmentIdx);
    entry->v2cEnabled = op.stageNum_ >= 12U ? 1U : 0U;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + debugIdx), sizeof(DispatchFFNCombineSwigluDoneDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::SwigluWriteFinalSyncDebug(const Stage &op)
{
    const uint64_t requiredBytes = DeviceDebug::SwigluFinalSyncDebugOffset(op) -
                                   op.tilingData_->swigluTiling.swigluDebugOffset +
                                   DeviceDebug::SwigluFinalSyncDebugTableBytes(op);
    if (op.tilingData_->swigluTiling.swigluDebugBytes < requiredBytes) {
        return;
    }

    __gm__ DispatchFFNCombineSwigluFinalSyncDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineSwigluFinalSyncDebug *>(op.workspaceGM_ +
                                                                          DeviceDebug::SwigluFinalSyncDebugOffset(op));
    volatile __gm__ DispatchFFNCombineSwigluFinalSyncDebug *entry = debug + op.coreIdx_;
    entry->magic = kSwigluFinalSyncDebugMagic;
    entry->coreIdx = op.coreIdx_;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->stageNum = op.stageNum_;
    entry->segmentNum = op.SwigluSegmentNum();
    entry->finalSyncEnabled = 1U;
    entry->finalSyncAfterAllSegments = 1U;
    entry->v2cEnabled = 0U;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + op.coreIdx_), sizeof(DispatchFFNCombineSwigluFinalSyncDebug));
}
#endif

#if defined(__DAV_CUBE__)
template <typename Stage>
AICORE inline bool DeviceDebug::Gmm2DebugEnabled(const Stage &op)
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DEVICE_DEBUG
    return op.tilingData_->gmm2Tiling.gmm2DebugMode != 0U;
#else
    (void)op;
    return false;
#endif
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm2V2CDebugOffset(const Stage &op)
{
    return op.tilingData_->gmm2Tiling.gmm2DebugOffset + sizeof(DispatchFFNCombineGmm2LayoutDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm2V2CDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.SwigluSegmentNum()) * op.coreNum_ * sizeof(DispatchFFNCombineGmm2V2CDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm2SegmentDebugOffset(const Stage &op)
{
    return DeviceDebug::Gmm2V2CDebugOffset(op) + DeviceDebug::Gmm2V2CDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm2SegmentDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.SwigluSegmentNum()) * sizeof(DispatchFFNCombineGmm2SegmentDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm2TaskDebugOffset(const Stage &op)
{
    return DeviceDebug::Gmm2SegmentDebugOffset(op) + DeviceDebug::Gmm2SegmentDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm2TaskDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.expertPerRank_) * op.coreNum_ * sizeof(DispatchFFNCombineGmm2TaskDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm2DoneDebugOffset(const Stage &op)
{
    return DeviceDebug::Gmm2TaskDebugOffset(op) + DeviceDebug::Gmm2TaskDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm2DoneDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.coreNum_) * sizeof(DispatchFFNCombineGmm2DoneDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm2FinalSyncDebugOffset(const Stage &op)
{
    return DeviceDebug::Gmm2DoneDebugOffset(op) + DeviceDebug::Gmm2DoneDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::Gmm2FinalSyncDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.coreNum_) * sizeof(DispatchFFNCombineGmm2FinalSyncDebug);
}

template <typename Stage>
AICORE inline void DeviceDebug::Gmm2WriteLayoutDebugHeader(const Stage &op)
{
    if (op.coreIdx_ != 0 || op.tilingData_->gmm2Tiling.gmm2DebugBytes < sizeof(DispatchFFNCombineGmm2LayoutDebug)) {
        return;
    }

    __gm__ DispatchFFNCombineGmm2LayoutDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineGmm2LayoutDebug *>(
        op.workspaceGM_ + op.tilingData_->gmm2Tiling.gmm2DebugOffset);
    debug->magic = kGmm2LayoutDebugMagic;
    debug->workspaceBase = reinterpret_cast<uint64_t>(op.workspaceGM_);
    debug->gmPermutedTokenOffset = op.tilingData_->swigluTiling.gmPermutedTokenOffset;
    debug->perTokenScale2Offset = op.tilingData_->swigluTiling.perTokenScale2Offset;
    debug->gmm2OutputOffset = op.tilingData_->gmm2Tiling.gmm2OutputOffset;
    debug->weight2Base = reinterpret_cast<uint64_t>(op.weight2Ptr_);
    debug->scale2Base = reinterpret_cast<uint64_t>(op.scale2Ptr_);
    debug->cumsumMMOffset = op.tilingData_->frontReorderTiling.cumsumMMOffset;
    debug->expertTokenNumsBase = reinterpret_cast<uint64_t>(op.expertTokenNumsPtr_);
    debug->gmm2DebugOffset = op.tilingData_->gmm2Tiling.gmm2DebugOffset;
    debug->gmm2DebugBytes = op.tilingData_->gmm2Tiling.gmm2DebugBytes;
    debug->frontWorkspaceBytes = op.tilingData_->frontReorderTiling.frontWorkspaceBytes;
    debug->dispatchScratchOffset = op.tilingData_->dispatchTiling.dispatchScratchOffset;
    debug->gmm1DebugOffset = op.tilingData_->gmm1Tiling.gmm1DebugOffset;
    debug->swigluDebugOffset = op.tilingData_->swigluTiling.swigluDebugOffset;
    debug->gmPermutedTokenBytes = static_cast<uint64_t>(op.maxOutputSize_) * op.inputK_ * sizeof(int8_t);
    debug->perTokenScale2Bytes = static_cast<uint64_t>(op.maxOutputSize_) * sizeof(float);
    debug->gmm2OutputBytes = static_cast<uint64_t>(op.maxOutputSize_) * op.outputN_ * sizeof(half);
    debug->rank = op.rank_;
    debug->rankSize = op.rankSize_;
    debug->coreIdx = op.coreIdx_;
    debug->coreNum = op.coreNum_;
    debug->stageNum = op.stageNum_;
    debug->problemK = op.problemK_;
    debug->problemN = op.problemN_;
    debug->inputK = op.inputK_;
    debug->outputN = op.outputN_;
    debug->maxOutputSize = op.maxOutputSize_;
    debug->expertPerRank = op.expertPerRank_;
    debug->segmentNum = op.SwigluSegmentNum();
    debug->epilogueGranularity = op.SwigluEpilogueGranularity();
    debug->l1TileM = op.tilingData_->gmm2Tiling.l1TileM;
    debug->l1TileN = op.tilingData_->gmm2Tiling.l1TileN;
    debug->l1TileK = op.tilingData_->gmm2Tiling.l1TileK;
    debug->l0TileM = op.tilingData_->gmm2Tiling.l0TileM;
    debug->l0TileN = op.tilingData_->gmm2Tiling.l0TileN;
    debug->l0TileK = op.tilingData_->gmm2Tiling.l0TileK;
    debug->debugMode = op.tilingData_->gmm2Tiling.gmm2DebugMode;
    debug->commonReuseMode = kGmm2CommonReuseMode;
    debug->gmPermutedTokenRowBytes = op.inputK_ * sizeof(int8_t);
    debug->gmm2OutputRowBytes = op.outputN_ * sizeof(half);
    debug->perTokenScale2BytesPerRow = sizeof(float);
    debug->layoutVersion = kGmm2LayoutVersion;
    debug->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug), sizeof(DispatchFFNCombineGmm2LayoutDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::Gmm2WriteV2CDebug(const Stage &op, uint32_t segmentIdx)
{
    const uint64_t requiredBytes = DeviceDebug::Gmm2V2CDebugOffset(op) - op.tilingData_->gmm2Tiling.gmm2DebugOffset +
                                   DeviceDebug::Gmm2V2CDebugTableBytes(op);
    if (op.tilingData_->gmm2Tiling.gmm2DebugBytes < requiredBytes) {
        return;
    }

    __gm__ DispatchFFNCombineGmm2V2CDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineGmm2V2CDebug *>(
        op.workspaceGM_ + DeviceDebug::Gmm2V2CDebugOffset(op));
    const uint64_t debugIdx = static_cast<uint64_t>(segmentIdx) * op.coreNum_ + op.coreIdx_;
    volatile __gm__ DispatchFFNCombineGmm2V2CDebug *entry = debug + debugIdx;
    entry->magic = kGmm2V2CDebugMagic;
    entry->segmentIdx = segmentIdx;
    entry->coreIdx = op.coreIdx_;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->stageNum = op.stageNum_;
    entry->waitSource = kGmm2WaitSourceV2COnly;
    entry->v2cFlagId = op.V2CFlagId(segmentIdx);
    entry->v2cEnabled = 1U;
    entry->swigluDoneConsumed = 0U;
    entry->finalSyncConsumed = 0U;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + debugIdx), sizeof(DispatchFFNCombineGmm2V2CDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::Gmm2WriteSegmentDebug(const Stage &op, uint32_t segmentIdx, uint32_t segmentStartExpert,
                                                      uint32_t segmentEndExpert, uint32_t segmentRowBase,
                                                      uint32_t segmentRows, uint32_t cumsumRows,
                                                      uint32_t expertTokenRows)
{
    const uint64_t requiredBytes = DeviceDebug::Gmm2SegmentDebugOffset(op) -
                                   op.tilingData_->gmm2Tiling.gmm2DebugOffset +
                                   DeviceDebug::Gmm2SegmentDebugTableBytes(op);
    if (op.coreIdx_ != 0 || op.tilingData_->gmm2Tiling.gmm2DebugBytes < requiredBytes) {
        return;
    }

    __gm__ DispatchFFNCombineGmm2SegmentDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineGmm2SegmentDebug *>(
        op.workspaceGM_ + DeviceDebug::Gmm2SegmentDebugOffset(op));
    volatile __gm__ DispatchFFNCombineGmm2SegmentDebug *entry = debug + segmentIdx;
    entry->magic = kGmm2SegmentDebugMagic;
    entry->segmentIdx = segmentIdx;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->segmentStartExpert = segmentStartExpert;
    entry->segmentEndExpert = segmentEndExpert;
    entry->segmentRowBase = segmentRowBase;
    entry->segmentRows = segmentRows;
    entry->cumsumRows = cumsumRows;
    entry->expertTokenRows = expertTokenRows;
    entry->maxOutputSize = op.maxOutputSize_;
    entry->segmentNum = op.SwigluSegmentNum();
    entry->epilogueGranularity = op.SwigluEpilogueGranularity();
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + segmentIdx), sizeof(DispatchFFNCombineGmm2SegmentDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::Gmm2WriteTaskDebug(const Stage &op, uint32_t segmentIdx, uint32_t groupIdx,
                                                   uint32_t groupBase, uint32_t currentMRaw, uint32_t currentM,
                                                   uint32_t expertTokenNums, uint32_t startCoreIdx, uint32_t coreLoops)
{
    const uint64_t requiredBytes = DeviceDebug::Gmm2TaskDebugOffset(op) - op.tilingData_->gmm2Tiling.gmm2DebugOffset +
                                   DeviceDebug::Gmm2TaskDebugTableBytes(op);
    if (op.tilingData_->gmm2Tiling.gmm2DebugBytes < requiredBytes) {
        return;
    }

    __gm__ DispatchFFNCombineGmm2TaskDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineGmm2TaskDebug *>(
        op.workspaceGM_ + DeviceDebug::Gmm2TaskDebugOffset(op));
    const uint64_t debugIdx = static_cast<uint64_t>(groupIdx) * op.coreNum_ + op.coreIdx_;
    volatile __gm__ DispatchFFNCombineGmm2TaskDebug *entry = debug + debugIdx;

    const uint32_t tileM = op.TileM(currentM);
    const uint32_t tileN = op.TileN();
    const uint32_t startLoopIdx = op.StartLoopIdx(startCoreIdx);
    const uint32_t assignedTileCount = op.AssignedTileCount(startLoopIdx, coreLoops);
    uint32_t firstLoop = kGmm2InvalidTask;
    uint32_t lastLoop = kGmm2InvalidTask;
    uint32_t firstBlockM = kGmm2InvalidTask;
    uint32_t firstBlockN = kGmm2InvalidTask;
    uint32_t lastBlockM = kGmm2InvalidTask;
    uint32_t lastBlockN = kGmm2InvalidTask;
    uint32_t firstActualM = 0;
    uint32_t firstActualN = 0;
    uint32_t lastActualM = 0;
    uint32_t lastActualN = 0;

    if (assignedTileCount > 0U) {
        firstLoop = startLoopIdx;
        lastLoop = startLoopIdx + (assignedTileCount - 1U) * op.coreNum_;
        op.GetBlockCoordMN(firstLoop, tileM, tileN, firstBlockM, firstBlockN);
        op.GetBlockCoordMN(lastLoop, tileM, tileN, lastBlockM, lastBlockN);
        op.GetActualBlockShapeMN(firstBlockM, firstBlockN, tileM, tileN, currentM, firstActualM, firstActualN);
        op.GetActualBlockShapeMN(lastBlockM, lastBlockN, tileM, tileN, currentM, lastActualM, lastActualN);
    }

    entry->magic = kGmm2TaskDebugMagic;
    entry->segmentIdx = segmentIdx;
    entry->groupIdx = groupIdx;
    entry->coreIdx = op.coreIdx_;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->groupBase = groupBase;
    entry->currentMRaw = currentMRaw;
    entry->currentM = currentM;
    entry->expertTokenNums = expertTokenNums;
    entry->tileM = tileM;
    entry->tileN = tileN;
    entry->coreLoops = coreLoops;
    entry->startCoreIdx = startCoreIdx;
    entry->startLoopIdx = startLoopIdx;
    entry->assignedTileCount = assignedTileCount;
    entry->firstLoop = firstLoop;
    entry->lastLoop = lastLoop;
    entry->firstBlockM = firstBlockM;
    entry->firstBlockN = firstBlockN;
    entry->lastBlockM = lastBlockM;
    entry->lastBlockN = lastBlockN;
    entry->firstActualM = firstActualM;
    entry->firstActualN = firstActualN;
    entry->lastActualM = lastActualM;
    entry->lastActualN = lastActualN;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + debugIdx), sizeof(DispatchFFNCombineGmm2TaskDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::Gmm2WriteDoneDebug(const Stage &op)
{
    const uint64_t requiredBytes = DeviceDebug::Gmm2DoneDebugOffset(op) - op.tilingData_->gmm2Tiling.gmm2DebugOffset +
                                   DeviceDebug::Gmm2DoneDebugTableBytes(op);
    if (op.tilingData_->gmm2Tiling.gmm2DebugBytes < requiredBytes) {
        return;
    }

    __gm__ DispatchFFNCombineGmm2DoneDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineGmm2DoneDebug *>(
        op.workspaceGM_ + DeviceDebug::Gmm2DoneDebugOffset(op));
    volatile __gm__ DispatchFFNCombineGmm2DoneDebug *entry = debug + op.coreIdx_;
    entry->magic = kGmm2DoneDebugMagic;
    entry->coreIdx = op.coreIdx_;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->stageNum = op.stageNum_;
    entry->segmentNum = op.SwigluSegmentNum();
    entry->expertPerRank = op.expertPerRank_;
    entry->finalSyncEnabled = op.stageNum_ == 12U ? 1U : 0U;
    entry->combineReadyEnabled = op.CombineReadyEnabled() ? 1U : 0U;
    entry->pipelineDrained = 1U;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + op.coreIdx_), sizeof(DispatchFFNCombineGmm2DoneDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::Gmm2WriteFinalSyncDebug(const Stage &op)
{
    const uint64_t requiredBytes = DeviceDebug::Gmm2FinalSyncDebugOffset(op) -
                                   op.tilingData_->gmm2Tiling.gmm2DebugOffset +
                                   DeviceDebug::Gmm2FinalSyncDebugTableBytes(op);
    if (op.tilingData_->gmm2Tiling.gmm2DebugBytes < requiredBytes) {
        return;
    }

    __gm__ DispatchFFNCombineGmm2FinalSyncDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineGmm2FinalSyncDebug *>(op.workspaceGM_ +
                                                                        DeviceDebug::Gmm2FinalSyncDebugOffset(op));
    volatile __gm__ DispatchFFNCombineGmm2FinalSyncDebug *entry = debug + op.coreIdx_;
    entry->magic = kGmm2FinalSyncDebugMagic;
    entry->coreIdx = op.coreIdx_;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->stageNum = op.stageNum_;
    entry->segmentNum = op.SwigluSegmentNum();
    entry->expertPerRank = op.expertPerRank_;
    entry->finalSyncEnabled = op.stageNum_ == 12U ? 1U : 0U;
    entry->finalSyncAfterAllGroups = op.stageNum_ == 12U ? 1U : 0U;
    entry->combineReadyEnabled = op.CombineReadyEnabled() ? 1U : 0U;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + op.coreIdx_), sizeof(DispatchFFNCombineGmm2FinalSyncDebug));
}
#endif

#if defined(__DAV_VEC__)
template <typename Stage>
AICORE inline bool DeviceDebug::CombineDebugEnabled(const Stage &op)
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DEVICE_DEBUG
    return op.tilingData_->combineTiling.combineDebugMode != 0U;
#else
    (void)op;
    return false;
#endif
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineReadyDebugOffset(const Stage &op)
{
    return op.tilingData_->combineTiling.combineDebugOffset + sizeof(DispatchFFNCombineCombineLayoutDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineReadyDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.expertPerRank_) * op.coreNum_ * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineMetaDebugOffset(const Stage &op)
{
    return DeviceDebug::CombineReadyDebugOffset(op) + DeviceDebug::CombineReadyDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineMetaDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.expertPerRank_) * op.rankSize_ * sizeof(DispatchFFNCombineCombineMetadataDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineLoadDebugOffset(const Stage &op)
{
    return DeviceDebug::CombineMetaDebugOffset(op) + DeviceDebug::CombineMetaDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineLoadDebugTableBytes(const Stage &op)
{
    const uint32_t slots = op.rankSize_ > op.coreNum_ ? op.rankSize_ : op.coreNum_;
    return static_cast<uint64_t>(op.expertPerRank_) * slots * sizeof(DispatchFFNCombineCombineLoadDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineDequantDebugOffset(const Stage &op)
{
    return DeviceDebug::CombineLoadDebugOffset(op) + DeviceDebug::CombineLoadDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineDequantDebugTableBytes(const Stage &op)
{
    const uint32_t slots = op.rankSize_ > op.coreNum_ ? op.rankSize_ : op.coreNum_;
    return static_cast<uint64_t>(op.expertPerRank_) * slots * sizeof(DispatchFFNCombineCombineDequantDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineStoreDebugOffset(const Stage &op)
{
    return DeviceDebug::CombineDequantDebugOffset(op) + DeviceDebug::CombineDequantDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineStoreDebugTableBytes(const Stage &op)
{
    const uint32_t slots = op.rankSize_ > op.coreNum_ ? op.rankSize_ : op.coreNum_;
    return static_cast<uint64_t>(op.expertPerRank_) * slots * sizeof(DispatchFFNCombineCombineStoreDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineFinalizeDebugOffset(const Stage &op)
{
    return DeviceDebug::CombineStoreDebugOffset(op) + DeviceDebug::CombineStoreDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineFinalizeDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.coreNum_) * sizeof(DispatchFFNCombineCombineFinalizeDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineTaskDebugOffset(const Stage &op)
{
    return DeviceDebug::CombineFinalizeDebugOffset(op) + DeviceDebug::CombineFinalizeDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineTaskDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.expertPerRank_) * op.coreNum_ * sizeof(DispatchFFNCombineCombineTaskStatsDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineDoneDebugOffset(const Stage &op)
{
    return DeviceDebug::CombineTaskDebugOffset(op) + DeviceDebug::CombineTaskDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::CombineDoneDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.coreNum_) * sizeof(DispatchFFNCombineCombineDoneDebug);
}

template <typename Stage>
AICORE inline void DeviceDebug::CombineWriteLayoutDebugHeader(const Stage &op)
{
    if (op.coreIdx_ != 0 ||
        op.tilingData_->combineTiling.combineDebugBytes < sizeof(DispatchFFNCombineCombineLayoutDebug)) {
        return;
    }
    __gm__ DispatchFFNCombineCombineLayoutDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineCombineLayoutDebug *>(
            op.workspaceGM_ + op.tilingData_->combineTiling.combineDebugOffset);
    debug->magic = kCombineLayoutDebugMagic;
    debug->workspaceBase = reinterpret_cast<uint64_t>(op.workspaceGM_);
    debug->gmm2OutputOffset = op.tilingData_->combineTiling.gmm2OutputOffset;
    debug->perTokenScale2Offset = op.tilingData_->combineTiling.perTokenScale2Offset;
    debug->cumsumMMOffset = op.tilingData_->frontReorderTiling.cumsumMMOffset;
    debug->preSumBeforeRankOffset = op.tilingData_->frontReorderTiling.preSumBeforeRankOffset;
    debug->combineScratchOffset = op.tilingData_->combineTiling.combineScratchOffset;
    debug->combineScratchBytes = op.tilingData_->combineTiling.combineScratchBytes;
    debug->combineScratchBytesPerAiv = op.tilingData_->combineTiling.combineScratchBytesPerAiv;
    debug->combineScratchCoreOffset =
        op.tilingData_->combineTiling.combineScratchOffset +
        static_cast<uint64_t>(op.coreIdx_) * op.tilingData_->combineTiling.combineScratchBytesPerAiv;
    debug->combineDebugOffset = op.tilingData_->combineTiling.combineDebugOffset;
    debug->combineDebugBytes = op.tilingData_->combineTiling.combineDebugBytes;
    debug->peerOffsetD = static_cast<uint64_t>(op.peerMemoryLayout_.offsetD);
    debug->peerOffsetPeerTokenPerExpert = static_cast<uint64_t>(op.peerMemoryLayout_.offsetPeerTokenPerExpert);
    debug->peerOffsetScale2 = static_cast<uint64_t>(op.peerMemoryLayout_.offsetScale2);
    debug->peerSignalBaseOffset = op.remoteWindow_.SegmentSize() - MB_SIZE;
    debug->gmm2OutputBytes = static_cast<uint64_t>(op.maxOutputSize_) * op.problemK_ * sizeof(half);
    debug->perTokenScale2Bytes = static_cast<uint64_t>(op.maxOutputSize_) * sizeof(float);
    debug->offsetDBytes =
        static_cast<uint64_t>(op.maxOutputSize_) * op.problemK_ * sizeof(typename Stage::OutputElementType);
    debug->offsetScale2Bytes = static_cast<uint64_t>(op.maxOutputSize_) * sizeof(float);
    debug->offsetScale2CapacityBytes = MB_SIZE;
    debug->rank = op.rank_;
    debug->rankSize = op.rankSize_;
    debug->coreIdx = op.coreIdx_;
    debug->coreNum = op.coreNum_;
    debug->stageNum = op.stageNum_;
    debug->problemK = op.problemK_;
    debug->maxOutputSize = op.maxOutputSize_;
    debug->expertPerRank = op.expertPerRank_;
    debug->combineTileCols = op.TileCols();
    debug->combineTileRows = 0U;
    debug->debugMode = op.tilingData_->combineTiling.combineDebugMode;
    debug->outputElementBytes = sizeof(typename Stage::OutputElementType);
    debug->gmm2OutputRowBytes = op.problemK_ * sizeof(half);
    debug->offsetDRowBytes = op.problemK_ * sizeof(typename Stage::OutputElementType);
    debug->tileBytes = op.TileCols() * sizeof(typename Stage::OutputElementType);
    debug->scratchBytesPerBuffer = op.tilingData_->combineTiling.combineScratchBytesPerAiv / 2U;
    debug->layoutVersion = kCombineLayoutVersion;
    debug->marker = 1U;
    debug->offsetScale2CapacityOk = debug->offsetScale2Bytes <= debug->offsetScale2CapacityBytes ? 1U : 0U;
    debug->offsetScale2DNoOverlap =
        debug->peerOffsetScale2 + debug->offsetScale2CapacityBytes <= debug->peerOffsetD ? 1U : 0U;
    debug->offsetDIsRawHalf = 0U;
    debug->tokenVolume = static_cast<uint32_t>(op.TokenVolume());
    debug->isLargePath = op.IsSmallTokenPath() ? 0U : 1U;
    debug->combineImplMode = op.CombineImplMode();
    debug->combineStopStep = op.tilingData_->combineTiling.combineStopStep;
    debug->directLargeEnabled = op.DirectLargeEnabled() ? 1U : 0U;
    debug->directSmallEnabled = op.DirectSmallEnabled() ? 1U : 0U;
    debug->directLargeRequiredUbBytes = 0U;
    debug->directLargeUbCOffset0 = 0U;
    debug->directLargeUbDOffset0 = 0U;
    debug->directLargeUbFp32Offset0 = 0U;
    debug->directLargeUbCOffset1 = 0U;
    debug->directLargeUbDOffset1 = 0U;
    debug->directLargeUbFp32Offset1 = 0U;
    debug->directLargeTileCols = 0U;
    debug->directLargeUbStages = 0U;
    debug->directLargeRequiredUbBytesOk = 0U;
    debug->usesLaneSplit = 0U;
    debug->usesColumnFallback = 0U;
    debug->usesOldBusinessHelper = 0U;
    debug->directSmallRequiredUbBytes =
        kCombineDirectSmallUbStages *
        (alignUp(static_cast<uint64_t>(kCombineSmallMaxElems) * sizeof(half), UB_ALIGN) +
         alignUp(static_cast<uint64_t>(kCombineSmallMaxElems) * sizeof(typename Stage::OutputElementType), UB_ALIGN) +
         alignUp(static_cast<uint64_t>(kCombineSmallMaxElems) * sizeof(float), UB_ALIGN) +
         alignUp(static_cast<uint64_t>(kCombineSmallScaleElems) * sizeof(float), UB_ALIGN));
    debug->directSmallUbCOffset0 = op.ubCOffset_[0];
    debug->directSmallUbDOffset0 = op.ubDOffset_[0];
    debug->directSmallUbFp32Offset0 = op.ubFp32Offset_[0];
    debug->directSmallUbScaleOffset0 = op.ubScaleOffset_[0];
    debug->directSmallUbCOffset1 = op.ubCOffset_[1];
    debug->directSmallUbDOffset1 = op.ubDOffset_[1];
    debug->directSmallUbFp32Offset1 = op.ubFp32Offset_[1];
    debug->directSmallUbScaleOffset1 = op.ubScaleOffset_[1];
    debug->directSmallTileRows = kCombineSmallTokenSubtileRows;
    debug->directSmallTileCols = kCombineSmallTokenSubtileCols;
    debug->directSmallN0 = kCombineSmallTokenSubtileCols;
    debug->directSmallUbStages = kCombineDirectSmallUbStages;
    debug->directSmallMaxElems = kCombineSmallMaxElems;
    debug->directSmallScaleElems = kCombineSmallScaleElems;
    debug->directSmallRequiredUbBytesOk = debug->directSmallRequiredUbBytes <= AtlasA5::UB_SIZE ? 1U : 0U;
    debug->usesFrontCaseBranch = 0U;
    debug->gmm2CombineCvMode = op.tilingData_->combineTiling.gmm2CombineCvMode;
    debug->gmm2CombineCvDebugMode = op.tilingData_->combineTiling.gmm2CombineCvDebugMode;
    debug->gmm2CombineCvReadyFlag = V8_GMM2_COMBINE_CV_READY_HARD_FLAG;
    debug->gmm2CombineCvFreeFlag = V8_GMM2_COMBINE_CV_FREE_HARD_FLAG;
    debug->gmm2CombineCvSlotOffset = V8_GMM2_COMBINE_CV_SLOT_OFFSET;
    debug->gmm2CombineCvTileScaleOffset = V8_GMM2_COMBINE_CV_TILE_SCALE_OFFSET;
    debug->gmm2CombineCvRequiredUbBytes = V8_GMM2_COMBINE_CV_REQUIRED_UB_BYTES;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug), sizeof(DispatchFFNCombineCombineLayoutDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::CombineWriteReadyDebug(const Stage &op, uint32_t groupIdx, bool aivSyncAfterWait,
                                                       uint64_t waitStartSyscnt, uint64_t waitEndSyscnt,
                                                       uint64_t syncAllStartSyscnt, uint64_t syncAllEndSyscnt)
{
    const uint64_t requiredBytes = DeviceDebug::CombineReadyDebugOffset(op) -
                                   op.tilingData_->combineTiling.combineDebugOffset +
                                   DeviceDebug::CombineReadyDebugTableBytes(op);
    if (op.tilingData_->combineTiling.combineDebugBytes < requiredBytes) {
        return;
    }
    __gm__ DispatchFFNCombineCombineGmm2ReadyDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineCombineGmm2ReadyDebug *>(op.workspaceGM_ +
                                                                           DeviceDebug::CombineReadyDebugOffset(op));
    const uint64_t idx = static_cast<uint64_t>(groupIdx) * op.coreNum_ + op.coreIdx_;
    volatile __gm__ DispatchFFNCombineCombineGmm2ReadyDebug *entry = debug + idx;
    entry->magic = kCombineReadyDebugMagic;
    entry->waitStartSyscnt = waitStartSyscnt;
    entry->waitEndSyscnt = waitEndSyscnt;
    entry->syncAllStartSyscnt = syncAllStartSyscnt;
    entry->syncAllEndSyscnt = syncAllEndSyscnt;
    entry->groupIdx = groupIdx;
    entry->coreIdx = op.coreIdx_;
    entry->flagId = op.Gmm2ToCombineFlagId(groupIdx);
    entry->aivSyncAfterWait = aivSyncAfterWait ? 1U : 0U;
    entry->initialEventMask = 0x00030003U;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + idx), sizeof(DispatchFFNCombineCombineGmm2ReadyDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::CombineWriteMetaDebug(const Stage &op, uint32_t groupIdx, uint32_t srcRank,
                                                      uint32_t groupBase, uint32_t currentM)
{
    const uint64_t requiredBytes = DeviceDebug::CombineMetaDebugOffset(op) -
                                   op.tilingData_->combineTiling.combineDebugOffset +
                                   DeviceDebug::CombineMetaDebugTableBytes(op);
    if (op.coreIdx_ != 0 || op.tilingData_->combineTiling.combineDebugBytes < requiredBytes) {
        return;
    }
    __gm__ DispatchFFNCombineCombineMetadataDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineCombineMetadataDebug *>(op.workspaceGM_ +
                                                                          DeviceDebug::CombineMetaDebugOffset(op));
    const uint64_t idx = static_cast<uint64_t>(groupIdx) * op.rankSize_ + srcRank;
    const uint32_t cumsumBeforeSrc = op.CumsumBeforeSource(srcRank, groupIdx);
    const uint32_t srcRowOffset = groupBase + cumsumBeforeSrc;
    const uint32_t rowsRaw = op.RowsRaw(srcRank, groupIdx);
    const uint32_t rows = op.RowsClipped(srcRowOffset, rowsRaw);
    volatile __gm__ DispatchFFNCombineCombineMetadataDebug *entry = debug + idx;
    entry->magic = kCombineMetaDebugMagic;
    entry->groupIdx = groupIdx;
    entry->srcRank = srcRank;
    entry->coreIdx = op.coreIdx_;
    entry->coreNum = op.coreNum_;
    entry->groupBase = groupBase;
    entry->groupBaseAfter = groupBase + currentM;
    entry->rowsRaw = rowsRaw;
    entry->rows = rows;
    entry->srcRowOffset = srcRowOffset;
    entry->dstRowOffset = op.DstRowOffset(srcRank, groupIdx);
    entry->cumsumBeforeSrc = cumsumBeforeSrc;
    entry->clipped = rows != rowsRaw ? 1U : 0U;
    entry->skipReason = srcRowOffset >= op.maxOutputSize_ ? 1U : 0U;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + idx), sizeof(DispatchFFNCombineCombineMetadataDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::CombineWriteDirectLargeMetaDebug(const Stage &op, uint32_t groupIdx, uint32_t srcRank,
                                                                 uint32_t groupBase, uint32_t currentM)
{
    const uint64_t requiredBytes = DeviceDebug::CombineMetaDebugOffset(op) -
                                   op.tilingData_->combineTiling.combineDebugOffset +
                                   DeviceDebug::CombineMetaDebugTableBytes(op);
    if (op.tilingData_->combineTiling.combineDebugBytes < requiredBytes) {
        return;
    }
    __gm__ DispatchFFNCombineCombineMetadataDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineCombineMetadataDebug *>(op.workspaceGM_ +
                                                                          DeviceDebug::CombineMetaDebugOffset(op));
    const uint64_t idx = static_cast<uint64_t>(groupIdx) * op.rankSize_ + srcRank;
    const uint32_t cumsumBeforeSrc = op.CumsumBeforeSource(srcRank, groupIdx);
    const uint32_t srcRowOffset = groupBase + cumsumBeforeSrc;
    const uint32_t rowsRaw = op.RowsRaw(srcRank, groupIdx);
    const uint32_t rows = op.RowsClipped(srcRowOffset, rowsRaw);
    volatile __gm__ DispatchFFNCombineCombineMetadataDebug *entry = debug + idx;
    entry->magic = kCombineMetaDebugMagic;
    entry->groupIdx = groupIdx;
    entry->srcRank = srcRank;
    entry->coreIdx = op.coreIdx_;
    entry->coreNum = op.coreNum_;
    entry->groupBase = groupBase;
    entry->groupBaseAfter = groupBase + currentM;
    entry->rowsRaw = rowsRaw;
    entry->rows = rows;
    entry->srcRowOffset = srcRowOffset;
    entry->dstRowOffset = op.DstRowOffset(srcRank, groupIdx);
    entry->cumsumBeforeSrc = cumsumBeforeSrc;
    entry->clipped = rows != rowsRaw ? 1U : 0U;
    entry->skipReason = srcRowOffset >= op.maxOutputSize_ ? 1U : 0U;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

template <typename Stage>
AICORE inline void DeviceDebug::CombineWriteLoadDebug(const Stage &op, uint32_t groupIdx, uint32_t srcRank,
                                                      uint32_t srcRow, uint32_t dstRow, uint32_t rows, float scaleValue,
                                                      float cFirst, float cLast)
{
    const uint64_t requiredBytes = DeviceDebug::CombineLoadDebugOffset(op) -
                                   op.tilingData_->combineTiling.combineDebugOffset +
                                   DeviceDebug::CombineLoadDebugTableBytes(op);
    if (op.tilingData_->combineTiling.combineDebugBytes < requiredBytes) {
        return;
    }
    union FloatBits {
        float f;
        uint32_t u;
    };
    __gm__ DispatchFFNCombineCombineLoadDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineCombineLoadDebug *>(
        op.workspaceGM_ + DeviceDebug::CombineLoadDebugOffset(op));
    const uint32_t slots = op.rankSize_ > op.coreNum_ ? op.rankSize_ : op.coreNum_;
    const uint64_t idx = static_cast<uint64_t>(groupIdx) * slots + srcRank;
    FloatBits scaleBits;
    FloatBits firstBits;
    FloatBits lastBits;
    scaleBits.f = scaleValue;
    firstBits.f = cFirst;
    lastBits.f = cLast;
    volatile __gm__ DispatchFFNCombineCombineLoadDebug *entry = debug + idx;
    entry->magic = kCombineLoadDebugMagic;
    entry->groupIdx = groupIdx;
    entry->srcRank = srcRank;
    entry->coreIdx = op.coreIdx_;
    entry->coreNum = op.coreNum_;
    entry->srcRow = srcRow;
    entry->dstRow = dstRow;
    entry->rows = rows;
    entry->cLoadBytes = rows == 0U ? 0U : op.problemK_ * sizeof(half);
    entry->scaleOffsetBytes = rows == 0U ? 0U : srcRow * sizeof(float);
    entry->scaleBits = rows == 0U ? 0U : scaleBits.u;
    entry->cFirstBits = rows == 0U ? 0U : firstBits.u;
    entry->cLastBits = rows == 0U ? 0U : lastBits.u;
    entry->scaleReadMode = kCombineScaleReadDirectScalar;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

template <typename Stage>
AICORE inline void DeviceDebug::CombineWriteDequantDebug(const Stage &op, uint32_t groupIdx, uint32_t srcRank,
                                                         uint32_t srcRow, uint32_t rows, float scaleValue,
                                                         float fp32BeforeFirst, float fp32AfterFirst,
                                                         float fp32AfterLast, float dFirst, float dLast)
{
    const uint64_t requiredBytes = DeviceDebug::CombineDequantDebugOffset(op) -
                                   op.tilingData_->combineTiling.combineDebugOffset +
                                   DeviceDebug::CombineDequantDebugTableBytes(op);
    if (op.tilingData_->combineTiling.combineDebugBytes < requiredBytes) {
        return;
    }
    union FloatBits {
        float f;
        uint32_t u;
    };
    __gm__ DispatchFFNCombineCombineDequantDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineCombineDequantDebug *>(op.workspaceGM_ +
                                                                         DeviceDebug::CombineDequantDebugOffset(op));
    const uint32_t slots = op.rankSize_ > op.coreNum_ ? op.rankSize_ : op.coreNum_;
    const uint64_t idx = static_cast<uint64_t>(groupIdx) * slots + srcRank;
    FloatBits scaleBits;
    FloatBits beforeBits;
    FloatBits afterFirstBits;
    FloatBits afterLastBits;
    FloatBits dFirstBits;
    FloatBits dLastBits;
    scaleBits.f = scaleValue;
    beforeBits.f = fp32BeforeFirst;
    afterFirstBits.f = fp32AfterFirst;
    afterLastBits.f = fp32AfterLast;
    dFirstBits.f = dFirst;
    dLastBits.f = dLast;
    volatile __gm__ DispatchFFNCombineCombineDequantDebug *entry = debug + idx;
    entry->magic = kCombineDequantDebugMagic;
    entry->groupIdx = groupIdx;
    entry->srcRank = srcRank;
    entry->coreIdx = op.coreIdx_;
    entry->coreNum = op.coreNum_;
    entry->srcRow = srcRow;
    entry->rows = rows;
    entry->scaleBits = rows == 0U ? 0U : scaleBits.u;
    entry->fp32BeforeFirstBits = rows == 0U ? 0U : beforeBits.u;
    entry->fp32AfterFirstBits = rows == 0U ? 0U : afterFirstBits.u;
    entry->fp32AfterLastBits = rows == 0U ? 0U : afterLastBits.u;
    entry->dFirstBits = rows == 0U ? 0U : dFirstBits.u;
    entry->dLastBits = rows == 0U ? 0U : dLastBits.u;
    entry->opCounts = rows == 0U ? 0U : kCombineDequantOpCounts;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

template <typename Stage>
AICORE inline void DeviceDebug::CombineWriteStoreDebug(const Stage &op, uint32_t groupIdx, uint32_t srcRank,
                                                       uint32_t srcRow, uint32_t dstRow, uint32_t rows,
                                                       bool remoteBaseValid, float storedFirst, float storedLast)
{
    const uint64_t requiredBytes = DeviceDebug::CombineStoreDebugOffset(op) -
                                   op.tilingData_->combineTiling.combineDebugOffset +
                                   DeviceDebug::CombineStoreDebugTableBytes(op);
    if (op.tilingData_->combineTiling.combineDebugBytes < requiredBytes) {
        return;
    }
    union FloatBits {
        float f;
        uint32_t u;
    };
    __gm__ DispatchFFNCombineCombineStoreDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineCombineStoreDebug *>(
        op.workspaceGM_ + DeviceDebug::CombineStoreDebugOffset(op));
    const uint32_t slots = op.rankSize_ > op.coreNum_ ? op.rankSize_ : op.coreNum_;
    const uint64_t idx = static_cast<uint64_t>(groupIdx) * slots + srcRank;
    FloatBits firstBits;
    FloatBits lastBits;
    firstBits.f = storedFirst;
    lastBits.f = storedLast;
    const bool hasStore = rows != 0U && remoteBaseValid;
    volatile __gm__ DispatchFFNCombineCombineStoreDebug *entry = debug + idx;
    entry->magic = kCombineStoreDebugMagic;
    entry->dstGmOffsetBytes =
        hasStore ? static_cast<uint64_t>(dstRow) * op.problemK_ * sizeof(typename Stage::OutputElementType) : 0U;
    entry->groupIdx = groupIdx;
    entry->srcRank = srcRank;
    entry->coreIdx = op.coreIdx_;
    entry->srcRow = srcRow;
    entry->dstRow = dstRow;
    entry->rows = rows;
    entry->remoteRank = srcRank;
    entry->remoteBaseValid = hasStore ? 1U : 0U;
    entry->storeBytes = hasStore ? op.problemK_ * sizeof(typename Stage::OutputElementType) : 0U;
    entry->storedFirstBits = hasStore ? firstBits.u : 0U;
    entry->storedLastBits = hasStore ? lastBits.u : 0U;
    entry->localOrRemote = srcRank == op.rank_ ? 0U : 1U;
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

template <typename Stage>
AICORE inline void DeviceDebug::CombineWriteFinalizeDebug(const Stage &op, uint64_t finalizeWaitStartSyscnt,
                                                          uint64_t finalizeWaitEndSyscnt, uint64_t syncAllStartSyscnt,
                                                          uint64_t syncAllEndSyscnt, uint64_t resetStartSyscnt,
                                                          uint64_t resetEndSyscnt, uint64_t crossRankSyncStartSyscnt,
                                                          uint64_t crossRankSyncEndSyscnt, uint32_t resetElems,
                                                          bool resetWriter)
{
    const uint64_t requiredBytes = DeviceDebug::CombineFinalizeDebugOffset(op) -
                                   op.tilingData_->combineTiling.combineDebugOffset +
                                   DeviceDebug::CombineFinalizeDebugTableBytes(op);
    if (op.tilingData_->combineTiling.combineDebugBytes < requiredBytes) {
        return;
    }
    __gm__ DispatchFFNCombineCombineFinalizeDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineCombineFinalizeDebug *>(op.workspaceGM_ +
                                                                          DeviceDebug::CombineFinalizeDebugOffset(op));
    volatile __gm__ DispatchFFNCombineCombineFinalizeDebug *entry = debug + op.coreIdx_;
    const uint32_t firstSample =
        (!resetWriter || resetElems == 0U) ? 0U : static_cast<uint32_t>(op.tokenPerExpertPtr_[0]);
    const uint32_t lastSample =
        (!resetWriter || resetElems == 0U) ?
            0U :
            static_cast<uint32_t>(op.tokenPerExpertPtr_[static_cast<uint64_t>(resetElems) - 1U]);
    entry->magic = kCombineFinalizeDebugMagic;
    entry->finalizeWaitStartSyscnt = finalizeWaitStartSyscnt;
    entry->finalizeWaitEndSyscnt = finalizeWaitEndSyscnt;
    entry->syncAllStartSyscnt = syncAllStartSyscnt;
    entry->syncAllEndSyscnt = syncAllEndSyscnt;
    entry->resetStartSyscnt = resetStartSyscnt;
    entry->resetEndSyscnt = resetEndSyscnt;
    entry->crossRankSyncStartSyscnt = crossRankSyncStartSyscnt;
    entry->crossRankSyncEndSyscnt = crossRankSyncEndSyscnt;
    entry->tokenPerExpertBaseOffsetBytes = static_cast<uint64_t>(op.peerMemoryLayout_.offsetPeerTokenPerExpert);
    entry->coreIdx = op.coreIdx_;
    entry->coreNum = op.coreNum_;
    entry->resetElems = resetElems;
    entry->resetWriter = resetWriter ? 1U : 0U;
    entry->tokenPerExpertResetFirstSample = firstSample;
    entry->tokenPerExpertResetLastSample = lastSample;
    entry->boundaryDoneMarker = 1U;
    entry->crossRankSyncAfterReset = crossRankSyncStartSyscnt >= resetEndSyscnt ? 1U : 0U;
    entry->marker = 1U;
    entry->reserved0 = 0U;
    entry->reserved1 = 0U;
    entry->reserved2 = 0U;
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

template <typename Stage>
AICORE inline void DeviceDebug::CombineWriteTaskDebug(const Stage &op, uint32_t groupIdx, uint32_t groupBase,
                                                      uint32_t currentM)
{
    const uint64_t requiredBytes = DeviceDebug::CombineTaskDebugOffset(op) -
                                   op.tilingData_->combineTiling.combineDebugOffset +
                                   DeviceDebug::CombineTaskDebugTableBytes(op);
    if (op.tilingData_->combineTiling.combineDebugBytes < requiredBytes) {
        return;
    }
    const uint32_t l1TileM = op.tilingData_->gmm2Tiling.l1TileM;
    const uint32_t l1TileN = op.tilingData_->gmm2Tiling.l1TileN;
    const uint32_t aicCoreIdx = get_block_idx();
    const uint32_t aicCoreNum = get_block_num();
    const uint32_t aivSubCoreIdx = get_subblockid();
    uint32_t startCoreIdx = 0U;
    uint32_t scanGroupBase = 0U;
    for (uint32_t group = 0U; group < groupIdx; ++group) {
        const uint32_t prevCurrentM = op.ClipCurrentM(op.CurrentM(group), scanGroupBase);
        const uint32_t prevCoreLoops = GmmCommonCoreLoops(prevCurrentM, op.problemK_, l1TileM, l1TileN);
        startCoreIdx = aicCoreNum == 0U ? 0U : (startCoreIdx + prevCoreLoops) % aicCoreNum;
        scanGroupBase += prevCurrentM;
    }
    const uint32_t currentMRaw = op.CurrentM(groupIdx);
    const uint32_t tileM = GmmCommonTileM(currentM, l1TileM);
    const uint32_t tileN = GmmCommonTileN(op.problemK_, l1TileN);
    const uint32_t coreLoops = tileM * tileN;
    const uint32_t startLoopIdx = aicCoreNum == 0U ? 0U : GmmCommonStartLoopIdx(aicCoreIdx, aicCoreNum, startCoreIdx);

    uint32_t assignedBlockCount = 0U;
    uint32_t coreRowCount = 0U;
    uint32_t firstLoopIdx = kCombineInvalidTask;
    uint32_t firstRows = 0U;
    uint32_t firstRowStart = 0U;
    uint32_t firstColStart = 0U;
    uint32_t firstActualM = 0U;
    uint32_t firstActualN = 0U;
    uint32_t lastLoopIdx = kCombineInvalidTask;
    uint32_t lastRows = 0U;
    uint32_t lastRowStart = 0U;
    uint32_t lastColStart = 0U;
    uint32_t lastActualM = 0U;
    uint32_t lastActualN = 0U;

    for (uint32_t loopIdx = startLoopIdx; aicCoreNum != 0U && loopIdx < coreLoops; loopIdx += aicCoreNum) {
        const GmmCommonTileInfo tileInfo = GmmCommonBuildTileInfo(currentM, op.problemK_, l1TileM, l1TileN, loopIdx);
        const uint32_t subtileCount = static_cast<uint32_t>(ceilDiv(tileInfo.actualM, kCombineSmallTokenSubtileRows));
        const uint32_t firstSubtile = op.SubtileFirstIndex(subtileCount, aivSubCoreIdx);
        const uint32_t assignedSubtiles = op.SubtileAssignedCount(subtileCount, aivSubCoreIdx);
        for (uint32_t subtile = 0U; subtile < assignedSubtiles; ++subtile) {
            const uint32_t subtileIdx = firstSubtile + subtile;
            const uint32_t rowInTile = subtileIdx * kCombineSmallTokenSubtileRows;
            if (rowInTile >= tileInfo.actualM) {
                continue;
            }
            const uint32_t rows = (tileInfo.actualM - rowInTile > kCombineSmallTokenSubtileRows) ?
                                      kCombineSmallTokenSubtileRows :
                                      (tileInfo.actualM - rowInTile);
            const uint32_t rowStart = tileInfo.blockRowStart + rowInTile;
            const uint32_t colStart = tileInfo.blockColStart;
            if (firstLoopIdx == kCombineInvalidTask) {
                firstLoopIdx = loopIdx;
                firstRows = rows;
                firstRowStart = rowStart;
                firstColStart = colStart;
                firstActualM = tileInfo.actualM;
                firstActualN = tileInfo.actualN;
            }
            lastLoopIdx = loopIdx;
            lastRows = rows;
            lastRowStart = rowStart;
            lastColStart = colStart;
            lastActualM = tileInfo.actualM;
            lastActualN = tileInfo.actualN;
            ++assignedBlockCount;
            coreRowCount += rows;
        }
    }

    __gm__ DispatchFFNCombineCombineTaskStatsDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineCombineTaskStatsDebug *>(op.workspaceGM_ +
                                                                           DeviceDebug::CombineTaskDebugOffset(op));
    const uint64_t idx = static_cast<uint64_t>(groupIdx) * op.coreNum_ + op.coreIdx_;
    volatile __gm__ DispatchFFNCombineCombineTaskStatsDebug *entry = debug + idx;
    entry->magic = kCombineTaskDebugMagic;
    entry->groupIdx = groupIdx;
    entry->coreIdx = op.coreIdx_;
    entry->rankSize = op.rankSize_;
    entry->assignedRankCount = assignedBlockCount;
    entry->coreRowCount = coreRowCount;
    entry->groupBase = groupBase;
    entry->currentM = currentM;
    entry->firstSrcRank = firstLoopIdx;
    entry->firstRows = firstRows;
    entry->firstSrcRowOffset = firstRowStart;
    entry->firstDstRowOffset = firstColStart;
    entry->lastSrcRank = lastLoopIdx;
    entry->lastRows = lastRows;
    entry->lastSrcRowOffset = lastRowStart;
    entry->lastDstRowOffset = lastColStart;
    entry->marker = 1U;
    entry->currentMRaw = currentMRaw;
    entry->coreLoops = coreLoops;
    entry->startCoreIdx = startCoreIdx;
    entry->startLoopIdx = startLoopIdx;
    entry->aicCoreIdx = aicCoreIdx;
    entry->aicCoreNum = aicCoreNum;
    entry->aivSubCoreIdx = aivSubCoreIdx;
    entry->tileM = tileM;
    entry->tileN = tileN;
    entry->firstActualM = firstActualM;
    entry->firstActualN = firstActualN;
    entry->lastActualM = lastActualM;
    entry->lastActualN = lastActualN;
    entry->reserved0 = 0U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + idx), sizeof(DispatchFFNCombineCombineTaskStatsDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::CombineWriteDoneDebug(const Stage &op)
{
    const uint64_t requiredBytes = DeviceDebug::CombineDoneDebugOffset(op) -
                                   op.tilingData_->combineTiling.combineDebugOffset +
                                   DeviceDebug::CombineDoneDebugTableBytes(op);
    if (op.tilingData_->combineTiling.combineDebugBytes < requiredBytes) {
        return;
    }
    __gm__ DispatchFFNCombineCombineDoneDebug *debug = reinterpret_cast<__gm__ DispatchFFNCombineCombineDoneDebug *>(
        op.workspaceGM_ + DeviceDebug::CombineDoneDebugOffset(op));
    volatile __gm__ DispatchFFNCombineCombineDoneDebug *entry = debug + op.coreIdx_;
    entry->magic = kCombineDoneDebugMagic;
    entry->coreIdx = op.coreIdx_;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->stageNum = op.stageNum_;
    entry->expertPerRank = op.expertPerRank_;
    entry->syncBeforeDone = 1U;
    entry->crossRankSync = 1U;
    entry->marker = 1U;
    entry->localSegmentCount = op.localSegmentCount_;
    entry->remoteSegmentCount = op.remoteSegmentCount_;
    entry->localRows = op.localRows_;
    entry->remoteRows = op.remoteRows_;
    entry->localBytes = op.localBytes_;
    entry->remoteBytes = op.remoteBytes_;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + op.coreIdx_), sizeof(DispatchFFNCombineCombineDoneDebug));
}

template <typename Stage>
AICORE inline bool DeviceDebug::UnpermuteDebugEnabled(const Stage &op)
{
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DEVICE_DEBUG
    return op.tilingData_->unpermuteTiling.unpermuteDebugMode != 0U;
#else
    (void)op;
    return false;
#endif
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::UnpermuteTaskDebugOffset(const Stage &op)
{
    return op.tilingData_->unpermuteTiling.unpermuteDebugOffset + sizeof(DispatchFFNCombineUnpermuteLayoutDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::UnpermuteTaskDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.coreNum_) * sizeof(DispatchFFNCombineUnpermuteTaskDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::UnpermuteMetaDebugOffset(const Stage &op)
{
    return DeviceDebug::UnpermuteTaskDebugOffset(op) + DeviceDebug::UnpermuteTaskDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::UnpermuteMetaDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.coreNum_) * sizeof(DispatchFFNCombineUnpermuteMetaDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::UnpermuteAccumDebugOffset(const Stage &op)
{
    return DeviceDebug::UnpermuteMetaDebugOffset(op) + DeviceDebug::UnpermuteMetaDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::UnpermuteAccumDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.coreNum_) * sizeof(DispatchFFNCombineUnpermuteAccumDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::UnpermuteOutputDebugOffset(const Stage &op)
{
    return DeviceDebug::UnpermuteAccumDebugOffset(op) + DeviceDebug::UnpermuteAccumDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::UnpermuteOutputDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.coreNum_) * sizeof(DispatchFFNCombineUnpermuteOutputDebug);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::UnpermuteDoneDebugOffset(const Stage &op)
{
    return DeviceDebug::UnpermuteOutputDebugOffset(op) + DeviceDebug::UnpermuteOutputDebugTableBytes(op);
}

template <typename Stage>
AICORE inline uint64_t DeviceDebug::UnpermuteDoneDebugTableBytes(const Stage &op)
{
    return static_cast<uint64_t>(op.coreNum_) * sizeof(DispatchFFNCombineUnpermuteDoneDebug);
}

template <typename Stage>
AICORE inline void DeviceDebug::UnpermuteWriteLayoutDebugHeader(const Stage &op)
{
    if (op.coreIdx_ != 0 ||
        op.tilingData_->unpermuteTiling.unpermuteDebugBytes < sizeof(DispatchFFNCombineUnpermuteLayoutDebug)) {
        return;
    }
    __gm__ DispatchFFNCombineUnpermuteLayoutDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineUnpermuteLayoutDebug *>(
            op.workspaceGM_ + op.tilingData_->unpermuteTiling.unpermuteDebugOffset);
    debug->magic = kUnpermuteLayoutDebugMagic;
    debug->workspaceBase = reinterpret_cast<uint64_t>(op.workspaceGM_);
    debug->offsetDBase = reinterpret_cast<uint64_t>(op.offsetDPtr_);
    debug->expandedRowIdxBase = reinterpret_cast<uint64_t>(op.expandedRowIdxPtr_);
    debug->probsBase = reinterpret_cast<uint64_t>(op.probsPtr_);
    debug->outBase = reinterpret_cast<uint64_t>(op.outPtr_);
    debug->unpermuteDebugOffset = op.tilingData_->unpermuteTiling.unpermuteDebugOffset;
    debug->unpermuteDebugBytes = op.tilingData_->unpermuteTiling.unpermuteDebugBytes;
    debug->peerOffsetD = static_cast<uint64_t>(op.peerMemoryLayout_.offsetD);
    debug->offsetScale2Base = reinterpret_cast<uint64_t>(op.remoteWindow_() + op.peerMemoryLayout_.offsetScale2);
    debug->peerOffsetScale2 = static_cast<uint64_t>(op.peerMemoryLayout_.offsetScale2);
    debug->offsetDBytes =
        static_cast<uint64_t>(op.maxOutputSize_) * op.problemK_ * sizeof(typename Stage::OutputElementType);
    debug->offsetScale2Bytes = static_cast<uint64_t>(op.maxOutputSize_) * sizeof(float);
    debug->offsetScale2CapacityBytes = MB_SIZE;
    debug->outBytes = static_cast<uint64_t>(op.problemM_) * op.problemK_ * sizeof(typename Stage::OutputElementType);
    debug->expandedRowIdxBytes = static_cast<uint64_t>(alignUp(op.problemM_, 256U)) * op.topK_ * sizeof(int32_t);
    debug->probsBytes = static_cast<uint64_t>(op.problemM_) * op.topK_ * sizeof(float);
    debug->frontWorkspaceBytes = op.tilingData_->frontReorderTiling.frontWorkspaceBytes;
    debug->rank = op.rank_;
    debug->rankSize = op.rankSize_;
    debug->coreIdx = op.coreIdx_;
    debug->coreNum = op.coreNum_;
    debug->stageNum = op.stageNum_;
    debug->problemM = op.problemM_;
    debug->problemK = op.problemK_;
    debug->topK = op.topK_;
    debug->maxOutputSize = op.maxOutputSize_;
    debug->expandedRowsValid = op.expandedRowsValid_;
    debug->tileCols = op.TileCols();
    debug->tokenBatch = op.TokenBatch();
    debug->debugMode = op.tilingData_->unpermuteTiling.unpermuteDebugMode;
    debug->layoutVersion = op.tilingData_->unpermuteTiling.unpermuteLayoutVersion;
    debug->outputElementBytes = sizeof(typename Stage::OutputElementType);
    debug->offsetDRowBytes = op.problemK_ * sizeof(typename Stage::OutputElementType);
    debug->outRowBytes = op.problemK_ * sizeof(typename Stage::OutputElementType);
    debug->tileBytes = op.TileCols() * sizeof(typename Stage::OutputElementType);
    debug->metadataBytesPerBatch = op.TokenBatch() * op.topK_ * (sizeof(int32_t) + sizeof(float));
    debug->ubMainBytes = static_cast<uint32_t>(op.ubMainBytes_);
    debug->metadataBufferNum = kUnpermuteMetadataBufferNum;
    debug->tokenBufferNum = kUnpermuteTokenBufferNum;
    debug->syncBoundaryInCombine = 1U;
    debug->crossRankSyncInUnpermute = 0U;
    debug->taskSplitMode = kUnpermuteTaskSplitOutputToken;
    debug->kTileMode = kUnpermuteKTileMode;
    debug->probsDtypeBytes = sizeof(float);
    debug->indexDtypeBytes = sizeof(int32_t);
    debug->marker = 1U;
    debug->offsetScale2CapacityOk = debug->offsetScale2Bytes <= debug->offsetScale2CapacityBytes ? 1U : 0U;
    debug->offsetScale2DNoOverlap =
        debug->peerOffsetScale2 + debug->offsetScale2CapacityBytes <= debug->peerOffsetD ? 1U : 0U;
    debug->offsetDIsRawHalf = 0U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug), sizeof(DispatchFFNCombineUnpermuteLayoutDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::UnpermuteWriteTaskDebug(const Stage &op)
{
    const uint64_t requiredBytes = DeviceDebug::UnpermuteTaskDebugOffset(op) -
                                   op.tilingData_->unpermuteTiling.unpermuteDebugOffset +
                                   DeviceDebug::UnpermuteTaskDebugTableBytes(op);
    if (op.tilingData_->unpermuteTiling.unpermuteDebugBytes < requiredBytes) {
        return;
    }
    __gm__ DispatchFFNCombineUnpermuteTaskDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineUnpermuteTaskDebug *>(op.workspaceGM_ +
                                                                        DeviceDebug::UnpermuteTaskDebugOffset(op));
    volatile __gm__ DispatchFFNCombineUnpermuteTaskDebug *entry = debug + op.coreIdx_;
    entry->magic = kUnpermuteTaskDebugMagic;
    entry->coreIdx = op.coreIdx_;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->stageNum = op.stageNum_;
    entry->tokenStart = op.tokenStart_;
    entry->tokenCount = op.tokenCount_;
    entry->expandedStart = op.tokenStart_ * op.topK_;
    entry->expandedCount = op.tokenCount_ * op.topK_;
    entry->splitBase = op.splitBase_;
    entry->splitRem = op.splitRem_;
    entry->problemM = op.problemM_;
    entry->topK = op.topK_;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + op.coreIdx_), sizeof(DispatchFFNCombineUnpermuteTaskDebug));
}

template <typename Stage>
AICORE inline void DeviceDebug::UnpermuteWriteMetaDebug(Stage &op, uint32_t token, uint32_t topkIdx,
                                                        int32_t expandedRow, float prob, uint32_t batchStart,
                                                        uint32_t localToken)
{
    const uint64_t requiredBytes = DeviceDebug::UnpermuteMetaDebugOffset(op) -
                                   op.tilingData_->unpermuteTiling.unpermuteDebugOffset +
                                   DeviceDebug::UnpermuteMetaDebugTableBytes(op);
    if (op.tilingData_->unpermuteTiling.unpermuteDebugBytes < requiredBytes) {
        return;
    }
    __gm__ DispatchFFNCombineUnpermuteMetaDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineUnpermuteMetaDebug *>(op.workspaceGM_ +
                                                                        DeviceDebug::UnpermuteMetaDebugOffset(op));
    volatile __gm__ DispatchFFNCombineUnpermuteMetaDebug *entry = debug + op.coreIdx_;
    entry->magic = kUnpermuteMetaDebugMagic;
    entry->coreIdx = op.coreIdx_;
    entry->token = token;
    entry->topkIdx = topkIdx;
    entry->expandedRow = static_cast<uint32_t>(expandedRow);
    entry->valid = expandedRow >= 0 && static_cast<uint32_t>(expandedRow) < op.expandedRowsValid_ ? 1U : 0U;
    entry->probBits = op.FloatBits(prob);
    entry->batchStart = batchStart;
    entry->localToken = localToken;
    entry->expandedRowsValid = op.expandedRowsValid_;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + op.coreIdx_), sizeof(DispatchFFNCombineUnpermuteMetaDebug));
    op.metaDebugWritten_ = true;
}

template <typename Stage>
AICORE inline void DeviceDebug::UnpermuteWriteAccumDebug(Stage &op, uint32_t token, uint32_t col, uint32_t cols,
                                                         uint32_t validTopk, uint32_t topkProcessed,
                                                         int32_t firstExpandedRow, float firstProb)
{
    const uint64_t requiredBytes = DeviceDebug::UnpermuteAccumDebugOffset(op) -
                                   op.tilingData_->unpermuteTiling.unpermuteDebugOffset +
                                   DeviceDebug::UnpermuteAccumDebugTableBytes(op);
    if (op.tilingData_->unpermuteTiling.unpermuteDebugBytes < requiredBytes) {
        return;
    }
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    const float accSample = PtoGetValue<float, kUnpermuteVecTileElems>(op.ubAccOffset_, 0);
    __gm__ DispatchFFNCombineUnpermuteAccumDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineUnpermuteAccumDebug *>(op.workspaceGM_ +
                                                                         DeviceDebug::UnpermuteAccumDebugOffset(op));
    volatile __gm__ DispatchFFNCombineUnpermuteAccumDebug *entry = debug + op.coreIdx_;
    entry->magic = kUnpermuteAccumDebugMagic;
    entry->coreIdx = op.coreIdx_;
    entry->token = token;
    entry->col = col;
    entry->cols = cols;
    entry->validTopk = validTopk;
    entry->topkProcessed = topkProcessed;
    entry->firstExpandedRow = static_cast<uint32_t>(firstExpandedRow);
    entry->firstProbBits = op.FloatBits(firstProb);
    entry->accSample = accSample;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + op.coreIdx_), sizeof(DispatchFFNCombineUnpermuteAccumDebug));
    op.accumDebugWritten_ = true;
}

template <typename Stage>
AICORE inline void DeviceDebug::UnpermuteWriteOutputDebug(Stage &op, uint32_t token, uint32_t col, uint32_t cols,
                                                          uint32_t validTopk, uint32_t topkProcessed)
{
    const uint64_t requiredBytes = DeviceDebug::UnpermuteOutputDebugOffset(op) -
                                   op.tilingData_->unpermuteTiling.unpermuteDebugOffset +
                                   DeviceDebug::UnpermuteOutputDebugTableBytes(op);
    if (op.tilingData_->unpermuteTiling.unpermuteDebugBytes < requiredBytes) {
        return;
    }
    __gm__ DispatchFFNCombineUnpermuteOutputDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineUnpermuteOutputDebug *>(op.workspaceGM_ +
                                                                          DeviceDebug::UnpermuteOutputDebugOffset(op));
    volatile __gm__ DispatchFFNCombineUnpermuteOutputDebug *entry = debug + op.coreIdx_;
    entry->magic = kUnpermuteOutputDebugMagic;
    entry->coreIdx = op.coreIdx_;
    entry->token = token;
    entry->col = col;
    entry->cols = cols;
    entry->outputValue = 0U;
    entry->validTopk = validTopk;
    entry->topkProcessed = topkProcessed;
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + op.coreIdx_), sizeof(DispatchFFNCombineUnpermuteOutputDebug));
    op.outputDebugWritten_ = true;
}

template <typename Stage>
AICORE inline void DeviceDebug::UnpermuteWriteDoneDebug(const Stage &op)
{
    const uint64_t requiredBytes = DeviceDebug::UnpermuteDoneDebugOffset(op) -
                                   op.tilingData_->unpermuteTiling.unpermuteDebugOffset +
                                   DeviceDebug::UnpermuteDoneDebugTableBytes(op);
    if (op.tilingData_->unpermuteTiling.unpermuteDebugBytes < requiredBytes) {
        return;
    }
    __gm__ DispatchFFNCombineUnpermuteDoneDebug *debug =
        reinterpret_cast<__gm__ DispatchFFNCombineUnpermuteDoneDebug *>(op.workspaceGM_ +
                                                                        DeviceDebug::UnpermuteDoneDebugOffset(op));
    volatile __gm__ DispatchFFNCombineUnpermuteDoneDebug *entry = debug + op.coreIdx_;
    entry->magic = kUnpermuteDoneDebugMagic;
    entry->coreIdx = op.coreIdx_;
    entry->rank = op.rank_;
    entry->rankSize = op.rankSize_;
    entry->stageNum = op.stageNum_;
    entry->tokenStart = op.tokenStart_;
    entry->tokenCount = op.tokenCount_;
    entry->syncBeforeDone = 1U;
    entry->marker = 1U;
    entry->rows = op.rowsDone_;
    entry->values = op.valuesDone_;
    entry->bytes = op.bytesDone_;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(reinterpret_cast<__gm__ void *>(debug + op.coreIdx_), sizeof(DispatchFFNCombineUnpermuteDoneDebug));
}
#endif

template <typename Stage>
AICORE inline void DeviceDebug::DispatchFfnLogProcessDebug(const Stage &op)
{
#ifdef _DEBUG
    if ASCEND_IS_AIV {
        cce::printf("V8 DispatchFFNCombine::Process AIV block=%d sub=%d\n", int(get_block_idx()),
                    int(get_subblockid()));
    } else {
        cce::printf("V8 DispatchFFNCombine::Process AIC block=%d\n", int(get_block_idx()));
    }
#endif
}
} // namespace dispatch_ffn_combine_v8

#endif // DISPATCH_FFN_COMBINE_V8_DEVICE_DEBUG_IMPL_H
#endif // DISPATCH_FFN_COMBINE_V8_DEVICE_DEBUG_IMPLEMENTATION
