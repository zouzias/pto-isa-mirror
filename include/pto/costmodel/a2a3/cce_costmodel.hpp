#pragma once

#include <cstddef>
#include <cstdint>

#include <pto/costmodel/arch_config.hpp>
#include <pto/costmodel/trace.hpp>

namespace pto {

enum QuantMode_t
{
    NoQuant = 0,
    F322F16 = 1,
    F322BF16 = 16,
    DEQF16 = 5,
    VDEQF16 = 4,
    QF322B8_PRE = 24,
    QF322HIF8_PRE = 25,
    QF322FP8_PRE = 26,
    QF322F32_PRE = 27,
    QF322F16_PRE = 32,
    QF322BF16_PRE = 34,
    QS322BF16_PRE = 35,
    VQF322B8_PRE = 23,
    VQF322HIF8_PRE = 28,
    VQF322F16_PRE = 33,
    VQF322BF16_PRE = 36,
    VQF322FP8_PRE = 37,
    VQF322F32_PRE = 38,
    REQ8 = 3,
    VREQ8 = 2,
    VQS322BF16_PRE = 39,
    VSHIFTS322S16 = 12,
    SHIFTS322S16 = 13,
};

} // namespace pto

constexpr int DSB_UB = 0;
constexpr int ONLY_VALUE = 0;
constexpr int PIPE_FIX = 0;
constexpr int VA0 = 0;
constexpr int VA1 = 1;
constexpr int VA2 = 2;
constexpr int VA3 = 3;
constexpr int VA4 = 4;
constexpr int VA5 = 5;
constexpr int VA6 = 6;
constexpr int VA7 = 7;

inline int sbitset0(int val, int bit) { return val & ~(1 << bit); }
inline int sbitset1(int val, int bit) { return val | (1 << bit); }

inline int get_ctrl(...) { return 0; }
inline int get_vms4_sr(...) { return 0; }
inline int get_imm(...) { return 0; }


inline const ::pto::mocker::evaluator::ArchConfig &CurrentArch()
{
    return ::pto::mocker::evaluator::GetDefaultArchConfig();
}

inline uint64_t EstimateBandwidthCycles(uint64_t bytes, ::pto::mocker::evaluator::PipeKey key)
{
    const auto &arch = CurrentArch();
    const double bandwidth = arch.bandwidth[key];
    if (bandwidth <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>((static_cast<long double>(bytes) / ::pto::mocker::evaluator::kBytesPerGb) /
                                 static_cast<long double>(bandwidth) * arch.frequency_hz);
}

inline void FlushPipeTail(::pto::mocker::evaluator::PipeKey pipe)
{
    ::pto::mocker::FlushPendingTail(pipe);
}

inline void FlushTailsForPipe(auto pipe)
{
    switch (pipe) {
        case PIPE_V:
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::VECTOR);
            break;
        case PIPE_M:
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::CUBE);
            break;
        case PIPE_MTE1:
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L1_TO_L0A);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L1_TO_L0B);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L1_TO_BT);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L1_TO_FB);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L1_FILL);
            break;
        case PIPE_MTE2:
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::GM_TO_UB);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::GM_TO_L1);
            break;
        case PIPE_MTE3:
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::UB_TO_GM);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L1_TO_GM);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L0C_TO_GM);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L0C_TO_L1);
            break;
        case PIPE_ALL:
            ::pto::mocker::FlushAllPendingTails();
            break;
        default:
            break;
    }
}

inline uint64_t EstimateLinearCycles(::pto::mocker::evaluator::PipeKey pipe, uint64_t repeat, uint64_t head = 6,
                                     uint64_t slope = 2, uint64_t tail = 0)
{
    uint64_t cycles = slope * repeat;
    if (::pto::mocker::IsPipeQueueEmpty(pipe)) {
        cycles += head;
    }
    ::pto::mocker::SetLastCceTail(pipe, tail);
    return cycles;
}

inline uint64_t EstimateLinearCycles(uint64_t repeat, uint64_t head = 6, uint64_t slope = 2, uint64_t tail = 0)
{
    return EstimateLinearCycles(::pto::mocker::evaluator::PipeKey::VECTOR, repeat, head, slope, tail);
}

inline uint64_t EstimateConstCycles(uint64_t cycles = 1)
{
    return cycles;
}

inline uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    return (x + y - 1) / y;
}

inline uint64_t ExtractBits(uint64_t value, uint32_t shift, uint64_t mask)
{
    return (value >> shift) & mask;
}

// Temporary common latency model for vconv_*; the detailed behavior is not fully understood yet.
inline uint64_t _EstimateVconvCycles(uint64_t repeat)
{
    return EstimateLinearCycles(repeat, 14, 2, 18);
}

inline void copy_cbuf_to_gm(...) {}
inline void copy_matrix_cc_to_gm(...) {}
inline void copy_ubuf_to_gm_align_b16(...) {}
inline void copy_ubuf_to_gm_align_b32(...) {}
inline void copy_ubuf_to_gm_align_b8(...) {}
inline void scatter_vnchwconv_b16(...) {}
inline void scatter_vnchwconv_b32(...) {}
inline void scatter_vnchwconv_b8(...) {}

inline void set_l3d_rpt(auto rptConfig)
{
    ::pto::mocker::RecordCceCall("set_l3d_rpt", 0, rptConfig);
}


inline void copy_cbuf_to_bt(auto dst, auto src, auto convControl, auto nBurst, auto lenBurst, auto srcStride, auto dstStride)
{
const uint64_t bytes = nBurst * lenBurst * 64;
        const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L1_TO_BT);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::L1_TO_BT, "copy_cbuf_to_bt", cycles, dst, src,
                           convControl, nBurst, lenBurst, srcStride, dstStride);
}
inline void copy_cbuf_to_fbuf(auto dst, auto src, auto nBurst, auto lenBurst, auto srcStride, auto dstStride)
{
const uint64_t bytes = nBurst * lenBurst * 128;
        const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L1_TO_FB);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::L1_TO_FB, "copy_cbuf_to_fbuf", cycles, dst, src, nBurst,
                           lenBurst, srcStride, dstStride);
}
inline void copy_cbuf_to_gm(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto srcStride, auto dstStride)
{
const uint64_t bytes = nBurst * lenBurst * ::pto::mocker::evaluator::kBlockBytes;
        const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L1_TO_GM);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::L1_TO_GM, "copy_cbuf_to_gm", cycles, dst, src, sid,
                           nBurst, lenBurst, srcStride, dstStride);
}
inline void copy_gm_to_cbuf(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto gmGap, auto l1Gap, auto pad)
{
const uint64_t bytes = nBurst * lenBurst * ::pto::mocker::evaluator::kBlockBytes;
        const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::GM_TO_L1);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::GM_TO_L1, "copy_gm_to_cbuf", cycles, dst, src, sid,
                           nBurst, lenBurst, gmGap, l1Gap, pad);
}
inline void copy_gm_to_cbuf_multi_nd2nz_b16(auto dst, auto src, auto sid, auto ndNum, auto nValue, auto dValue, auto srcNdMatrixStride, auto srcDValue, auto dstNzC0Stride, auto dstNzNStride, auto dstNzMatrixStride)
{
const uint64_t bytes = ndNum * nValue * dValue * 2;
        const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::GM_TO_L1);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::GM_TO_L1, "copy_gm_to_cbuf_multi_nd2nz_b16", cycles,
                           dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                           dstNzNStride, dstNzMatrixStride);
}
inline void copy_gm_to_cbuf_multi_nd2nz_b32s(auto dst, auto src, auto sid, auto ndNum, auto nValue, auto dValue, auto srcNdMatrixStride, auto srcDValue, auto dstNzC0Stride, auto dstNzNStride, auto dstNzMatrixStride)
{
const uint64_t bytes = ndNum * nValue * dValue * 4;
        const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::GM_TO_L1);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::GM_TO_L1, "copy_gm_to_cbuf_multi_nd2nz_b32s", cycles,
                           dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                           dstNzNStride, dstNzMatrixStride);
}
inline void copy_gm_to_cbuf_multi_nd2nz_b8(auto dst, auto src, auto sid, auto ndNum, auto nValue, auto dValue, auto srcNdMatrixStride, auto srcDValue, auto dstNzC0Stride, auto dstNzNStride, auto dstNzMatrixStride)
{
const uint64_t bytes = ndNum * nValue * dValue;
        const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::GM_TO_L1);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::GM_TO_L1, "copy_gm_to_cbuf_multi_nd2nz_b8", cycles,
                           dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                           dstNzNStride, dstNzMatrixStride);
}
inline void copy_gm_to_ubuf_align_b16(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto gmGap, auto ubGap)
{
    const uint64_t cycles = EstimateBandwidthCycles(nBurst * lenBurst, ::pto::mocker::evaluator::PipeKey::GM_TO_UB);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::GM_TO_UB, "copy_gm_to_ubuf_align_b16", cycles, dst,
                           src, sid, nBurst, lenBurst, leftPadding, rightPadding, gmGap, ubGap);
}
inline void copy_gm_to_ubuf_align_b32(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto gmGap, auto ubGap)
{
    const uint64_t cycles = EstimateBandwidthCycles(nBurst * lenBurst, ::pto::mocker::evaluator::PipeKey::GM_TO_UB);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::GM_TO_UB, "copy_gm_to_ubuf_align_b32", cycles, dst,
                           src, sid, nBurst, lenBurst, leftPadding, rightPadding, gmGap, ubGap);
}
inline void copy_gm_to_ubuf_align_b8(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto gmGap, auto ubGap)
{
    const uint64_t cycles = EstimateBandwidthCycles(nBurst * lenBurst, ::pto::mocker::evaluator::PipeKey::GM_TO_UB);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::GM_TO_UB, "copy_gm_to_ubuf_align_b8", cycles, dst,
                           src, sid, nBurst, lenBurst, leftPadding, rightPadding, gmGap, ubGap);
}
inline void copy_matrix_cc_to_cbuf(auto dst, auto src, auto sid, auto nSize, auto mSize, auto dstStrideD, auto srcStride, auto reserved, auto quantPre, auto reluMode, auto flag0, auto flag1)
{
const uint64_t bytes = nSize * mSize * 2;
        const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L0C_TO_L1);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::L0C_TO_L1, "copy_matrix_cc_to_cbuf", cycles, dst, src,
                           sid, nSize, mSize, dstStrideD, srcStride, reserved, quantPre, reluMode, flag0, flag1);
}
inline void copy_matrix_cc_to_gm(auto dst, auto src, auto xmReg, auto xtReg)
{
const uint64_t rows = ExtractBits(xmReg, 16, 0xffffULL);
    const uint64_t cols = ExtractBits(xmReg, 4, 0xfffULL);
    const uint64_t bytes = rows * cols * 2;
        const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L0C_TO_GM);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::L0C_TO_GM, "copy_matrix_cc_to_gm", cycles, dst, src,
                           xmReg, xtReg);
}
inline void copy_ubuf_to_gm_align_b16(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto ubGap, auto gmGap)
{
    const uint64_t cycles = EstimateBandwidthCycles(nBurst * lenBurst, ::pto::mocker::evaluator::PipeKey::UB_TO_GM);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::UB_TO_GM, "copy_ubuf_to_gm_align_b16", cycles, dst,
                           src, sid, nBurst, lenBurst, leftPadding, rightPadding, ubGap, gmGap);
}
inline void copy_ubuf_to_gm_align_b32(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto ubGap, auto gmGap)
{
    const uint64_t cycles = EstimateBandwidthCycles(nBurst * lenBurst, ::pto::mocker::evaluator::PipeKey::UB_TO_GM);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::UB_TO_GM, "copy_ubuf_to_gm_align_b32", cycles, dst,
                           src, sid, nBurst, lenBurst, leftPadding, rightPadding, ubGap, gmGap);
}
inline void copy_ubuf_to_gm_align_b8(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto ubGap, auto gmGap)
{
    const uint64_t cycles = EstimateBandwidthCycles(nBurst * lenBurst, ::pto::mocker::evaluator::PipeKey::UB_TO_GM);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::UB_TO_GM, "copy_ubuf_to_gm_align_b8", cycles, dst,
                           src, sid, nBurst, lenBurst, leftPadding, rightPadding, ubGap, gmGap);
}
inline void copy_ubuf_to_ubuf(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto srcGap, auto dstGap)
{
const uint64_t bytes = nBurst * lenBurst * ::pto::mocker::evaluator::kBlockBytes;
        const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::UB_TO_UB);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::UB_TO_UB, "copy_ubuf_to_ubuf", cycles, dst, src, sid,
                           nBurst, lenBurst, srcGap, dstGap);
}
inline void create_cbuf_matrix(auto dst, auto repeatConfig, auto value)
{
const uint64_t repeatTimes = ExtractBits(repeatConfig, 0, 0x7fffULL);
    const uint64_t blockLen = ExtractBits(repeatConfig, 16, 0xffffULL);
    const uint64_t bytes = repeatTimes * blockLen * ::pto::mocker::evaluator::kBlockBytes;
        const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L1_FILL);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::L1_FILL, "create_cbuf_matrix", cycles, dst,
                           repeatConfig, value);
}
inline void create_cbuf_matrix_bf16(auto dst, auto repeatConfig, auto value)
{
const uint64_t repeatTimes = ExtractBits(repeatConfig, 0, 0x7fffULL);
    const uint64_t blockLen = ExtractBits(repeatConfig, 16, 0xffffULL);
    const uint64_t bytes = repeatTimes * blockLen * ::pto::mocker::evaluator::kBlockBytes;
        const uint64_t cycles = EstimateBandwidthCycles(bytes, ::pto::mocker::evaluator::PipeKey::L1_FILL);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::L1_FILL, "create_cbuf_matrix_bf16", cycles, dst,
                           repeatConfig, value);
}
inline void dsb(auto barrierType)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("dsb", cycles, barrierType);
}
inline void ffts_cross_core_sync(auto srcPipe, auto msg)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("ffts_cross_core_sync", cycles, srcPipe, msg);
}
inline void img2colv2_cbuf_to_ca(auto dst, auto src, auto stepK, auto stepM, auto posK, auto posM, auto strideW, auto strideH, auto filterW, auto filterH, auto dilationW, auto dilationH, auto highFilterW, auto highFilterH, auto transpose, auto fmatrixCtrl, auto channelSize)
{
    const uint64_t cycles = EstimateBandwidthCycles(stepK * stepM * 2, ::pto::mocker::evaluator::PipeKey::L1_TO_L0A);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::L1_TO_L0A, "img2colv2_cbuf_to_ca", cycles, dst, src,
                           stepK, stepM, posK, posM, strideW, strideH, filterW, filterH, dilationW, dilationH,
                           highFilterW, highFilterH, transpose, fmatrixCtrl, channelSize);
}
inline void img2colv2_cbuf_to_cb(auto dst, auto src, auto stepK, auto stepM, auto posK, auto posM, auto strideW, auto strideH, auto filterW, auto filterH, auto dilationW, auto dilationH, auto highFilterW, auto highFilterH, auto transpose, auto fmatrixCtrl, auto channelSize)
{
    const uint64_t cycles = EstimateBandwidthCycles(stepK * stepM * 2, ::pto::mocker::evaluator::PipeKey::L1_TO_L0B);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::L1_TO_L0B, "img2colv2_cbuf_to_cb", cycles, dst, src,
                           stepK, stepM, posK, posM, strideW, strideH, filterW, filterH, dilationW, dilationH,
                           highFilterW, highFilterH, transpose, fmatrixCtrl, channelSize);
}
inline void load_cbuf_to_ca(auto dst, auto src, auto baseIdx, auto repeat, auto srcStride, auto sid, auto transpose)
{
    const uint64_t cycles = EstimateBandwidthCycles(repeat * 16 * ::pto::mocker::evaluator::kBlockBytes, ::pto::mocker::evaluator::PipeKey::L1_TO_L0A);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::L1_TO_L0A, "load_cbuf_to_ca", cycles, dst, src,
                           baseIdx, repeat, srcStride, sid, transpose);
}
inline void load_cbuf_to_ca_transpose(auto dst, auto src, auto baseIdx, auto repeat, auto srcStride, auto dstStride, auto addrCalMode, auto dstFracStride)
{
    const uint64_t cycles = EstimateBandwidthCycles(repeat * 16 * ::pto::mocker::evaluator::kBlockBytes, ::pto::mocker::evaluator::PipeKey::L1_TO_L0A);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::L1_TO_L0A, "load_cbuf_to_ca_transpose", cycles, dst,
                           src, baseIdx, repeat, srcStride, dstStride, addrCalMode, dstFracStride);
}
inline void load_cbuf_to_cb(auto dst, auto src, auto baseIdx, auto repeat, auto srcStride, auto dstStride, auto sid, auto transpose, auto addrCalMode)
{
    const uint64_t cycles = EstimateBandwidthCycles(repeat * 16 * ::pto::mocker::evaluator::kBlockBytes, ::pto::mocker::evaluator::PipeKey::L1_TO_L0B);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::L1_TO_L0B, "load_cbuf_to_cb", cycles, dst, src,
                           baseIdx, repeat, srcStride, dstStride, sid, transpose, addrCalMode);
}
inline void load_cbuf_to_cb_transpose(auto dst, auto src, auto baseIdx, auto repeat, auto srcStride, auto dstStride, auto addrCalMode, auto dstFracStride)
{
    const uint64_t cycles = EstimateBandwidthCycles(repeat * 16 * ::pto::mocker::evaluator::kBlockBytes, ::pto::mocker::evaluator::PipeKey::L1_TO_L0B);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::L1_TO_L0B, "load_cbuf_to_cb_transpose", cycles, dst,
                           src, baseIdx, repeat, srcStride, dstStride, addrCalMode, dstFracStride);
}
inline void mad(auto c, auto a, auto b, auto m, auto k, auto n, auto phase, auto kDirectionAlign, auto cmatrixSource, auto cmatrixInitVal)
{
const uint64_t mTiles = CeilDiv(m, 16);
    const uint64_t kTiles = CeilDiv(k, 16);
    const uint64_t nTiles = CeilDiv(n, 16);
        const uint64_t cycles = EstimateLinearCycles(::pto::mocker::evaluator::PipeKey::CUBE, mTiles * kTiles * nTiles);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::CUBE, "mad", cycles, c, a, b, m, k, n, phase, kDirectionAlign, cmatrixSource, cmatrixInitVal);
}
inline void pipe_barrier(auto pipe)
{
    FlushTailsForPipe(pipe);
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("pipe_barrier", cycles, pipe);
}
inline void scatter_vnchwconv_b16(auto dst, auto src, auto repeat, auto dstStride, auto srcStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "scatter_vnchwconv_b16", cycles, dst, src, repeat, dstStride, srcStride);
}
inline void scatter_vnchwconv_b32(auto dst, auto src, auto repeat, auto dstStride, auto srcStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "scatter_vnchwconv_b32", cycles, dst, src, repeat, dstStride, srcStride);
}
inline void scatter_vnchwconv_b8(auto dst, auto src, auto repeat, auto dstStride, auto srcStride, auto dstHighHalf, auto srcHighHalf)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "scatter_vnchwconv_b8", cycles, dst, src, repeat, dstStride, srcStride, dstHighHalf,
                        srcHighHalf);
}
inline void set_atomic_add()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_add", cycles);
}
inline void set_atomic_bf16()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_bf16", cycles);
}
inline void set_atomic_f16()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_f16", cycles);
}
inline void set_atomic_f32()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_f32", cycles);
}
inline void set_atomic_none()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_none", cycles);
}
inline void set_atomic_s16()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_s16", cycles);
}
inline void set_atomic_s32()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_s32", cycles);
}
inline void set_atomic_s8()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_s8", cycles);
}
inline void set_cmpmask(auto cmpMaskPtr)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_cmpmask", cycles, cmpMaskPtr);
}
inline void set_ctrl(auto ctrl)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_ctrl", cycles, ctrl);
}
inline void set_deqscale(auto scale)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_deqscale", cycles, scale);
}
inline void set_ffts_base_addr(auto fftsAddr)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_ffts_base_addr", cycles, fftsAddr);
}
inline void set_flag(auto srcPipe, auto dstPipe, auto token)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_flag", cycles, srcPipe, dstPipe, token);
}
inline void set_fmatrix(auto regFmatrix)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_fmatrix", cycles, regFmatrix);
}
inline void set_fmatrix_b(auto regFmatrix)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_fmatrix_b", cycles, regFmatrix);
}
inline void set_fpc(auto deqTensorAddr)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_fpc", cycles, deqTensorAddr);
}
inline void set_mask_count()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_mask_count", cycles);
}
inline void set_mask_norm()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_mask_norm", cycles);
}
inline void set_mov_pad_val(auto value)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_mov_pad_val", cycles, value);
}
inline void set_nd_para(auto ndParaSPR)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_nd_para", cycles, ndParaSPR);
}
inline void set_padding(auto paddingValue)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_padding", cycles, paddingValue);
}
inline void set_quant_pre(auto preQuantScalar)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_quant_pre", cycles, preQuantScalar);
}
inline void set_va_reg_sb(auto vaReg, auto addrArray)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_va_reg_sb", cycles, vaReg, addrArray);
}
inline void set_vector_mask(auto mask0, auto mask1)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_vector_mask", cycles, mask0, mask1);
}
inline void vabs(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 1, 16);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vabs", cycles, dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride,
                        srcRepeatStride);
}
inline void vadd(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 18);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vadd", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride,
                        dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vadds(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 1, 18);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vadds", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, dstRepeatStride,
                        src0RepeatStride);
}
inline void vand(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vand", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride,
                        dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vaxpy(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vaxpy", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, dstRepeatStride,
                        src0RepeatStride);
}
inline void vbitsort(auto dst, auto src, auto idx, auto repeat)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vbitsort", cycles, dst, src, idx, repeat);
}
inline void vbrcb(auto dst, auto src, auto dstBlockStride, auto dstRepeatStride, auto repeat)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 0, 0, 18);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vbrcb", cycles, dst, src, dstBlockStride, dstRepeatStride, repeat);
}
inline void vcadd(auto dst, auto src, auto repeat, auto dstRepeatStride, auto srcBlockStride, auto srcRepeatStride, auto mode)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 7, 32);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcadd", cycles, dst, src, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride, mode);
}
inline void vcgadd(auto dst, auto src, auto repeat, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 24);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcgadd", cycles, dst, src, repeat, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcgmax(auto dst, auto src, auto repeat, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 17);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcgmax", cycles, dst, src, repeat, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcgmin(auto dst, auto src, auto repeat, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 17);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcgmin", cycles, dst, src, repeat, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmax(auto dst, auto src, auto repeat, auto dstRepeatStride, auto srcBlockStride, auto srcRepeatStride, auto mode)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmax", cycles, dst, src, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride, mode);
}
inline void vcmin(auto dst, auto src, auto repeat, auto dstRepeatStride, auto srcBlockStride, auto srcRepeatStride, auto mode)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmin", cycles, dst, src, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride, mode);
}
inline void vcmpv_eq(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpv_eq", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride,
                        src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmpv_ge(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 2, 22);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpv_ge", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride,
                        src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmpv_gt(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpv_gt", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride,
                        src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmpv_le(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpv_le", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride,
                        src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmpv_lt(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpv_lt", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride,
                        src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmpv_ne(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 2, 22);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpv_ne", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride,
                        src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmpvs_eq(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpvs_eq", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vcmpvs_ge(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpvs_ge", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vcmpvs_gt(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpvs_gt", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vcmpvs_le(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpvs_le", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vcmpvs_lt(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpvs_lt", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vcmpvs_ne(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmpvs_ne", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162f32(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162f32", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_deq(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_deq", cycles, dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride,
                        srcRepeatStride);
}
inline void vconv_f162f32(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162f32", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16o(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16o", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f32a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f32a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f32c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f32c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f32f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f32f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f32r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f32r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f32z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f32z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s16a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s16a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s16c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s16c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s16f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s16f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s16r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s16r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s16z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s16z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s32a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s32a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s32c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s32c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s32f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s32f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s32r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s32r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s32z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s32z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s64a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s64a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s64c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s64c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s64f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s64f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s64r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s64r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s64z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s64z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f16(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f16", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f16a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f16a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f16c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f16c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f16f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f16f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f16r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f16r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f16z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f16z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f32(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f32", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322f32(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322f32", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322f32a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322f32a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322f32c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322f32c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322f32f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322f32f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322f32r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322f32r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322f32z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322f32z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322s16(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322s16", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322s64(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322s64", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s642f32a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s642f32a", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s642f32c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s642f32c", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s642f32f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s642f32f", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s642f32r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s642f32r", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s642f32z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s642f32z", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s642s32(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s642s32", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s82f16(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s82f16", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_s42f16(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s42f16", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vconv_u82f16(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_u82f16", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
                        dstRepeatStride, srcRepeatStride);
}
inline void vcopy(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 11, 1, 13);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcopy", cycles, dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride,
                        srcRepeatStride);
}
inline void vdiv(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 8, 25);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vdiv", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride,
                        dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vector_dup(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 11, 1, 13);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vector_dup", cycles, dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride,
                        srcRepeatStride);
}
inline void vexp(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 4, 24);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vexp", cycles, dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride,
                        srcRepeatStride);
}
inline void vgather(auto dst, auto offset, auto srcBaseAddr, auto dstRepeatStride, auto repeat)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vgather", cycles, dst, offset, srcBaseAddr, dstRepeatStride, repeat);
}
inline void vgatherb(auto dst, auto offset, auto srcBaseAddr, auto dstRepeatStride, auto dstBlockStride, auto repeat)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vgatherb", cycles, dst, offset, srcBaseAddr, dstRepeatStride, dstBlockStride, repeat);
}
inline void vln(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vln", cycles, dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride,
                        srcRepeatStride);
}
inline void vlrelu(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vlrelu", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride,
                        dstRepeatStride, src0RepeatStride);
}
inline void vmax(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 2, 16);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmax", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride,
                        dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vmaxs(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmaxs", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, dstRepeatStride,
                        src0RepeatStride);
}
inline void vmin(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 2, 16);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmin", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride,
                        dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vmins(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 16);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmins", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, dstRepeatStride,
                        src0RepeatStride);
}
inline void vmrgsort4(auto dst, auto addrArray, auto count, auto config)
{
    const uint64_t cycles = EstimateLinearCycles(ExtractBits(config, 0, 0xffULL));
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmrgsort4", cycles, dst, addrArray, count, config);
}
inline void vmul(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 2, 19);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmul", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride,
                        dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vmuls(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 19);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmuls", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, dstRepeatStride,
                        src0RepeatStride);
}
inline void vnot(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vnot", cycles, dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride,
                        srcRepeatStride);
}
inline void vor(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vor", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride,
                        dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vreducev2(auto dst, auto src0, auto src1, auto repeat, auto src0BlockStride, auto modeOrMaskPattern, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 14, 20);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vreducev2", cycles, dst, src0, src1, repeat, src0BlockStride, modeOrMaskPattern,
                        src0RepeatStride, src1RepeatStride);
}
inline void vrelu(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vrelu", cycles, dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride,
                        srcRepeatStride);
}
inline void vrsqrt(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vrsqrt", cycles, dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride,
                        srcRepeatStride);
}
inline void vsel(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride, auto mode)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 2, 14);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vsel", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride,
                        dstRepeatStride, src0RepeatStride, src1RepeatStride, mode);
}
inline void vshl(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vshl", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, dstRepeatStride,
                        src0RepeatStride);
}
inline void vshr(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto isArithmetic = false)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vshr", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, dstRepeatStride,
                        src0RepeatStride, isArithmetic);
}
inline void vsqrt(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 2, 25);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vsqrt", cycles, dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride,
                        srcRepeatStride);
}
inline void vsub(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 2, 18);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vsub", cycles, dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride,
                        dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void wait_flag(auto srcPipe, auto dstPipe, auto token)
{
    FlushTailsForPipe(srcPipe);
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("wait_flag", cycles, srcPipe, dstPipe, token);
}
inline void wait_flag_dev(auto flagId)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("wait_flag_dev", cycles, flagId);
}

namespace cce {
inline void printf(...) {}
inline void print(...) {}
inline void print_str(...) {}
inline void print_int(...) {}
inline void print_float(...) {}
inline void print_half(...) {}
inline void print_hex(...) {}
inline void print_matrix(...) {}
inline void print_tensor(...) {}
inline void print_ubuf(...) {}
inline void print_cbuf(...) {}
inline void print_gm(...) {}
} // namespace cce
