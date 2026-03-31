// a2a3/cce_stub.hpp
#pragma once

#include <cstddef>
#include <cstdint>

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

// Constants used in various instructions
constexpr int DSB_UB = 0;      // TFillPad.hpp
constexpr int ONLY_VALUE = 0;  // TRowMax.hpp, TRowMin.hpp
constexpr int PIPE_FIX = 0;    // TStore.hpp
constexpr int VA0 = 0;
constexpr int VA1 = 1;
constexpr int VA2 = 2;
constexpr int VA3 = 3;
constexpr int VA4 = 4;
constexpr int VA5 = 5;
constexpr int VA6 = 6;
constexpr int VA7 = 7;
// Bit manipulation functions (TMatmul.hpp)
inline int sbitset0(int val, int bit) { return val & ~(1 << bit); }  // Clear bit
inline int sbitset1(int val, int bit) { return val | (1 << bit); }   // Set bit

// Helper functions
inline int get_ctrl(...) { return 0; }       // TMatmul.hpp
inline int get_vms4_sr(...) { return 0; }    // TMrgSort.hpp
inline int get_imm(...) { return 0; }        // TSel.hpp

inline void copy_cbuf_to_bt(...) {}
inline void copy_cbuf_to_fbuf(...) {}
inline void copy_cbuf_to_gm(...) {}
inline void copy_cbuf_to_gm(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto srcStride, auto dstStride)
{
    pto::mocker::RecordCceCall(
        "copy_cbuf_to_gm",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("nBurst", nBurst),
         pto::mocker::MakeTraceArg("lenBurst", lenBurst), pto::mocker::MakeTraceArg("srcStride", srcStride),
         pto::mocker::MakeTraceArg("dstStride", dstStride)});
}
inline void copy_gm_to_cbuf(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto gmGap, auto l1Gap, auto pad)
{
    pto::mocker::RecordCceCall("copy_gm_to_cbuf",
                               {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
                                pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("nBurst", nBurst),
                                pto::mocker::MakeTraceArg("lenBurst", lenBurst),
                                pto::mocker::MakeTraceArg("gmGap", gmGap),
                                pto::mocker::MakeTraceArg("l1Gap", l1Gap),
                                pto::mocker::MakeTraceArg("pad", pad)});
}
inline void copy_gm_to_cbuf_multi_nd2nz_b16(auto dst, auto src, auto sid, auto ndNum, auto nValue, auto dValue,
                                            auto srcNdMatrixStride, auto srcDValue, auto dstNzC0Stride,
                                            auto dstNzNStride, auto dstNzMatrixStride)
{
    pto::mocker::RecordCceCall(
        "copy_gm_to_cbuf_multi_nd2nz_b16",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("ndNum", ndNum),
         pto::mocker::MakeTraceArg("nValue", nValue), pto::mocker::MakeTraceArg("dValue", dValue),
         pto::mocker::MakeTraceArg("srcNdMatrixStride", srcNdMatrixStride),
         pto::mocker::MakeTraceArg("srcDValue", srcDValue),
         pto::mocker::MakeTraceArg("dstNzC0Stride", dstNzC0Stride),
         pto::mocker::MakeTraceArg("dstNzNStride", dstNzNStride),
         pto::mocker::MakeTraceArg("dstNzMatrixStride", dstNzMatrixStride)});
}
inline void copy_gm_to_cbuf_multi_nd2nz_b32s(auto dst, auto src, auto sid, auto ndNum, auto nValue, auto dValue,
                                             auto srcNdMatrixStride, auto srcDValue, auto dstNzC0Stride,
                                             auto dstNzNStride, auto dstNzMatrixStride)
{
    pto::mocker::RecordCceCall(
        "copy_gm_to_cbuf_multi_nd2nz_b32s",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("ndNum", ndNum),
         pto::mocker::MakeTraceArg("nValue", nValue), pto::mocker::MakeTraceArg("dValue", dValue),
         pto::mocker::MakeTraceArg("srcNdMatrixStride", srcNdMatrixStride),
         pto::mocker::MakeTraceArg("srcDValue", srcDValue),
         pto::mocker::MakeTraceArg("dstNzC0Stride", dstNzC0Stride),
         pto::mocker::MakeTraceArg("dstNzNStride", dstNzNStride),
         pto::mocker::MakeTraceArg("dstNzMatrixStride", dstNzMatrixStride)});
}
inline void copy_gm_to_cbuf_multi_nd2nz_b8(auto dst, auto src, auto sid, auto ndNum, auto nValue, auto dValue,
                                           auto srcNdMatrixStride, auto srcDValue, auto dstNzC0Stride,
                                           auto dstNzNStride, auto dstNzMatrixStride)
{
    pto::mocker::RecordCceCall(
        "copy_gm_to_cbuf_multi_nd2nz_b8",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("ndNum", ndNum),
         pto::mocker::MakeTraceArg("nValue", nValue), pto::mocker::MakeTraceArg("dValue", dValue),
         pto::mocker::MakeTraceArg("srcNdMatrixStride", srcNdMatrixStride),
         pto::mocker::MakeTraceArg("srcDValue", srcDValue),
         pto::mocker::MakeTraceArg("dstNzC0Stride", dstNzC0Stride),
         pto::mocker::MakeTraceArg("dstNzNStride", dstNzNStride),
         pto::mocker::MakeTraceArg("dstNzMatrixStride", dstNzMatrixStride)});
}
inline void copy_gm_to_ubuf_align_b16(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding,
                                      auto rightPadding, auto gmGap, auto ubGap)
{
    pto::mocker::RecordCceCall(
        "copy_gm_to_ubuf_align_b16",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("nBurst", nBurst),
         pto::mocker::MakeTraceArg("lenBurst", lenBurst),
         pto::mocker::MakeTraceArg("leftPadding", leftPadding),
         pto::mocker::MakeTraceArg("rightPadding", rightPadding),
         pto::mocker::MakeTraceArg("gmGap", gmGap), pto::mocker::MakeTraceArg("ubGap", ubGap)});
}
inline void copy_gm_to_ubuf_align_b32(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding,
                                      auto rightPadding, auto gmGap, auto ubGap)
{
    pto::mocker::RecordCceCall(
        "copy_gm_to_ubuf_align_b32",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("nBurst", nBurst),
         pto::mocker::MakeTraceArg("lenBurst", lenBurst),
         pto::mocker::MakeTraceArg("leftPadding", leftPadding),
         pto::mocker::MakeTraceArg("rightPadding", rightPadding),
         pto::mocker::MakeTraceArg("gmGap", gmGap), pto::mocker::MakeTraceArg("ubGap", ubGap)});
}
inline void copy_gm_to_ubuf_align_b8(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding,
                                     auto rightPadding, auto gmGap, auto ubGap)
{
    pto::mocker::RecordCceCall(
        "copy_gm_to_ubuf_align_b8",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("nBurst", nBurst),
         pto::mocker::MakeTraceArg("lenBurst", lenBurst),
         pto::mocker::MakeTraceArg("leftPadding", leftPadding),
         pto::mocker::MakeTraceArg("rightPadding", rightPadding),
         pto::mocker::MakeTraceArg("gmGap", gmGap), pto::mocker::MakeTraceArg("ubGap", ubGap)});
}
inline void copy_matrix_cc_to_cbuf(...) {}
inline void copy_matrix_cc_to_gm(...) {}
inline void copy_matrix_cc_to_gm(auto dst, auto src, auto xmReg, auto xtReg)
{
    pto::mocker::RecordCceCall(
        "copy_matrix_cc_to_gm",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("xmReg", xmReg), pto::mocker::MakeTraceArg("xtReg", xtReg)});
}
inline void copy_ubuf_to_gm_align_b16(...) {}
inline void copy_ubuf_to_gm_align_b32(...) {}
inline void copy_ubuf_to_gm_align_b8(...) {}
inline void copy_ubuf_to_gm_align_b16(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding,
                                      auto rightPadding, auto ubGap, auto gmGap)
{
    pto::mocker::RecordCceCall(
        "copy_ubuf_to_gm_align_b16",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("nBurst", nBurst),
         pto::mocker::MakeTraceArg("lenBurst", lenBurst),
         pto::mocker::MakeTraceArg("leftPadding", leftPadding),
         pto::mocker::MakeTraceArg("rightPadding", rightPadding), pto::mocker::MakeTraceArg("ubGap", ubGap),
         pto::mocker::MakeTraceArg("gmGap", gmGap)});
}
inline void copy_ubuf_to_gm_align_b32(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding,
                                      auto rightPadding, auto ubGap, auto gmGap)
{
    pto::mocker::RecordCceCall(
        "copy_ubuf_to_gm_align_b32",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("nBurst", nBurst),
         pto::mocker::MakeTraceArg("lenBurst", lenBurst),
         pto::mocker::MakeTraceArg("leftPadding", leftPadding),
         pto::mocker::MakeTraceArg("rightPadding", rightPadding), pto::mocker::MakeTraceArg("ubGap", ubGap),
         pto::mocker::MakeTraceArg("gmGap", gmGap)});
}
inline void copy_ubuf_to_gm_align_b8(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding,
                                     auto rightPadding, auto ubGap, auto gmGap)
{
    pto::mocker::RecordCceCall(
        "copy_ubuf_to_gm_align_b8",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("nBurst", nBurst),
         pto::mocker::MakeTraceArg("lenBurst", lenBurst),
         pto::mocker::MakeTraceArg("leftPadding", leftPadding),
         pto::mocker::MakeTraceArg("rightPadding", rightPadding), pto::mocker::MakeTraceArg("ubGap", ubGap),
         pto::mocker::MakeTraceArg("gmGap", gmGap)});
}
inline void copy_ubuf_to_ubuf(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto srcGap, auto dstGap)
{
    pto::mocker::RecordCceCall(
        "copy_ubuf_to_ubuf",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("nBurst", nBurst),
         pto::mocker::MakeTraceArg("lenBurst", lenBurst), pto::mocker::MakeTraceArg("srcGap", srcGap),
         pto::mocker::MakeTraceArg("dstGap", dstGap)});
}
inline void create_cbuf_matrix(...) {}
inline void create_cbuf_matrix_bf16(...) {}
inline void dsb(...) {}
inline void ffts_cross_core_sync(...) {}
inline void img2colv2_cbuf_to_ca(...) {}
inline void img2colv2_cbuf_to_cb(...) {}
inline void load_cbuf_to_ca(...) {}
inline void load_cbuf_to_ca_transpose(...) {}
inline void load_cbuf_to_cb(...) {}
inline void load_cbuf_to_cb_transpose(...) {}
inline void mad(auto c, auto a, auto b, auto m, auto k, auto n, auto phase, auto kDirectionAlign, auto cmatrixSource,
                auto cmatrixInitVal)
{
    pto::mocker::RecordCceCall(
        "mad",
        {pto::mocker::MakeTraceArg("c", c), pto::mocker::MakeTraceArg("a", a), pto::mocker::MakeTraceArg("b", b),
         pto::mocker::MakeTraceArg("m", m), pto::mocker::MakeTraceArg("k", k), pto::mocker::MakeTraceArg("n", n),
         pto::mocker::MakeTraceArg("phase", phase),
         pto::mocker::MakeTraceArg("kDirectionAlign", kDirectionAlign),
         pto::mocker::MakeTraceArg("cmatrixSource", cmatrixSource),
         pto::mocker::MakeTraceArg("cmatrixInitVal", cmatrixInitVal)});
}
inline void pipe_barrier(...)
{
    pto::mocker::RecordCceCall("pipe_barrier");
}
inline void scatter_vnchwconv_b16(...) {}
inline void scatter_vnchwconv_b16(auto dst, auto src, auto repeat, auto dstStride, auto srcStride)
{
    pto::mocker::RecordCceCall(
        "scatter_vnchwconv_b16",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("repeat", repeat), pto::mocker::MakeTraceArg("dstStride", dstStride),
         pto::mocker::MakeTraceArg("srcStride", srcStride)});
}
inline void scatter_vnchwconv_b32(...) {}
inline void scatter_vnchwconv_b32(auto dst, auto src, auto repeat, auto dstStride, auto srcStride)
{
    pto::mocker::RecordCceCall(
        "scatter_vnchwconv_b32",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("repeat", repeat), pto::mocker::MakeTraceArg("dstStride", dstStride),
         pto::mocker::MakeTraceArg("srcStride", srcStride)});
}
inline void scatter_vnchwconv_b8(...) {}
inline void scatter_vnchwconv_b8(auto dst, auto src, auto repeat, auto dstStride, auto srcStride, auto dstHighHalf,
                                 auto srcHighHalf)
{
    pto::mocker::RecordCceCall(
        "scatter_vnchwconv_b8",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("repeat", repeat), pto::mocker::MakeTraceArg("dstStride", dstStride),
         pto::mocker::MakeTraceArg("srcStride", srcStride),
         pto::mocker::MakeTraceArg("dstHighHalf", dstHighHalf),
         pto::mocker::MakeTraceArg("srcHighHalf", srcHighHalf)});
}
inline void set_atomic_add(...) {}
inline void set_atomic_bf16(...) {}
inline void set_atomic_f16(...) {}
inline void set_atomic_f32(...) {}
inline void set_atomic_none(...) {}
inline void set_atomic_s16(...) {}
inline void set_atomic_s32(...) {}
inline void set_atomic_s8(...) {}
inline void set_cmpmask(...)
{
    pto::mocker::RecordCceCall("set_cmpmask");
}
inline void set_ctrl(...) {}
inline void set_deqscale(...) {}
inline void set_ffts_base_addr(...) {}
inline void set_flag(...)
{
    pto::mocker::RecordCceCall("set_flag");
}
inline void set_fmatrix(...) {}
inline void set_fmatrix_b(...) {}
inline void set_fpc(...) {}
inline void set_mask_count()
{
    pto::mocker::RecordCceCall("set_mask_count");
}
inline void set_mask_norm()
{
    pto::mocker::RecordCceCall("set_mask_norm");
}
inline void set_mov_pad_val(auto value)
{
    pto::mocker::RecordCceCall("set_mov_pad_val", {pto::mocker::MakeTraceArg("value", value)});
}
inline void set_nd_para(...) {}
inline void set_padding(...)
{
    pto::mocker::RecordCceCall("set_padding");
}
inline void set_quant_pre(...) {}
inline void set_va_reg_sb(...) {}
inline void set_vector_mask(auto mask0, auto mask1)
{
    pto::mocker::RecordCceCall("set_vector_mask", {pto::mocker::MakeTraceArg("mask0", mask0),
                                                   pto::mocker::MakeTraceArg("mask1", mask1)});
}
inline void vadd(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vadd",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("src1BlockStride", src1BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride),
         pto::mocker::MakeTraceArg("src1RepeatStride", src1RepeatStride)});
}
inline void vadds(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                  auto dstRepeatStride, auto src0RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vadds",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride)});
}
inline void vand(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vand",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("src1BlockStride", src1BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride),
         pto::mocker::MakeTraceArg("src1RepeatStride", src1RepeatStride)});
}
inline void vbitsort(...) {}
inline void vbrcb(...) {}
inline void vcadd(...) {}
inline void vcgadd(...) {}
inline void vcgmax(...) {}
inline void vcgmin(...) {}
inline void vcmax(...) {}
inline void vcmin(...) {}
#define PTO_MOCKER_DEFINE_VCMPV_STUB(name)                                                                        \
    inline void name(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,    \
                     auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)   \
    {                                                                                                             \
        pto::mocker::RecordCceCall(                                                                               \
            #name,                                                                                                \
            {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),                    \
             pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeat", repeat),              \
             pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),                                        \
             pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),                                      \
             pto::mocker::MakeTraceArg("src1BlockStride", src1BlockStride),                                      \
             pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),                                      \
             pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride),                                    \
             pto::mocker::MakeTraceArg("src1RepeatStride", src1RepeatStride)});                                  \
    }

#define PTO_MOCKER_DEFINE_VCMPVS_STUB(name)                                                                       \
    inline void name(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,    \
                     auto dstRepeatStride, auto srcRepeatStride)                                                  \
    {                                                                                                             \
        pto::mocker::RecordCceCall(                                                                               \
            #name,                                                                                                \
            {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),                    \
             pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeat", repeat),              \
             pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),                                        \
             pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),                                      \
             pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),                                      \
             pto::mocker::MakeTraceArg("srcRepeatStride", srcRepeatStride)});                                    \
    }

PTO_MOCKER_DEFINE_VCMPV_STUB(vcmpv_eq)
PTO_MOCKER_DEFINE_VCMPV_STUB(vcmpv_ge)
PTO_MOCKER_DEFINE_VCMPV_STUB(vcmpv_gt)
PTO_MOCKER_DEFINE_VCMPV_STUB(vcmpv_le)
PTO_MOCKER_DEFINE_VCMPV_STUB(vcmpv_lt)
PTO_MOCKER_DEFINE_VCMPV_STUB(vcmpv_ne)
PTO_MOCKER_DEFINE_VCMPVS_STUB(vcmpvs_eq)
PTO_MOCKER_DEFINE_VCMPVS_STUB(vcmpvs_ge)
PTO_MOCKER_DEFINE_VCMPVS_STUB(vcmpvs_gt)
PTO_MOCKER_DEFINE_VCMPVS_STUB(vcmpvs_le)
PTO_MOCKER_DEFINE_VCMPVS_STUB(vcmpvs_lt)
PTO_MOCKER_DEFINE_VCMPVS_STUB(vcmpvs_ne)

#undef PTO_MOCKER_DEFINE_VCMPV_STUB
#undef PTO_MOCKER_DEFINE_VCMPVS_STUB
inline void vconv_bf162f32(...) {}
inline void vconv_bf162s32a(...) {}
inline void vconv_bf162s32c(...) {}
inline void vconv_bf162s32f(...) {}
inline void vconv_bf162s32r(...) {}
inline void vconv_bf162s32z(...) {}
inline void vconv_deq(...) {}
inline void vconv_f162f32(...) {}
inline void vconv_f162s16a(...) {}
inline void vconv_f162s16c(...) {}
inline void vconv_f162s16f(...) {}
inline void vconv_f162s16r(...) {}
inline void vconv_f162s16z(...) {}
inline void vconv_f162s32a(...) {}
inline void vconv_f162s32c(...) {}
inline void vconv_f162s32f(...) {}
inline void vconv_f162s32r(...) {}
inline void vconv_f162s32z(...) {}
inline void vconv_f162s8a(...) {}
inline void vconv_f162s8c(...) {}
inline void vconv_f162s8f(...) {}
inline void vconv_f162s8r(...) {}
inline void vconv_f162s8z(...) {}
inline void vconv_f162u8a(...) {}
inline void vconv_f162u8c(...) {}
inline void vconv_f162u8f(...) {}
inline void vconv_f162u8r(...) {}
inline void vconv_f162u8z(...) {}
inline void vconv_f322bf16a(...) {}
inline void vconv_f322bf16c(...) {}
inline void vconv_f322bf16f(...) {}
inline void vconv_f322bf16r(...) {}
inline void vconv_f322bf16z(...) {}
inline void vconv_f322f16(...) {}
inline void vconv_f322f16a(...) {}
inline void vconv_f322f16c(...) {}
inline void vconv_f322f16f(...) {}
inline void vconv_f322f16o(...) {}
inline void vconv_f322f16r(...) {}
inline void vconv_f322f16z(...) {}
inline void vconv_f322f32a(...) {}
inline void vconv_f322f32c(...) {}
#define PTO_MOCKER_DEFINE_VCONV_STUB(name)                                                                        \
    inline void name(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride, \
                     auto srcRepeatStride)                                                                        \
    {                                                                                                             \
        pto::mocker::RecordCceCall(                                                                               \
            #name,                                                                                                \
            {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),                      \
             pto::mocker::MakeTraceArg("repeat", repeat),                                                        \
             pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),                                        \
             pto::mocker::MakeTraceArg("srcBlockStride", srcBlockStride),                                        \
             pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),                                      \
             pto::mocker::MakeTraceArg("srcRepeatStride", srcRepeatStride)});                                    \
    }

PTO_MOCKER_DEFINE_VCONV_STUB(vconv_f322f32f)
inline void vconv_f322f32r(...) {}
PTO_MOCKER_DEFINE_VCONV_STUB(vconv_f322f32z)
inline void vconv_f322s16a(...) {}
inline void vconv_f322s16c(...) {}
inline void vconv_f322s16f(...) {}
inline void vconv_f322s16r(...) {}
inline void vconv_f322s16z(...) {}
inline void vconv_f322s32a(...) {}
inline void vconv_f322s32c(...) {}
inline void vconv_f322s32f(...) {}
PTO_MOCKER_DEFINE_VCONV_STUB(vconv_f322s32r)
inline void vconv_f322s32z(...) {}
inline void vconv_f322s64a(...) {}
inline void vconv_f322s64c(...) {}
inline void vconv_f322s64f(...) {}
inline void vconv_f322s64r(...) {}
inline void vconv_f322s64z(...) {}
inline void vconv_s162f16(...) {}
inline void vconv_s162f16a(...) {}
inline void vconv_s162f16c(...) {}
inline void vconv_s162f16f(...) {}
inline void vconv_s162f16r(...) {}
inline void vconv_s162f16z(...) {}
inline void vconv_s162f32(...) {}
PTO_MOCKER_DEFINE_VCONV_STUB(vconv_s322f32)
inline void vconv_s322f32a(...) {}
inline void vconv_s322f32c(...) {}
inline void vconv_s322f32f(...) {}
inline void vconv_s322f32r(...) {}
inline void vconv_s322f32z(...) {}
inline void vconv_s322s16(...) {}
inline void vconv_s322s64(...) {}
inline void vconv_s642f32a(...) {}
inline void vconv_s642f32c(...) {}
inline void vconv_s642f32f(...) {}
inline void vconv_s642f32r(...) {}
inline void vconv_s642f32z(...) {}
inline void vconv_s642s32(...) {}
inline void vconv_s82f16(...) {}
inline void vconv_u82f16(...) {}

#undef PTO_MOCKER_DEFINE_VCONV_STUB
inline void vcopy(...) {}
inline void vcopy(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                  auto srcRepeatStride)
{
    pto::mocker::RecordCceCall(
        "vcopy",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("repeat", repeat),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("srcBlockStride", srcBlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("srcRepeatStride", srcRepeatStride)});
}
inline void vdiv(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vdiv",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("src1BlockStride", src1BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride),
         pto::mocker::MakeTraceArg("src1RepeatStride", src1RepeatStride)});
}
inline void vector_dup(auto dst, auto src, auto repeats, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                       auto srcRepeatStride)
{
    pto::mocker::RecordCceCall(
        "vector_dup",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("repeats", repeats), pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("srcBlockStride", srcBlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("srcRepeatStride", srcRepeatStride)});
}
inline void vgather(...) {}
inline void vgatherb(...) {}
inline void set_l3d_rpt(...)
{
    pto::mocker::RecordCceCall("set_l3d_rpt");
}
inline void vmax(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vmax",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("src1BlockStride", src1BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride),
         pto::mocker::MakeTraceArg("src1RepeatStride", src1RepeatStride)});
}
inline void vmaxs(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                  auto dstRepeatStride, auto src0RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vmaxs",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride)});
}
inline void vmin(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vmin",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("src1BlockStride", src1BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride),
         pto::mocker::MakeTraceArg("src1RepeatStride", src1RepeatStride)});
}
inline void vmins(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                  auto dstRepeatStride, auto src0RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vmins",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride)});
}
inline void vmrgsort4(...) {}
inline void vmul(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vmul",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("src1BlockStride", src1BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride),
         pto::mocker::MakeTraceArg("src1RepeatStride", src1RepeatStride)});
}
inline void vmuls(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                  auto dstRepeatStride, auto src0RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vmuls",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride)});
}
inline void vabs(auto dst, auto src, auto repeats, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                 auto srcRepeatStride)
{
    pto::mocker::RecordCceCall(
        "vabs",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("repeats", repeats), pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("srcBlockStride", srcBlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("srcRepeatStride", srcRepeatStride)});
}
inline void vaxpy(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                  auto dstRepeatStride, auto src0RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vaxpy",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride)});
}
inline void vexp(auto dst, auto src, auto repeats, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                 auto srcRepeatStride)
{
    pto::mocker::RecordCceCall(
        "vexp",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("repeats", repeats), pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("srcBlockStride", srcBlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("srcRepeatStride", srcRepeatStride)});
}
inline void vln(auto dst, auto src, auto repeats, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                auto srcRepeatStride)
{
    pto::mocker::RecordCceCall(
        "vln",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("repeats", repeats), pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("srcBlockStride", srcBlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("srcRepeatStride", srcRepeatStride)});
}
inline void vlrelu(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                   auto dstRepeatStride, auto src0RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vlrelu",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride)});
}
inline void vnot(auto dst, auto src, auto repeats, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                 auto srcRepeatStride)
{
    pto::mocker::RecordCceCall(
        "vnot",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("repeats", repeats), pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("srcBlockStride", srcBlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("srcRepeatStride", srcRepeatStride)});
}
inline void vor(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vor",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("src1BlockStride", src1BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride),
         pto::mocker::MakeTraceArg("src1RepeatStride", src1RepeatStride)});
}
inline void vreducev2(...) {}
inline void vrelu(auto dst, auto src, auto repeats, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                  auto srcRepeatStride)
{
    pto::mocker::RecordCceCall(
        "vrelu",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("repeats", repeats), pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("srcBlockStride", srcBlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("srcRepeatStride", srcRepeatStride)});
}
inline void vrsqrt(auto dst, auto src, auto repeats, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                   auto srcRepeatStride)
{
    pto::mocker::RecordCceCall(
        "vrsqrt",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("repeats", repeats), pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("srcBlockStride", srcBlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("srcRepeatStride", srcRepeatStride)});
}
inline void vsel(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride, auto mode)
{
    pto::mocker::RecordCceCall(
        "vsel",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeat", repeat),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("src1BlockStride", src1BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride),
         pto::mocker::MakeTraceArg("src1RepeatStride", src1RepeatStride), pto::mocker::MakeTraceArg("mode", mode)});
}
inline void vshl(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                 auto dstRepeatStride, auto src0RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vshl",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride)});
}
inline void vshr(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                 auto dstRepeatStride, auto src0RepeatStride, auto isArithmetic = false)
{
    pto::mocker::RecordCceCall(
        "vshr",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride),
         pto::mocker::MakeTraceArg("isArithmetic", isArithmetic)});
}
inline void vsqrt(auto dst, auto src, auto repeats, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                  auto srcRepeatStride)
{
    pto::mocker::RecordCceCall(
        "vsqrt",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("repeats", repeats), pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("srcBlockStride", srcBlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("srcRepeatStride", srcRepeatStride)});
}
inline void vsub(auto dst, auto src0, auto src1, auto repeats, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    pto::mocker::RecordCceCall(
        "vsub",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src0", src0),
         pto::mocker::MakeTraceArg("src1", src1), pto::mocker::MakeTraceArg("repeats", repeats),
         pto::mocker::MakeTraceArg("dstBlockStride", dstBlockStride),
         pto::mocker::MakeTraceArg("src0BlockStride", src0BlockStride),
         pto::mocker::MakeTraceArg("src1BlockStride", src1BlockStride),
         pto::mocker::MakeTraceArg("dstRepeatStride", dstRepeatStride),
         pto::mocker::MakeTraceArg("src0RepeatStride", src0RepeatStride),
         pto::mocker::MakeTraceArg("src1RepeatStride", src1RepeatStride)});
}
inline void wait_flag(...)
{
    pto::mocker::RecordCceCall("wait_flag");
}
inline void wait_flag_dev(...)
{
    pto::mocker::RecordCceCall("wait_flag_dev");
}

// cce namespace stubs for TPrint.hpp
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
