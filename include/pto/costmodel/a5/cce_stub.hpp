#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <pto/costmodel/a5/compat.hpp>
#include <pto/costmodel/a2a3/cce_stub.hpp>

inline void wait_intra_block(...) {}
inline void set_intra_block(...) {}

inline void set_loop1_stride_outtoub(auto config)
{
    pto::mocker::RecordCceCall("set_loop1_stride_outtoub", {pto::mocker::MakeTraceArg("config", config)});
}

inline void set_loop2_stride_outtoub(auto config)
{
    pto::mocker::RecordCceCall("set_loop2_stride_outtoub", {pto::mocker::MakeTraceArg("config", config)});
}

inline void set_loop_size_outtoub(auto config)
{
    pto::mocker::RecordCceCall("set_loop_size_outtoub", {pto::mocker::MakeTraceArg("config", config)});
}

inline void set_loop1_stride_ubtoout(auto config)
{
    pto::mocker::RecordCceCall("set_loop1_stride_ubtoout", {pto::mocker::MakeTraceArg("config", config)});
}

inline void set_loop2_stride_ubtoout(auto config)
{
    pto::mocker::RecordCceCall("set_loop2_stride_ubtoout", {pto::mocker::MakeTraceArg("config", config)});
}

inline void set_loop_size_ubtoout(auto config)
{
    pto::mocker::RecordCceCall("set_loop_size_ubtoout", {pto::mocker::MakeTraceArg("config", config)});
}

inline void set_loop3_para(auto config)
{
    pto::mocker::RecordCceCall("set_loop3_para", {pto::mocker::MakeTraceArg("config", config)});
}

inline void set_mte2_nz_para(...) {}
inline void set_loop_size_outtol1(...) {}
inline void set_loop1_stride_outtol1(...) {}
inline void set_loop2_stride_outtol1(...) {}
inline void set_pad_val_outtol1(...) {}
inline void set_channel_para(...) {}

inline void copy_gm_to_cbuf_multi_nd2nz(...) {}
inline void copy_gm_to_cbuf_multi_dn2nz(...) {}
inline void copy_gm_to_cbuf_align_v2(...) {}
inline void copy_gm_to_ubuf_align_v2(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding,
                                     auto rightPadding, auto enableUBPad, auto l2CacheCtl, auto gmStride, auto ubStride)
{
    pto::mocker::RecordCceCall(
        "copy_gm_to_ubuf_align_v2",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("nBurst", nBurst),
         pto::mocker::MakeTraceArg("lenBurst", lenBurst),
         pto::mocker::MakeTraceArg("leftPadding", leftPadding),
         pto::mocker::MakeTraceArg("rightPadding", rightPadding),
         pto::mocker::MakeTraceArg("enableUBPad", enableUBPad),
         pto::mocker::MakeTraceArg("l2CacheCtl", l2CacheCtl), pto::mocker::MakeTraceArg("gmStride", gmStride),
         pto::mocker::MakeTraceArg("ubStride", ubStride)});
}

inline void copy_ubuf_to_gm_align_v2(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto ctrl,
                                     auto dstStride, auto srcStride)
{
    pto::mocker::RecordCceCall(
        "copy_ubuf_to_gm_align_v2",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("src", src),
         pto::mocker::MakeTraceArg("sid", sid), pto::mocker::MakeTraceArg("nBurst", nBurst),
         pto::mocker::MakeTraceArg("lenBurst", lenBurst), pto::mocker::MakeTraceArg("ctrl", ctrl),
         pto::mocker::MakeTraceArg("dstStride", dstStride), pto::mocker::MakeTraceArg("srcStride", srcStride)});
}

template <typename Reg, typename Ptr, typename Offset, typename Mode, typename... Extra>
inline void vlds(Reg & /*dst*/, Ptr src, Offset offset, Mode mode, Extra... extra)
{
    pto::mocker::RecordCceCall(
        "vlds",
        {pto::mocker::MakeTraceArg("src", src), pto::mocker::MakeTraceArg("offset", offset),
         pto::mocker::MakeTraceArg("mode", mode), pto::mocker::MakeTraceArg("extraCount", sizeof...(extra))});
}

template <typename Dist>
inline uint64_t ToTraceDist(Dist dist)
{
    using Decayed = std::remove_cv_t<std::remove_reference_t<Dist>>;
    if constexpr (requires { Decayed::value; }) {
        return static_cast<uint64_t>(Decayed::value);
    } else {
        return static_cast<uint64_t>(dist);
    }
}

template <typename Reg, typename Ptr, typename Offset, typename Dist, typename Mask, typename... Extra>
inline void vsts(Reg & /*src*/, Ptr dst, Offset offset, Dist dist, Mask &preg, Extra... extra)
{
    pto::mocker::RecordCceCall(
        "vsts",
        {pto::mocker::MakeTraceArg("dst", dst), pto::mocker::MakeTraceArg("offset", offset),
         pto::mocker::MakeTraceArg("dist", ToTraceDist(dist)),
         pto::mocker::MakeTraceArg("maskCount", preg.count),
         pto::mocker::MakeTraceArg("extraCount", sizeof...(extra))});
}

template <typename RegDst, typename RegSrc0, typename RegSrc1, typename Mask, typename Mode>
inline void vadd(RegDst & /*dst*/, RegSrc0 & /*src0*/, RegSrc1 & /*src1*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vadd",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc0, typename RegSrc1, typename Mask, typename Mode>
inline void vsub(RegDst & /*dst*/, RegSrc0 & /*src0*/, RegSrc1 & /*src1*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vsub",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc0, typename RegSrc1, typename Mask, typename Mode>
inline void vmul(RegDst & /*dst*/, RegSrc0 & /*src0*/, RegSrc1 & /*src1*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vmul",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc0, typename RegSrc1, typename Mask, typename Mode>
inline void vmax(RegDst & /*dst*/, RegSrc0 & /*src0*/, RegSrc1 & /*src1*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vmax",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc0, typename RegSrc1, typename Mask, typename Mode>
inline void vmin(RegDst & /*dst*/, RegSrc0 & /*src0*/, RegSrc1 & /*src1*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vmin",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Scalar, typename Mask, typename Mode>
inline void vadds(RegDst & /*dst*/, RegSrc & /*src0*/, Scalar scalar, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vadds",
        {pto::mocker::MakeTraceArg("src1", scalar), pto::mocker::MakeTraceArg("maskCount", preg.count),
         pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Scalar, typename Mask, typename Mode>
inline void vmuls(RegDst & /*dst*/, RegSrc & /*src0*/, Scalar scalar, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vmuls",
        {pto::mocker::MakeTraceArg("src1", scalar), pto::mocker::MakeTraceArg("maskCount", preg.count),
         pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Scalar, typename Mask>
inline void vmuls(RegDst & /*dst*/, RegSrc & /*src0*/, Scalar scalar, Mask &preg)
{
    pto::mocker::RecordCceCall(
        "vmuls",
        {pto::mocker::MakeTraceArg("src1", scalar), pto::mocker::MakeTraceArg("maskCount", preg.count)});
}

template <typename RegDst, typename RegSrc, typename Scalar, typename Mask, typename Mode>
inline void vmaxs(RegDst & /*dst*/, RegSrc & /*src0*/, Scalar scalar, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vmaxs",
        {pto::mocker::MakeTraceArg("src1", scalar), pto::mocker::MakeTraceArg("maskCount", preg.count),
         pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Scalar, typename Mask, typename Mode>
inline void vmins(RegDst & /*dst*/, RegSrc & /*src0*/, Scalar scalar, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vmins",
        {pto::mocker::MakeTraceArg("src1", scalar), pto::mocker::MakeTraceArg("maskCount", preg.count),
         pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Scalar, typename Mask, typename Mode>
inline void vlrelu(RegDst & /*dst*/, RegSrc & /*src0*/, Scalar scalar, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vlrelu",
        {pto::mocker::MakeTraceArg("src1", scalar), pto::mocker::MakeTraceArg("maskCount", preg.count),
         pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename Scalar, typename Mask, typename Mode>
inline void vdup(RegDst & /*dst*/, Scalar scalar, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vdup",
        {pto::mocker::MakeTraceArg("src1", scalar), pto::mocker::MakeTraceArg("maskCount", preg.count),
         pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Mask, typename Mode>
inline void vexp(RegDst & /*dst*/, RegSrc & /*src*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vexp",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Mask, typename Mode>
inline void vnot(RegDst & /*dst*/, RegSrc & /*src*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vnot",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Mask, typename Mode>
inline void vrelu(RegDst & /*dst*/, RegSrc & /*src*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vrelu",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Mask, typename Mode>
inline void vsqrt(RegDst & /*dst*/, RegSrc & /*src*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vsqrt",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Mask, typename Mode>
inline void vabs(RegDst & /*dst*/, RegSrc & /*src*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vabs",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Mask, typename Mode>
inline void vln(RegDst & /*dst*/, RegSrc & /*src*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vln",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc0, typename RegSrc1, typename Mask>
inline void vand(RegDst & /*dst*/, RegSrc0 & /*src0*/, RegSrc1 & /*src1*/, Mask &preg)
{
    pto::mocker::RecordCceCall("vand", {pto::mocker::MakeTraceArg("maskCount", preg.count)});
}

template <typename RegDst, typename RegSrc0, typename RegSrc1, typename Mask, typename Mode>
inline void vand(RegDst & /*dst*/, RegSrc0 & /*src0*/, RegSrc1 & /*src1*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vand",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc0, typename RegSrc1, typename Mask>
inline void vdiv(RegDst & /*dst*/, RegSrc0 & /*src0*/, RegSrc1 & /*src1*/, Mask &preg)
{
    pto::mocker::RecordCceCall("vdiv", {pto::mocker::MakeTraceArg("maskCount", preg.count)});
}

template <typename RegDst, typename RegSrc0, typename RegSrc1, typename Mask, typename Mode>
inline void vdiv(RegDst & /*dst*/, RegSrc0 & /*src0*/, RegSrc1 & /*src1*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vdiv",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Scalar, typename Mask>
inline void vaxpy(RegDst & /*dst*/, RegSrc & /*src0*/, Scalar scalar, Mask &preg)
{
    pto::mocker::RecordCceCall(
        "vaxpy",
        {pto::mocker::MakeTraceArg("src1", scalar), pto::mocker::MakeTraceArg("maskCount", preg.count)});
}

template <typename RegDst, typename RegSrc, typename Mask, typename... Extra>
inline void vcvt(RegDst & /*dst*/, RegSrc & /*src0*/, Mask &preg, Extra... extra)
{
    pto::mocker::RecordCceCall(
        "vcvt",
        {pto::mocker::MakeTraceArg("maskCount", preg.count),
         pto::mocker::MakeTraceArg("extraCount", sizeof...(extra))});
}

template <typename RegDst, typename RegSrc0, typename RegSrc1, typename Mask, typename Mode>
inline void vxor(RegDst & /*dst*/, RegSrc0 & /*src0*/, RegSrc1 & /*src1*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vxor",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc0, typename RegSrc1, typename Mask, typename Mode>
inline void vshl(RegDst & /*dst*/, RegSrc0 & /*src0*/, RegSrc1 & /*src1*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vshl",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc0, typename RegSrc1, typename Mask, typename Mode>
inline void vshr(RegDst & /*dst*/, RegSrc0 & /*src0*/, RegSrc1 & /*src1*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vshr",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Scalar, typename Mask, typename Mode>
inline void vshls(RegDst & /*dst*/, RegSrc & /*src0*/, Scalar scalar, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vshls",
        {pto::mocker::MakeTraceArg("src1", scalar), pto::mocker::MakeTraceArg("maskCount", preg.count),
         pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc, typename Scalar, typename Mask, typename Mode>
inline void vshrs(RegDst & /*dst*/, RegSrc & /*src0*/, Scalar scalar, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vshrs",
        {pto::mocker::MakeTraceArg("src1", scalar), pto::mocker::MakeTraceArg("maskCount", preg.count),
         pto::mocker::MakeTraceArg("mode", mode)});
}

template <typename RegDst, typename RegSrc0, typename RegSrc1, typename Mask, typename Mode>
inline void vor(RegDst & /*dst*/, RegSrc0 & /*src0*/, RegSrc1 & /*src1*/, Mask &preg, Mode mode)
{
    pto::mocker::RecordCceCall(
        "vor",
        {pto::mocker::MakeTraceArg("maskCount", preg.count), pto::mocker::MakeTraceArg("mode", mode)});
}
