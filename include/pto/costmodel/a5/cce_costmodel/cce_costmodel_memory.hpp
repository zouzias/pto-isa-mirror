/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#pragma once

#include <pto/costmodel/arch_config.hpp>
#include <pto/costmodel/trace.hpp>

inline void set_loop1_stride_outtol1(auto config)
{
    ::pto::mocker::RecordCceCall("set_loop1_stride_outtol1", 0, config);
}
inline void set_loop2_stride_outtol1(auto config)
{
    ::pto::mocker::RecordCceCall("set_loop2_stride_outtol1", 0, config);
}
inline void set_loop_size_outtol1(auto config) { ::pto::mocker::RecordCceCall("set_loop_size_outtol1", 0, config); }
inline void set_loop1_stride_outtoub(auto config)
{
    ::pto::mocker::RecordCceCall("set_loop1_stride_outtoub", 0, config);
}
inline void set_loop2_stride_outtoub(auto config)
{
    ::pto::mocker::RecordCceCall("set_loop2_stride_outtoub", 0, config);
}
inline void set_loop_size_outtoub(auto config) { ::pto::mocker::RecordCceCall("set_loop_size_outtoub", 0, config); }
inline void set_loop1_stride_ubtoout(auto config)
{
    ::pto::mocker::RecordCceCall("set_loop1_stride_ubtoout", 0, config);
}
inline void set_loop2_stride_ubtoout(auto config)
{
    ::pto::mocker::RecordCceCall("set_loop2_stride_ubtoout", 0, config);
}
inline void set_loop_size_ubtoout(auto config) { ::pto::mocker::RecordCceCall("set_loop_size_ubtoout", 0, config); }
inline void set_mte2_nz_para(auto config) { ::pto::mocker::RecordCceCall("set_mte2_nz_para", 0, config); }
inline void set_loop3_para(auto config) { ::pto::mocker::RecordCceCall("set_loop3_para", 0, config); }
inline void set_channel_para(auto config) { ::pto::mocker::RecordCceCall("set_channel_para", 0, config); }

inline void copy_gm_to_ubuf_align_v2(
    auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto paddingCtl,
    auto l2CacheCtl, auto srcStride, auto dstStride)
{
    const uint64_t bytes = static_cast<uint64_t>(nBurst) * static_cast<uint64_t>(lenBurst);
    ::pto::mocker::evaluator::ApplyHillBandwidthFromEnv();
    const double bandwidth = ::pto::mocker::evaluator::gHillBandwidth.BwEff(
        ::pto::mocker::evaluator::PipeKey::GM_TO_UB, bytes, ::pto::mocker::evaluator::gActiveCoreCount);
    const uint64_t cycles = bandwidth <= 0.0 ?
                                0 :
                                static_cast<uint64_t>(
                                    (static_cast<long double>(bytes) / ::pto::mocker::evaluator::kBytesPerGb) /
                                    bandwidth * ::pto::mocker::evaluator::GetDefaultArchConfig().frequency_hz);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::GM_TO_UB, "copy_gm_to_ubuf_align_v2", cycles, dst, src, sid, nBurst,
        lenBurst, leftPadding, rightPadding, paddingCtl, l2CacheCtl, srcStride, dstStride);
}

inline void copy_gm_to_cbuf_align_v2(
    auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto leftPadding, auto rightPadding, auto paddingCtl,
    auto l2CacheCtl, auto srcStride, auto dstStride)
{
    const uint64_t bytes = static_cast<uint64_t>(nBurst) * static_cast<uint64_t>(lenBurst);
    ::pto::mocker::evaluator::ApplyHillBandwidthFromEnv();
    const double bandwidth = ::pto::mocker::evaluator::gHillBandwidth.BwEff(
        ::pto::mocker::evaluator::PipeKey::GM_TO_L1, bytes, ::pto::mocker::evaluator::gActiveCoreCount);
    const uint64_t cycles = bandwidth <= 0.0 ?
                                0 :
                                static_cast<uint64_t>(
                                    (static_cast<long double>(bytes) / ::pto::mocker::evaluator::kBytesPerGb) /
                                    bandwidth * ::pto::mocker::evaluator::GetDefaultArchConfig().frequency_hz);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::GM_TO_L1, "copy_gm_to_cbuf_align_v2", cycles, dst, src, sid, nBurst,
        lenBurst, leftPadding, rightPadding, paddingCtl, l2CacheCtl, srcStride, dstStride);
}

inline void copy_ubuf_to_gm_align_v2(
    auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto l2CacheCtl, auto dstStride, auto srcStride)
{
    const uint64_t bytes = static_cast<uint64_t>(nBurst) * static_cast<uint64_t>(lenBurst);
    ::pto::mocker::evaluator::ApplyHillBandwidthFromEnv();
    const double bandwidth = ::pto::mocker::evaluator::gHillBandwidth.BwEff(
        ::pto::mocker::evaluator::PipeKey::UB_TO_GM, bytes, ::pto::mocker::evaluator::gActiveCoreCount);
    const uint64_t cycles = bandwidth <= 0.0 ?
                                0 :
                                static_cast<uint64_t>(
                                    (static_cast<long double>(bytes) / ::pto::mocker::evaluator::kBytesPerGb) /
                                    bandwidth * ::pto::mocker::evaluator::GetDefaultArchConfig().frequency_hz);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::UB_TO_GM, "copy_ubuf_to_gm_align_v2", cycles, dst, src, sid, nBurst,
        lenBurst, l2CacheCtl, dstStride, srcStride);
}

inline void copy_matrix_cc_to_gm(auto dst, auto src, auto xmReg, auto xtReg)
{
    const uint64_t rows = (static_cast<uint64_t>(xmReg) >> 16) & 0xffffULL;
    const uint64_t cols = (static_cast<uint64_t>(xmReg) >> 4) & 0xfffULL;
    const uint64_t bytes = rows * cols * 2;
    ::pto::mocker::evaluator::ApplyHillBandwidthFromEnv();
    const double bandwidth = ::pto::mocker::evaluator::gHillBandwidth.BwEff(
        ::pto::mocker::evaluator::PipeKey::L0C_TO_GM, bytes, ::pto::mocker::evaluator::gActiveCoreCount);
    const uint64_t cycles = bandwidth <= 0.0 ?
                                0 :
                                static_cast<uint64_t>(
                                    (static_cast<long double>(bytes) / ::pto::mocker::evaluator::kBytesPerGb) /
                                    bandwidth * ::pto::mocker::evaluator::GetDefaultArchConfig().frequency_hz);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::L0C_TO_GM, "copy_matrix_cc_to_gm", cycles, dst, src, xmReg, xtReg);
}

inline void copy_ubuf_to_ubuf(auto dst, auto src, auto nBurst, auto lenBurst, auto srcGap, auto dstGap)
{
    const uint64_t bytes =
        static_cast<uint64_t>(nBurst) * static_cast<uint64_t>(lenBurst) * ::pto::mocker::evaluator::kBlockBytes;
    ::pto::mocker::evaluator::ApplyHillBandwidthFromEnv();
    const double bandwidth = ::pto::mocker::evaluator::gHillBandwidth.BwEff(
        ::pto::mocker::evaluator::PipeKey::UB_TO_UB, bytes, ::pto::mocker::evaluator::gActiveCoreCount);
    const uint64_t cycles = bandwidth <= 0.0 ?
                                0 :
                                static_cast<uint64_t>(
                                    (static_cast<long double>(bytes) / ::pto::mocker::evaluator::kBytesPerGb) /
                                    bandwidth * ::pto::mocker::evaluator::GetDefaultArchConfig().frequency_hz);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::UB_TO_UB, "copy_ubuf_to_ubuf", cycles, dst, src, nBurst, lenBurst, srcGap,
        dstGap);
}

inline void set_mov_pad_val(auto value) { ::pto::mocker::RecordCceCall("set_mov_pad_val", 1, value); }
inline void set_pad_val_outtol1(auto value) { ::pto::mocker::RecordCceCall("set_pad_val_outtol1", 1, value); }
inline void set_pad_val_outtoub(auto value) { ::pto::mocker::RecordCceCall("set_pad_val_outtoub", 1, value); }
