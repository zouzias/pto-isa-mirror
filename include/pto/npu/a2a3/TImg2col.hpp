/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TIMG2COL_HPP
#define TIMG2COL_HPP

namespace pto {

template <typename FmatrixMode>
__tf__ PTO_INTERNAL void SetFmatrix(Img2colTileConfig &cfg)
{
    if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_AUTO || FmatrixMode == SetFmatrixMode::FMATRIX_B_AUTO) {
        uint64_t regFmatrix = 0;
        regFmatrix |= uint64_t(cfg.fmapW & 0xFFFF);

        uint32_t l1ShiftBit = 16;
        regFmatrix |= uint64_t(cfg.fmapH & 0xFFFF) << l1ShiftBit;

        uint32_t padNumber = 4;
        uint32_t padListShiftBit = 8;
        uint32_t padListShiftBase = 32;

        for(uint32_t i = 0; i < padNumber; i++) {
            regFmatrix |= uint64_t(cfg.padList[i] & 0xFF) << (padListShiftBase + i * padListShiftBit);
        }
    }
    if constexpr(FmatrixMode == SetFmatrixMode::FMATRIX_A_AUTO) {
        set_fmatrix(regFmatrix);
    } else if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_B_AUTO) {
        set_fmatrix_b(regFmatrix);
    }
}

template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
AICORE void TIMG2COL_IMPL(TileData &dst, ConvTileData &src, uint16_t posM, uint16_t posK, Img2colTileConfig &cfg)
{
    using SrcType = std::conditional_t<(sizeof(typename ConvTileData::DType) == 2), half, typename ConvTileData::DType>;
    using DstType = std::conditional_t<(sizeof(typename TileData::DType) == 2), half, typename TileData::DType>;
    static_assert(std::is_same_v<DstType, int8_t> || std::is_same_v<DstType, half> ||
                  std::is_same_v<DstType, bfloat16_t> || std::is_same_v<DstType, float>,
                  "TExtract: Invalid data type.");
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)__cce_get_tile_ptr(src.data());
    __cb__ DstType *dstAddr = (__cb__ DstType *)__cce_get_tile_ptr(dst.data());

    if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_AUTO || FmatrixMode == SetFmatrixMode::FMATRIX_B_AUTO) {
        SetFmatrix<FmatrixMode>(cfg);
    }

    bool fmatrixCtrl = (FmatrixMode == SetFmatrixMode::FMATRIX_B_AUTO) ||
                       (FmatrixMode == SetFmatrixMode::FMATRIX_B_MANUAL);

    uint16_t stepM = CeilAlignment(dst.GetValidRow(), FRACTAL_NZ_ROW);
    uint16_t stepK = CeilAlignment(dst.GetValidCol(), c0Size);

    bool filterFlagW = (cfg.filterW > 255);
    bool filterFlagH = (cfg.filterH > 255);
    uint8_t filterW = cfg.filterW & 0xFF;
    uint8_t filterH = cfg.filterH & 0xFF;

    img2colv2_cbuf_to_ca(dstAddr, srcAddr, stepK, stepM, posK, posM, cfg.strideW, cfg.strideH, filterW, filterH,
                cfg.dilationW, cfg.dilationH, filterFlagW, filterFlagH, cfg.transpose, fmatrixCtrl, cfg.channelSize);
}
}  // namespace pto
#endif  // TIMG2COL_HPP
