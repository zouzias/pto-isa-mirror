/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_INSTR_SHARED_OVERLOADS_HPP
#define PTO_INSTR_SHARED_OVERLOADS_HPP

#include <cstdint>

#define PTO_DEFINE_TSTORE_OVERLOADS()                                                                                \
    template <typename TileData, typename GlobalData, typename... WaitEvents>                                        \
    PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)                               \
    {                                                                                                                \
        TSYNC(events...);                                                                                            \
        MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, AtomicType::AtomicNone), dst, src);         \
        return {};                                                                                                   \
    }                                                                                                                \
    template <STPhase Phase, typename TileData, typename GlobalData, typename... WaitEvents>                         \
    PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)                               \
    {                                                                                                                \
        TSYNC(events...);                                                                                            \
        MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, AtomicType::AtomicNone, Phase), dst, src);  \
        return {};                                                                                                   \
    }                                                                                                                \
    template <typename TileData, typename GlobalData, AtomicType atomicType, typename... WaitEvents>                 \
    PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)                               \
    {                                                                                                                \
        TSYNC(events...);                                                                                            \
        MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType), dst, src);                     \
        return {};                                                                                                   \
    }                                                                                                                \
    template <STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType, typename... WaitEvents>  \
    PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)                               \
    {                                                                                                                \
        TSYNC(events...);                                                                                            \
        MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, Phase), dst, src);              \
        return {};                                                                                                   \
    }                                                                                                                \
    template <                                                                                                       \
        typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,                      \
        ReluPreMode reluPreMode, typename... WaitEvents>                                                             \
    PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)                               \
    {                                                                                                                \
        TSYNC(events...);                                                                                            \
        MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode), dst, src);        \
        return {};                                                                                                   \
    }                                                                                                                \
    template <                                                                                                       \
        STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,       \
        ReluPreMode reluPreMode, typename... WaitEvents>                                                             \
    PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)                               \
    {                                                                                                                \
        TSYNC(events...);                                                                                            \
        MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode, Phase), dst, src); \
        return {};                                                                                                   \
    }                                                                                                                \
    template <                                                                                                       \
        typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,                      \
        ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>                                       \
    PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, uint64_t preQuantScalar, WaitEvents&... events)      \
    {                                                                                                                \
        TSYNC(events...);                                                                                            \
        MAP_INSTR_IMPL_T(                                                                                            \
            TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode), dst, src, preQuantScalar);     \
        return {};                                                                                                   \
    }                                                                                                                \
    template <                                                                                                       \
        STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,       \
        ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>                                       \
    PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, uint64_t preQuantScalar, WaitEvents&... events)      \
    {                                                                                                                \
        TSYNC(events...);                                                                                            \
        MAP_INSTR_IMPL_T(                                                                                            \
            TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode, Phase), dst, src,               \
            preQuantScalar);                                                                                         \
        return {};                                                                                                   \
    }

#define PTO_DEFINE_TSTORE_FP_OVERLOAD(MAPPED_INSTR)                                                                    \
    template <                                                                                                         \
        typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,   \
        ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>                                         \
    PTO_INST RecordEvent TSTORE_FP(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events)              \
    {                                                                                                                  \
        TSYNC(events...);                                                                                              \
        MAP_INSTR_IMPL_T(                                                                                              \
            MAPPED_INSTR, PTO_TEMPLATE_ARGS(TileData, GlobalData, FpTileData, atomicType, reluPreMode), dst, src, fp); \
        return {};                                                                                                     \
    }

#define PTO_DEFINE_TIMG2COL_AND_SETFMATRIX_OVERLOADS()                                                                \
    template <                                                                                                        \
        typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL,      \
        typename... WaitEvents>                                                                                       \
    PTO_INST RecordEvent TIMG2COL(                                                                                    \
        TileData& dst, ConvTileData& src, uint16_t posM = 0, uint16_t posK = 0, WaitEvents & ... events)              \
    {                                                                                                                 \
        TSYNC(events...);                                                                                             \
        MAP_INSTR_IMPL_T(TIMG2COL, PTO_TEMPLATE_ARGS(TileData, ConvTileData, FmatrixMode), dst, src, posM, posK);     \
        return {};                                                                                                    \
    }                                                                                                                 \
    template <                                                                                                        \
        typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents> \
    PTO_INST RecordEvent SETFMATRIX(ConvTileData& src, WaitEvents&... events)                                         \
    {                                                                                                                 \
        TSYNC(events...);                                                                                             \
        PTO_SETFMATRIX_IMPL_BODY(ConvTileData, FmatrixMode, src);                                                     \
        return {};                                                                                                    \
    }

#define PTO_DEFINE_SIMPLE_TGATHER_OVERLOAD()                                                                        \
    template <                                                                                                      \
        typename TileDataD, typename TileDataS0, typename TileDataS1, typename TileDataTmp, typename... WaitEvents> \
    PTO_INST RecordEvent TGATHER(                                                                                   \
        TileDataD& dst, TileDataS0& src0, TileDataS1& src1, TileDataTmp& tmp, WaitEvents&... events)                \
    {                                                                                                               \
        TSYNC(events...);                                                                                           \
        MAP_INSTR_IMPL(TGATHER, dst, src0, src1, tmp);                                                              \
        return {};                                                                                                  \
    }

#endif
