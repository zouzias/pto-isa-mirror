/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_FIFO_HPP
#define PTO_FIFO_HPP
#include <type_traits>

using namespace std;

namespace pto {

enum TileSplitAxis : uint8_t
{
    TILE_NO_SPLIT = 0,   // 1:1 mode, no split, using AIV0
    TILE_UP_DOWN = 1,    // Split/combine along rows: AIV0=upper half, AIV1=lower half
    TILE_LEFT_RIGHT = 2, // Split/combine along cols: AIV0=left half, AIV1=right half
};

enum Direction : uint8_t
{
    UNDEFINED = 0,
    DIR_C2V = 1,                  // Cube → Vector: Cube is producer, Vector is consumer
    DIR_V2C = 2,                  // Vector → Cube: Vector is producer, Cube is consumer
    DIR_BOTH = DIR_C2V | DIR_V2C, // Support both directions
    DIR_V2C_CTRL = 4,
#ifdef PTO_NPU_ARCH_A5
    DIR_C2V_GM = 5,
    DIR_V2C_GM = 6,
    DIR_BOTH_GM = DIR_C2V_GM | DIR_V2C_GM,
#endif
};

template <int SlotSize, int SlotNum, int LocalSlotNum>
struct DataFIFO {
    __gm__ void *GM_SLOT_BUFFER = nullptr; // Global memory
    uint32_t C2V_CONSUMER_BUF = 0x0;       // UB buffer
    uint32_t V2C_CONSUMER_BUF = 0x0;       // L1 buffer
    uint64_t V2C_CONTROL_BUF = 0x0;        // scalar buffer for control signals
    static constexpr uint32_t SLOT_SIZE = SlotSize;
    static constexpr uint32_t SLOT_NUM = SlotNum;
    static constexpr uint32_t SLOT_SYNCT = SlotNum;
    static constexpr uint32_t LOCAL_SLOT_NUM = LocalSlotNum;

    PTO_INTERNAL DataFIFO(__gm__ void *gmSlotBuf, uint32_t c2vConsBuf, uint32_t v2cConBuf)
        : GM_SLOT_BUFFER(gmSlotBuf), C2V_CONSUMER_BUF(c2vConsBuf), V2C_CONSUMER_BUF(v2cConBuf)
    {}
};

} // namespace pto
#endif
