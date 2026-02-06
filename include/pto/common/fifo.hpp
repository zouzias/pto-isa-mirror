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

namespace pto {

// FIFO definitons
enum class FIFOType : uint8_t {
    GLOBAL_MEM = 0,
    LOCAL_MEM = 1,
};

template <typename DataType, FIFOType FifoType, int Depth, int Period>
struct DataFifo;

// 1. Specialization for GM FIFO
template <typename DataType, int Depth, int Period>
struct DataFifo<DataType, FIFOType::GLOBAL_MEM, Depth, Period> {
    static constexpr int fifoDepth = Depth;
    static constexpr int fifoPeriod = Period;
    static constexpr FIFOType fifoType = FIFOType::GLOBAL_MEM;

    __gm__ DataType *fifoBase;

    PTO_INTERNAL DataFifo(__gm__ DataType *ptr) : fifoBase(ptr) {}
};

// 2. Specialization for LOCAL_MEM
template <typename DataType, int Depth, int Period>
struct DataFifo<DataType, FIFOType::LOCAL_MEM, Depth, Period> {
    static constexpr int fifoDepth = Depth;
    static constexpr int fifoPeriod = Period;
    static constexpr FIFOType fifoType = FIFOType::LOCAL_MEM;

    DataType *tilePtr;

    // Constructor for Pointer
    PTO_INTERNAL DataFifo(DataType *ptr) : tilePtr(ptr) {}

    // Constructor for Reference/Tile
    PTO_INTERNAL DataFifo(DataType &tile) { tilePtr = reinterpret_cast<DataType *>(tile.data()); }
};

} // namespace pto

#endif