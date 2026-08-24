/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifdef __CCE_AICORE__
inline AICORE bool InitSmokeKernelBlock(__gm__ uint8_t* fftsAddr, int gridRows, int gridCols, int& blockIdx)
{
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    blockIdx = get_block_idx();
    int totalBlocks = gridRows * gridCols;
    return blockIdx >= 0 && blockIdx < totalBlocks;
}
#else
#define GRID_SMOKE_UNUSED_ARGS(fftsAddr, windows, inBuf, outBuf, hcclCtxRaw, gridRows, gridCols) \
    do {                                                                                         \
        (void)(fftsAddr);                                                                        \
        (void)(windows);                                                                         \
        (void)(inBuf);                                                                           \
        (void)(outBuf);                                                                          \
        (void)(hcclCtxRaw);                                                                      \
        (void)(gridRows);                                                                        \
        (void)(gridCols);                                                                        \
    } while (0)
#endif
