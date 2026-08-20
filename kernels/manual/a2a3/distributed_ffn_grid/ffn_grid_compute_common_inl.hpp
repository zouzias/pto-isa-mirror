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

using namespace pto;

inline AICORE bool InitFfnGridKernelBlock(__gm__ uint8_t *fftsAddr, int gridRows, int gridCols, int &blockIdx)
{
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    blockIdx = get_block_idx();
    int totalBlocks = gridRows * gridCols;
    return blockIdx >= 0 && blockIdx < totalBlocks;
}

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

template <
    typename GX, typename WGateGlobal, typename WUpGlobal, typename WDownGlobal, typename XMat, typename WGateMat,
    typename WUpMat, typename WDownMat, typename LeftTile, typename RightTile, typename AccTile, typename GatePipe,
    typename UpPipe, typename HiddenMat, typename HiddenPipe>
AICORE void RunFfnCubePrelude(
    __gm__ uint8_t *xBlock, __gm__ uint8_t *wGateBlock, __gm__ uint8_t *wUpBlock, __gm__ uint8_t *wDownBlock,
    XMat &xMat, WGateMat &wGateMat, WUpMat &wUpMat, WDownMat &wDownMat, LeftTile &aT, RightTile &bT, AccTile &cT,
    GatePipe &gatePipe, UpPipe &upPipe, HiddenMat &hiddenMat, HiddenPipe &hiddenPipe)
{
    GX xG(reinterpret_cast<__gm__ half *>(xBlock));
    WGateGlobal wGateG(reinterpret_cast<__gm__ half *>(wGateBlock));
    WUpGlobal wUpG(reinterpret_cast<__gm__ half *>(wUpBlock));
    WDownGlobal wDownG(reinterpret_cast<__gm__ half *>(wDownBlock));

    TLOAD(xMat, xG);
    TLOAD(wGateMat, wGateG);
    TLOAD(wUpMat, wUpG);
    TLOAD(wDownMat, wDownG);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TMOV(aT, xMat);
    TMOV(bT, wGateMat);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cT, aT, bT);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TMOV(gatePipe, cT);

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    TMOV(aT, xMat);
    TMOV(bT, wUpMat);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cT, aT, bT);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TMOV(upPipe, cT);

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    TMOV(hiddenMat, hiddenPipe);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
}

template <typename GateTile, typename UpTile, typename HiddenF32TileT, typename HiddenF16TileT, typename GatePipe,
          typename UpPipe>
AICORE void RunFfnVecActivation(
    GateTile &gateF32, UpTile &upF32, HiddenF32TileT &hiddenF32, HiddenF16TileT &hiddenF16, GatePipe &gatePipe,
    UpPipe &upPipe)
{
    TMOV(gateF32, gatePipe);
    TMOV(upF32, upPipe);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TLRELU(gateF32, gateF32, FFN_PRELU_ALPHA);

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_V);
#endif

    TMUL(hiddenF32, gateF32, upF32);

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_V);
#endif

    TCVT(hiddenF16, hiddenF32, RoundMode::CAST_RINT);
}

#else
#define FFN_GRID_UNUSED_KERNEL_ARGS(                                                                               \
    fftsAddr, pipeWindow, x, wGate, wUp, wDown, gatePartial, upPartial, hiddenIn, downPartial, yOutput, hcclCtxRaw, \
    gridRows, gridCols)                                                                                            \
    do {                                                                                                           \
        (void)(fftsAddr);                                                                                          \
        (void)(pipeWindow);                                                                                        \
        (void)(x);                                                                                                 \
        (void)(wGate);                                                                                             \
        (void)(wUp);                                                                                               \
        (void)(wDown);                                                                                             \
        (void)(gatePartial);                                                                                       \
        (void)(upPartial);                                                                                         \
        (void)(hiddenIn);                                                                                          \
        (void)(downPartial);                                                                                       \
        (void)(yOutput);                                                                                           \
        (void)(hcclCtxRaw);                                                                                        \
        (void)(gridRows);                                                                                          \
        (void)(gridCols);                                                                                          \
    } while (0)

#endif
