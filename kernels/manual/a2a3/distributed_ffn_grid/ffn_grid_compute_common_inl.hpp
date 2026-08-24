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

using pto::get_block_idx;
using pto::RoundMode;
using pto::set_ffts_base_addr;
using pto::TileType;

inline AICORE bool InitFfnGridKernelBlock(__gm__ uint8_t* fftsAddr, int gridRows, int gridCols, int& blockIdx)
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

template <typename InputMat, typename WeightMat, typename LeftTile, typename RightTile, typename AccTile, typename Pipe>
struct FfnCubeMatmulPushArgs {
    InputMat& inputMat;
    WeightMat& weightMat;
    LeftTile& aT;
    RightTile& bT;
    AccTile& cT;
    Pipe& pipe;
};

template <
    typename GX, typename WGateGlobal, typename WUpGlobal, typename WDownGlobal, typename XMat, typename WGateMat,
    typename WUpMat, typename WDownMat>
struct FfnCubePreludeLoadArgs {
    using GxType = GX;
    using WGateType = WGateGlobal;
    using WUpType = WUpGlobal;
    using WDownType = WDownGlobal;

    __gm__ uint8_t* xBlock;
    __gm__ uint8_t* wGateBlock;
    __gm__ uint8_t* wUpBlock;
    __gm__ uint8_t* wDownBlock;
    XMat& xMat;
    WGateMat& wGateMat;
    WUpMat& wUpMat;
    WDownMat& wDownMat;
};

template <typename Args>
AICORE void RunFfnCubeMatmulPush(Args& args)
{
    TMOV(args.aT, args.inputMat);
    TMOV(args.bT, args.weightMat);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(args.cT, args.aT, args.bT);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TMOV(args.pipe, args.cT);

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
}

template <typename Args>
AICORE void LoadFfnCubePreludeGlobals(Args& args)
{
    using GX = typename Args::GxType;
    using WGateGlobal = typename Args::WGateType;
    using WUpGlobal = typename Args::WUpType;
    using WDownGlobal = typename Args::WDownType;
    GX xG(reinterpret_cast<__gm__ half*>(args.xBlock));
    WGateGlobal wGateG(reinterpret_cast<__gm__ half*>(args.wGateBlock));
    WUpGlobal wUpG(reinterpret_cast<__gm__ half*>(args.wUpBlock));
    WDownGlobal wDownG(reinterpret_cast<__gm__ half*>(args.wDownBlock));

    TLOAD(args.xMat, xG);
    TLOAD(args.wGateMat, wGateG);
    TLOAD(args.wUpMat, wUpG);
    TLOAD(args.wDownMat, wDownG);
}

AICORE void WaitFfnCubePreludeLoads()
{
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
}

template <typename HiddenMat, typename HiddenPipe>
AICORE void PopFfnHiddenToCube(HiddenMat& hiddenMat, HiddenPipe& hiddenPipe)
{
    TMOV(hiddenMat, hiddenPipe);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
}

template <
    typename GX, typename WGateGlobal, typename WUpGlobal, typename WDownGlobal, typename XMat, typename WGateMat,
    typename WUpMat, typename WDownMat, typename LeftTile, typename RightTile, typename AccTile, typename GatePipe,
    typename UpPipe, typename HiddenMat, typename HiddenPipe>
AICORE void RunFfnCubePrelude(
    __gm__ uint8_t* xBlock, __gm__ uint8_t* wGateBlock, __gm__ uint8_t* wUpBlock, __gm__ uint8_t* wDownBlock,
    XMat& xMat, WGateMat& wGateMat, WUpMat& wUpMat, WDownMat& wDownMat, LeftTile& aT, RightTile& bT, AccTile& cT,
    GatePipe& gatePipe, UpPipe& upPipe, HiddenMat& hiddenMat, HiddenPipe& hiddenPipe)
{
    FfnCubePreludeLoadArgs<GX, WGateGlobal, WUpGlobal, WDownGlobal, XMat, WGateMat, WUpMat, WDownMat> loadArgs{
        xBlock, wGateBlock, wUpBlock, wDownBlock, xMat, wGateMat, wUpMat, wDownMat};
    LoadFfnCubePreludeGlobals(loadArgs);
    WaitFfnCubePreludeLoads();

    FfnCubeMatmulPushArgs<XMat, WGateMat, LeftTile, RightTile, AccTile, GatePipe> gateArgs{xMat, wGateMat, aT,
                                                                                           bT,   cT,       gatePipe};
    FfnCubeMatmulPushArgs<XMat, WUpMat, LeftTile, RightTile, AccTile, UpPipe> upArgs{xMat, wUpMat, aT, bT, cT, upPipe};
    RunFfnCubeMatmulPush(gateArgs);
    RunFfnCubeMatmulPush(upArgs);
    PopFfnHiddenToCube(hiddenMat, hiddenPipe);
}

template <
    typename GateTile, typename UpTile, typename HiddenF32TileT, typename HiddenF16TileT, typename GatePipe,
    typename UpPipe>
AICORE void RunFfnVecActivation(
    GateTile& gateF32, UpTile& upF32, HiddenF32TileT& hiddenF32, HiddenF16TileT& hiddenF16, GatePipe& gatePipe,
    UpPipe& upPipe)
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
#define FFN_GRID_UNUSED_KERNEL_ARGS(                                                                                \
    fftsAddr, pipeWindow, x, wGate, wUp, wDown, gatePartial, upPartial, hiddenIn, downPartial, yOutput, hcclCtxRaw, \
    gridRows, gridCols)                                                                                             \
    do {                                                                                                            \
        (void)(fftsAddr);                                                                                           \
        (void)(pipeWindow);                                                                                         \
        (void)(x);                                                                                                  \
        (void)(wGate);                                                                                              \
        (void)(wUp);                                                                                                \
        (void)(wDown);                                                                                              \
        (void)(gatePartial);                                                                                        \
        (void)(upPartial);                                                                                          \
        (void)(hiddenIn);                                                                                           \
        (void)(downPartial);                                                                                        \
        (void)(yOutput);                                                                                            \
        (void)(hcclCtxRaw);                                                                                         \
        (void)(gridRows);                                                                                           \
        (void)(gridCols);                                                                                           \
    } while (0)

#endif
