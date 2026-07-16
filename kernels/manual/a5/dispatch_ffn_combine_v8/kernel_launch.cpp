#include <acl/acl.h>

#ifndef PTO_NPU_ARCH_A5
#define PTO_NPU_ARCH_A5
#endif

#ifndef __CCE_AICORE__
#define __CCE_AICORE__
#endif

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include "kernel_operator.h"

#include <pto/pto-inst.hpp>

#include <cstddef>

#include "dispatch_ffn_combine.h"
#include "kernel_launch.hpp"
#include "op_kernel/utils/hccl_window.hpp"

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

extern "C" __global__ __aicore__ void dispatch_ffn_combine_kernel(GM_ADDR fftsAddr, GM_ADDR x, GM_ADDR w1, GM_ADDR w2,
                                                                  GM_ADDR expertId, GM_ADDR scale1, GM_ADDR scale2,
                                                                  GM_ADDR probs, GM_ADDR xActiveMask, GM_ADDR c,
                                                                  GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
                                                                  GM_ADDR tilingGM, GM_ADDR profileGM,
                                                                  uint32_t stageProfile, uint32_t startSyncDebug)
{
#ifdef _DEBUG
    cce::printf("V8 kernel enter block=%d sub=%d DAV_VEC=%d workspace=%d tiling=%d\n", int(get_block_idx()),
                int(get_subblockid()), int(DAV_VEC), int(workspaceGM != nullptr), int(tilingGM != nullptr));
#endif
    __gm__ uint64_t *profileEntry = nullptr;
    if (profileGM != nullptr) {
        std::size_t profileOffset = static_cast<std::size_t>(get_block_idx()) * kDispatchFFNCombineProfileBytesPerBlock;
        if constexpr (DAV_VEC) {
            profileOffset +=
                (static_cast<std::size_t>(get_subblockid()) + 1U) * kDispatchFFNCombineProfileEntryBytes;
        }
        profileEntry =
            reinterpret_cast<__gm__ uint64_t *>(reinterpret_cast<__gm__ uint8_t *>(profileGM) + profileOffset);
    }

    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
    if (workspaceGM != nullptr && tilingGM != nullptr) {
        const __gm__ DispatchFFNCombineTilingData *tilingData =
            reinterpret_cast<__gm__ DispatchFFNCombineTilingData *>(tilingGM);
#ifdef _DEBUG
        cce::printf("V8 kernel tiling block=%d sub=%d stage=%d M=%d topK=%d expertNum=%d\n", int(get_block_idx()),
                    int(get_subblockid()), int(tilingData->frontReorderTiling.stageNum),
                    int(tilingData->dispatchFFNCombineInfo.M), int(tilingData->dispatchFFNCombineInfo.topK),
                    int(tilingData->frontReorderTiling.expertNum));
#endif
        if (startSyncDebug != 0U) {
            PtoRemoteWindow remoteWindow;
            remoteWindow.Init(reinterpret_cast<GM_ADDR>(tilingData->runtimeInfo.remoteWindowContext));
            remoteWindow.CrossRankStartSyncAiv();
            remoteWindow.CrossRankStartSyncAic();
            pto::SYNCALL<pto::SyncCoreType::Mix>();
        }
        uint64_t tStart = get_sys_cnt();
        if (profileEntry != nullptr) {
            profileEntry[kDispatchFFNCombineProfileKernelStart] = tStart;
        }
        if (tilingData->frontReorderTiling.stageNum != 0) {
            DispatchFFNCombine<int8_t, int8_t, bfloat16_t, false, true> op;
            __gm__ uint64_t *stageProfileEntry = stageProfile != 0U ? profileEntry : nullptr;
            op.Init(x, w1, w2, expertId, scale1, scale2, probs, xActiveMask, c, expertTokenNums, workspaceGM,
                    tilingData, stageProfileEntry);
            op.Process();
        }
    }

    pipe_barrier(PIPE_ALL);
    uint64_t tEnd = get_sys_cnt();
    if (profileEntry != nullptr) {
        profileEntry[kDispatchFFNCombineProfileKernelEnd] = tEnd;
    }
}

void launchDispatchFFNCombine(const DispatchFFNCombineLaunchArgs &args, void *stream)
{
    dispatch_ffn_combine_kernel<<<args.block_dim, nullptr, stream>>>(
        (GM_ADDR)args.ffts, (GM_ADDR)args.x, (GM_ADDR)args.weight1, (GM_ADDR)args.weight2, (GM_ADDR)args.expert_idx,
        (GM_ADDR)args.scale1, (GM_ADDR)args.scale2, (GM_ADDR)args.probs, (GM_ADDR)args.x_active_mask, (GM_ADDR)args.out,
        (GM_ADDR)args.expert_token_nums, (GM_ADDR)args.workspace, (GM_ADDR)args.tiling, (GM_ADDR)args.profile_data,
        args.stage_profile, args.start_sync_debug);
}
