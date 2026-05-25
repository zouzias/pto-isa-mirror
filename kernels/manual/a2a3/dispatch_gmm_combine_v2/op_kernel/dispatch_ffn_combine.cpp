#include "kernel_operator.h"
#include "dispatch_ffn_combine_tiling.h"
#include "protocol/remote_window.hpp"
#include "protocol/task_plan.hpp"
#include "dispatch/dispatch_pull.hpp"
#include "combine/combine_push.hpp"
#include "output/unpermute_reduce.hpp"
#include "../kernel_launch.hpp"

using V4RemoteWindowContext = RemoteWindowContext;
using V4DispatchRangeTask = DispatchRangeTask;
using V4CombineOnlyParams = CombineOnlyParams;
using V4StandaloneKernelTilingData = StandaloneKernelTilingData;

extern "C" __global__ __aicore__ void dispatch_ffn_combine(
    __gm__ V4RemoteWindowContext* remoteCtx,
    __gm__ uint8_t* modeArgs,
    __gm__ V4StandaloneKernelTilingData* tiling)
{
    if (tiling->mode == static_cast<uint32_t>(KernelMode::DispatchRangeOnly)) {
        v2_dispatch::RunDispatchRangeOnlyKernel(remoteCtx,
                                                reinterpret_cast<__gm__ V4DispatchRangeTask*>(modeArgs),
                                                tiling);
    } else if (tiling->mode == static_cast<uint32_t>(KernelMode::CombineOnly)) {
        v2_combine::RunCombineOnlyKernel(remoteCtx,
                                         reinterpret_cast<__gm__ V4CombineOnlyParams*>(modeArgs),
                                         tiling);
    }
}

void launchDispatchRange(const DispatchRangeLaunchArgs& args, void* stream)
{
    dispatch_ffn_combine<<<args.blockDim, nullptr, stream>>>(
        static_cast<RemoteWindowContext*>(args.remoteWindow),
        static_cast<uint8_t*>(args.dispatchRanges),
        static_cast<StandaloneKernelTilingData*>(args.tiling));
}

void launchCombineOnly(const CombineOnlyLaunchArgs& args, void* stream)
{
    dispatch_ffn_combine<<<args.blockDim, nullptr, stream>>>(
        static_cast<RemoteWindowContext*>(args.remoteWindow),
        static_cast<uint8_t*>(args.params),
        static_cast<StandaloneKernelTilingData*>(args.tiling));
}
