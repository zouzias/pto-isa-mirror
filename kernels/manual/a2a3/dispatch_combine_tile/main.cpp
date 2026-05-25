/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "args.h"
#include "comm_mpi.h"
#include "golden.h"
#include "hccl_context.h"
#include "kernel_launchers.h"
#include "layout.h"

#include <iostream>

namespace dispatch_combine_tile {

// Host flow contract from DESIGN.md section 14.1.3. Host/runtime code may use
// ACL, HCCL, and MPI at the boundary; device semantics remain PTO-only.
void BindDeviceContinuous(const DispatchCombineTileArgs &args, uint32_t myRank);
void ComputeLayouts(const DispatchCombineTileArgs &args, WorkspaceLayout *workspaceLayout,
                    PeerWindowLayout *peerWindowLayout);
void AllocateLocalBuffers(const DispatchCombineTileArgs &args, const WorkspaceLayout &workspaceLayout);
void SlicePeerWindow(const DispatchCombineTileArgs &args, HcclWindowContext *hcclContext,
                     const PeerWindowLayout &peerWindowLayout);
void RunDispatch(const DispatchCombineTileArgs &args, uint32_t myRank);
void PrepareExpertOutputIdentity(const DispatchCombineTileArgs &args, uint32_t myRank);
void RunCombine(const DispatchCombineTileArgs &args, uint32_t myRank);
void VerifyAndDump(const DispatchCombineTileArgs &args, uint32_t myRank);
void Cleanup();

} // namespace dispatch_combine_tile

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    std::cout << "dispatch_combine_tile scaffold" << std::endl;
    return 0;
}
