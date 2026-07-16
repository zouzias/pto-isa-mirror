/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_FFN_COMBINE_V8_GMM2_COMBINE_CV_PIPE_H
#define DISPATCH_FFN_COMBINE_V8_GMM2_COMBINE_CV_PIPE_H

#include <pto/pto-inst.hpp>

#include "utils/const_args.hpp"

namespace dispatch_ffn_combine_v8 {

using Gmm2CombineCvPipe =
    pto::TPipe<V8_GMM2_COMBINE_CV_READY_HARD_FLAG, pto::Direction::DIR_C2V, V8_GMM2_COMBINE_CV_SLOT_BYTES,
               V8_GMM2_COMBINE_CV_DEFAULT_FIFO_DEPTH, V8_GMM2_COMBINE_CV_DEFAULT_FIFO_DEPTH, false>;

} // namespace dispatch_ffn_combine_v8

#endif // DISPATCH_FFN_COMBINE_V8_GMM2_COMBINE_CV_PIPE_H
