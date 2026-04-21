/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "pto/comm/async/AsyncEventCommonDetail.hpp"

#ifdef PTO_URMA_SUPPORTED
namespace pto {
namespace comm {

template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace, uint32_t destRankId, AsyncSession &session)
{
    static_assert(engine == DmaEngine::URMA, "This overload is for URMA only");
    session.engine = engine;
    session.valid = urma::BuildUrmaSession(workspace, destRankId, session.urmaSession);
    return session.valid;
}

} // namespace comm
} // namespace pto
#endif
