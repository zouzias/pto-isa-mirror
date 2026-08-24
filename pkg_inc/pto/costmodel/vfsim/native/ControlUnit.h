// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VFSIM_NATIVE_CONTROL_UNIT_H_
#define VFSIM_NATIVE_CONTROL_UNIT_H_

#include <functional>
#include <string>
#include <vector>

#include "native/IFU.h"
#include "native/ParamDB.h"

namespace vfsim {

class ControlUnit {
public:
    explicit ControlUnit(const ParamDb* db = nullptr);

    void acceptMembar(const DynamicInst& inst);
    void update(const std::function<bool(int64_t, const std::string&)>& hasPendingBefore);
    bool blocks(const DynamicInst& inst, const ParamDb& db, const std::string& dtype) const;
    bool empty() const noexcept { return barriers_.empty(); }

private:
    struct Barrier {
        int64_t streamSeq = -1;
        int64_t pc = -1;
        std::string barrier;
        std::string waitClass;
        std::string blockClass;
        bool released = false;
    };

    const ParamDb* db_ = nullptr;
    std::vector<Barrier> barriers_;
};

} // namespace vfsim

#endif // VFSIM_NATIVE_CONTROL_UNIT_H_
