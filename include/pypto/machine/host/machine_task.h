/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file machine_task.h
 * \brief
 */

#pragma once

#ifndef MACHINE_TASK_H
#define MACHINE_TASK_H

#include <list>
#include <cstdint>
#include <unistd.h>
#include <memory>
#include <iostream>
#include "interface/function/function.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {
inline int64_t CalcShapeSizeFunc (const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto &i : shape) {
        size *= i;
    }
    return size;
}

struct InvokeParaOffset {
    uint8_t* rawTensorAddr{nullptr}; // 原始input output tensor基地址, 如果是子图间workspace incast outcast 则为null
    uint64_t offset{0};
    uint64_t rawTensorOffset{0};
    bool isTensorParam{false};
    uint64_t rawShapeSize{0};
    int rawMagic{0};
    std::string rawSymbol{""};
    int opOriginArgsSeq{INVALID_IN_OUT_INDEX}; // map origin args seq no
    int funcitonMagic{-1};
    int8_t ioIndex{-1};
    int8_t paramType{-1};
    std::vector<int64_t> tensorShape;
    int opMagic{0};
    DataType datatype{DataType::DT_INT32};
    std::vector<int64_t> rawTensorShape;
    void LogRawTensorInfo(std::shared_ptr<RawTensor> rawTensor) {
        auto rawShape = rawTensor->GetRawShape();
        rawShapeSize = CalcShapeSizeFunc(rawShape) * BytesOf(rawTensor->GetDataType());
        rawMagic = rawTensor->GetRawMagic();
        rawSymbol = rawTensor->GetSymbol();
        datatype = rawTensor->GetDataType();
    }
};

enum class CacheReuseType {
    None = 0,
    Function,
    Bin
};

class MachineTask {
public:
    MachineTask(uint64_t taskId, Function *function)
        : taskId_(taskId), function_(function), cacheReuseType_(CacheReuseType::None) {}

    uint64_t GetTaskId() const { return taskId_; }
    Function *GetFunction() const { return function_; }
    void SetFunction(Function *func) { function_ = func; }
    CacheReuseType GetCacheReuseType() const { return cacheReuseType_; }
    void SetCacheReuseType(const CacheReuseType cacheReuseType) { cacheReuseType_ = cacheReuseType; }
    const std::string& GetCacheKey() const { return cacheKey_; }
    void SetCacheKey(const std::string &cacheKey) { cacheKey_ = cacheKey; }
    void SetError(std::string msg) { error = std::move(msg); }
    const std::string &Error() { return error; }
private:
    uint64_t taskId_;
    Function *function_;
    std::string cacheKey_;
    CacheReuseType cacheReuseType_;
    std::string error;
};
}
#endif // MACHINE_TASK_H
