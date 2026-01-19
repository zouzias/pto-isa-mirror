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
 * \file core_func_data.h
 * \brief
 */

#pragma once

#ifndef CORE_FUNC_DATA_H
#define CORE_FUNC_DATA_H

#include <cstdint>
#include "tilefwk/aicore_data.h"

inline constexpr uint32_t DIST_COMM_GROUP_NUM = 8;
inline constexpr size_t MAX_CACHED_FUNC_NUM = 128;
constexpr int MAX_DIMS = 8;
using taskid_t = uint32_t;

namespace npu::tile_fwk {

// VIRTUAL_PURE & VIRTUAL_MIX are special core type ,just used by machine
enum class MachineType { AIV = 0, AIC = 1, MIX = 2, AICPU = 3, HUB = 4, VIRTUAL_PURE = 5, VIRTUAL_MIX = 6 };
#pragma pack(1)
struct CoreFunctionTopo {
    uint64_t coreType;  // aic=0 aiv=1 aicpu=2
    uint32_t extType; // 当coreType==aicpu时，extType表示OP类型
    uint32_t extParamNum; // aicpu上运行的OP使用的参数个数，参数跟在depIds后面
    uint64_t psgId;  // 同构后coreFuncton id(同构子图id),用来取CoreFunctionBin数据
    int64_t readyCount; // 此CoreFunction执行依赖的个数，0表示不依赖，可以被下发执行
    uint64_t depNum; // 依赖此CoreFunction的个数，用于内存分配，保存序号列表
    uint64_t depIds[0]; // 保存依赖此CoreFunction的id列表
};

struct CoreFunctionBin {
    uint64_t size;
    uint8_t data[0]; // kernel.o
};

#pragma pack()

// 每个core function 相关数据在workspace中的绝对地址
struct CoreFunctionWsAddr {
    uint64_t functionBinAddr; // 转成 CoreFunctionBin* 使用
    uint64_t invokeEntryAddr; // GMTensor,...Incast tensor,...Outcast tensor,...
    uint64_t psgId;
    uint64_t topoAddr; // 转换CoreFunctionTopo*使用
    uint64_t invokeEntryInfo; // funcTensorInfo
    uint64_t invokeEntryNum;
    uint64_t invokeEntryOriAddr;

    CoreFunctionWsAddr()
        : functionBinAddr(0), invokeEntryAddr(0), psgId(0), topoAddr(0), invokeEntryInfo(0), invokeEntryNum(0) {}
    CoreFunctionWsAddr(uint64_t bin, uint64_t invoke, uint64_t psg, uint64_t topo, uint64_t info, uint64_t num) :
        functionBinAddr(bin), invokeEntryAddr(invoke), psgId(psg), topoAddr(topo),
        invokeEntryInfo(info), invokeEntryNum(num) {}
    CoreFunctionWsAddr(uint64_t bin, uint64_t invoke, uint64_t psg, uint64_t topo, uint64_t info, uint64_t num, uint64_t oriAddr) :
        functionBinAddr(bin), invokeEntryAddr(invoke), psgId(psg), topoAddr(topo),
        invokeEntryInfo(info), invokeEntryNum(num), invokeEntryOriAddr(oriAddr) {}
};

// host machine 发给device machine的task数据
struct CoreFunctionData {
    uint64_t coreFunctionWsAddr; // 指针指向CoreFunctionWsAddr结构体列表
    uint64_t stackWorkSpaceAddr;
    uint64_t stackWorkSpaceSize;
    uint64_t hcclContextAddr[DIST_COMM_GROUP_NUM] {0};
    uint64_t commGroupNum {0};
};

struct CoreFunctionReadyState {
    int64_t readyCount;
    uint64_t coreType;

    CoreFunctionReadyState() : readyCount(0), coreType(0) {}
    CoreFunctionReadyState(int64_t cnt, uint64_t type) : readyCount(cnt), coreType(type) {}
};

#pragma pack(1)

struct ReadyCoreFunction {
    uint64_t id;
    uint64_t coreType; // 0: aic  1: aiv
};
#pragma pack()
struct ReadyCoreFunctionList {
    uint64_t count;
    ReadyCoreFunction readyFunction[0];
};

inline constexpr uint32_t MAX_PREFETCH_NUM = 4;
struct L2PreInfo {
    int64_t prefetchNum;
    uint64_t prefetchSizes[MAX_PREFETCH_NUM];
    uint64_t prefetchAddrs[MAX_PREFETCH_NUM];
};

// host machine 发给device machine的task数据
// 暂时放在此位置
struct DeviceTask {
    uint64_t coreFunctionCnt;            // core machine 执行函数个数
    uint64_t coreFunctionReadyStateAddr; // 指针指向CoreFunctionReadyState结构体列表
    uint64_t readyAicCoreFunctionQue; // 指针指向ReadyCoreFunctionQueue 结构
    uint64_t readyAivCoreFunctionQue; // 指针指向ReadyCoreFunctionQueue 结构
    uint64_t readyAicpuFunctionQue; // 指针指向ReadyCoreFunctionQueue 结构
#ifdef SUPPORT_MIX_SUBGRAPH_SCHE
    uint64_t readyWrapCoreFunctionQue; // 指针指向WrapInfoqQueue 结构
    uint64_t wrapTasklist; // 指针指向tasklist数组
    uint64_t wrapIdNum; // 包含的有效wrapId个数
    uint64_t opWrapList[MAX_CACHED_FUNC_NUM]; // 指针数组，指向每个function的callop对应的wrapId
    uint64_t opWrapTaskNumList[MAX_CACHED_FUNC_NUM]; // 指针数组，指向每个function的callop对应的wrapTaskNum
#endif
    CoreFunctionData coreFuncData;
    L2PreInfo l2Info;
    uint64_t costModelData;           // costmodel仿真时长
    uint64_t aicoreModel;             // costmodel aicore功能模型
};

// dfx 相关
struct TensorInfo {
    int functionMagic;
    uint32_t subgraphId;
    uint32_t deviceId;
    uint32_t rawMagic;
    uint32_t opMagic;
    int32_t stride[MAX_DIMS];
    int32_t hostpid{0};
    int32_t dataType;        // INT8...
    uint32_t dataByte;
    int32_t format;          // ND...
    int32_t paramType;       // 0:func incast; 1:func outcas; 2:mid incast; 3:mid outcast
    int32_t dumpType;        // 00: no dump; 01:func dunp; 10:subgraph dump; 11:func dunp & subgraph dump
    uint32_t idx;
    uint32_t dims;
    int shape[MAX_DIMS];
};

#pragma pack (8)
struct BaseArgs {
    uint64_t inputNum;
    uint64_t outputNum;
    uint64_t nrAic;
    uint64_t nrAiv;
    uint64_t nrAicpu;
    uint64_t nrValidAic;
    uint64_t opaque;          // store device global data, must be init with zero
    uint64_t sharedBuffer;       // SHARED_BUFFER_SIZE per core, need memset
    uint64_t coreRegAddr;        // core reg addr, uint64_t per core, aic first
    uint64_t taskId;          // initial task id
};
#pragma pack ()

using predcount_t = uint16_t;

struct DynFuncBin {
    uint32_t coreType;
    uint32_t psgId;
    uint64_t funcHash;
#ifdef SUPPORT_MIX_SUBGRAPH_SCHE
    int32_t wrapVecId {-1};
    uint32_t mixResourceType {0};
#endif
};

struct DynFuncHeader {
    uint64_t seqNo;
    uint32_t funcNum;
    uint32_t funcSize;
    __gm__ DynFuncBin *cceBinary;

    uint64_t GetIndex() {
        return seqNo;
    }

    inline DynFuncData &At(int index) {
        return (reinterpret_cast<DynFuncData *>(this + 1))[index];
    }
    inline uint32_t Size() {
        return funcNum;
    }
};

#pragma pack ()

} // namespace npu::tile_fwk

#endif
