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
 * \file aicpu_common.h
 * \brief
 */

#ifndef RUNTIME_COMMON_DEF_H
#define RUNTIME_COMMON_DEF_H

#include <cstdint>

const uint64_t AICORE_TASK_INIT = 0xFFFFFFFF;
const uint64_t AICORE_TASK_STOP = 0x7FFFFFF0;
const uint64_t AICORE_FUNC_STOP = 0x7FFFFFE0;
const uint64_t AICORE_FIN_MASK = 0x80000000;
const uint64_t AICORE_TASK_MAX = 0x70000000;

const uint64_t AICORE_SAY_HELLO = 0x80000000;
const uint64_t AICORE_SAY_ACK = 0x80000000;
const uint64_t AICORE_SAY_GOODBYE = 0x88888888;
const int64_t PRO_LEVEL1 = 2;
const int64_t PRO_LEVEL2 = 3;

constexpr int REG_LOW_TASK_PING = 0;
constexpr int REG_LOW_TASK_PONG = 1;
constexpr int MAX_DFX_TASK_NUM_PER_CORE = 10000;

constexpr int SHAK_BUF_PRINT_BUFFER_INDEX = 5;
constexpr int SHAK_BUF_COREFUNC_DATA_INDEX = 6;
constexpr int SHAK_BUF_DFX_DATA_INDEX = 7;

constexpr int CPU_TO_CORE_SHAK_BUF_COREFUNC_DATA_INDEX = 0;
constexpr int CPU_TO_CORE_SHAK_BUF_GOODBYE_INDEX = 0;

constexpr int FUNC_ID_BATCH = 0x7FF;

const uint64_t SHARED_BUFFER_SIZE = 512;
const uint64_t PMU_BUFFER_SIZE = 4096;
const uint64_t DEVICE_QUEUE_SIZE = 512;
const uint64_t PRINT_BUFFER_SIZE = 16384;

constexpr const int DEV_SHAPE_DIM_NUM_2 = 2;
constexpr const int DEV_SHAPE_DIM_NUM_3 = 3;
constexpr const int DEV_SHAPE_DIM_NUM_4 = 4;
constexpr const int DEV_SHAPE_DIM_NUM_5 = 5;

enum class ArchInfo {
    DAV_1001 = 1001,
    DAV_2201 = 2201,
    DAV_3510 = 3510,
    DAV_UNKNOWN
};

#define DEVICE_TASK_STOP 0x7FFFFFFE

#define DEVICE_TASK_TYPE_STATIC  0
#define DEVICE_TASK_TYPE_DYN     1
#define DEVICE_TASK_TYPE_INVALID 0xf

template <typename DerivedType, typename UnderlyingType>
class BitmaskBase {
public:
    using underlying_type = UnderlyingType;
    underlying_type value{0};
    constexpr BitmaskBase(underlying_type v = 0) : value(v) {}
    constexpr bool Empty() const {
        return value == 0;
    }
    constexpr bool Contains(underlying_type mask) const {
        return (value & mask) == mask;
    }
    constexpr bool Overlaps(underlying_type mask) const {
        return (value & mask) != 0;
    }
    constexpr void Add(underlying_type mask) {
        value |= mask;
    }
    constexpr void Remove(underlying_type mask) {
        value &= ~mask;
    }
    friend constexpr DerivedType operator|(DerivedType lhs, DerivedType rhs) {
        return DerivedType(lhs.value | rhs.value);
    }
    friend constexpr DerivedType operator&(DerivedType lhs, DerivedType rhs) {
        return DerivedType(lhs.value & rhs.value);
    }
    friend constexpr DerivedType operator^(DerivedType lhs, DerivedType rhs) {
        return DerivedType(lhs.value ^ rhs.value);
    }
    friend constexpr DerivedType operator~(DerivedType lhs) {
        return DerivedType(~lhs.value);
    }
    constexpr operator underlying_type() const {
        return value;
    }
};

struct ProfConfig : public BitmaskBase<ProfConfig, uint32_t> {
    using BitmaskBase::BitmaskBase;
    enum : underlying_type {
        OFF = 0x0,
        AICPU_FUNC = 0x1 << 0,
        AICORE_TIME = 0x1 << 1,
        AICORE_PMU = 0x1 << 2,
    };
};

struct ToSubMachineConfig {
    ProfConfig profConfig{ProfConfig::OFF};
    uint64_t isGETensorList{0};
};

struct OpMetaAddrs {
    uint64_t generalAddr{0};     // aicpu meta addr
    uint64_t stitchPoolAddr{0};  // aicpu meta addr
};

struct DeviceArgs {
    uint32_t nrAic{0};
    uint32_t nrAiv{0};
    uint32_t nrAicpu{0};
    uint32_t nrValidAic{0};
    uint64_t opaque{0};       // store device global data, must be init with zero
    uint64_t devQueueAddr;    // pcie/XLink mem, used between host and device, `DEVICE_QUEUE_SIZE`
    uint64_t sharedBuffer;    // SHARED_BUFFER_SIZE per core, aics first
    uint64_t coreRegAddr;     // core reg addr, uint64_t per core, aic first
    uint64_t corePmuRegAddr;  // pmu reg addr, uint64_t per core, aic first
    uint64_t corePmuAddr;     // pmu data addr, PAGE_SIZE per core, aic first
    uint64_t pmuEventAddr;    // pmu event addr
    uint64_t taskType : 4;    // initial task type
    uint64_t machineConfig : 8; // machine config
    uint64_t taskId   : 52;   // initial task id
    uint64_t taskData;        // initial task data
    uint64_t taskWastTime{0};
    uint64_t aicpuSoBin{0};    // server so Bin
    uint64_t aicpuSoLen{0};    // server so len
    uint64_t deviceId{0};      // for device copy fileName
    uint64_t startArgsAddr{0}; // DevStartArgs addr
    uint64_t taskQueue{0};     // task queue between ctrl and sche
    uint64_t taskCtrl{0};      // task ctrl between ctrl and sche
    uint32_t scheCpuNum{0};    // sche cpu num calc by host
    uint32_t enableCtrl : 2;    // if enable builtin ctrl
    uint32_t validGetPgMask : 2; // mark pgmask is invalid
    uint32_t disableSync : 2;    // close ctrl and sche soft sync
    uint32_t isGETensorList : 26;    // GE graph is tensor list
    uint64_t generalAddr{0};     // aicpu meta addr
    uint64_t stitchPoolAddr{0};  // aicpu meta addr
    uint64_t GetBlockNum() { return nrValidAic * (nrAiv / nrAic + 1); }
    ArchInfo archInfo{ArchInfo::DAV_2201};
    ToSubMachineConfig toSubMachineConfig;
};

#define TO_ENTRY_IMPL(name, line, key, type) (name##line##key##type)
#define TO_ENTRY(name, key, type) TO_ENTRY_IMPL(name, _, key, type)

#ifdef __MIX__
#ifdef __AIV__
#define KERNEL_ENTRY(x, y) TO_ENTRY(x, y, _mix_aiv)
#else
#define KERNEL_ENTRY(x, y) TO_ENTRY(x, y, _mix_aic)
#endif
#else
#define KERNEL_ENTRY(x, y) x
#endif

const uint64_t AICORE_REG_SAY_HELLO = 0xF000000080000000;
constexpr uint32_t REG_HIGH_DTASKID_SHIFT = 32;
enum class TASK_POS : size_t { LOW_REG = 0, HIGH_REG = 1, ALL_REG = 2, REG_POS_BUTT = 3 };

struct TaskStat {
    int16_t seqNo;
    int16_t subGraphId;
    int32_t taskId;
    int64_t execStart;
    int64_t execEnd;
    int64_t waitStart; // 2.0 dfx 当前未使用
};

constexpr uint32_t PERF_TRACE_INST_MAX_NUM_EVERY_TYPE = 10;
constexpr uint32_t INVALID_DEV_TASK_ID = 0xFFFFFFFF;
enum AicorePerfTrace {
    PERF_TRACE_CORE_BEGIN = 0,
    PERF_TRACE_CORE_INIT,
    PERF_TRACE_CORE_DEV_TASK_RCV_MODEL,
    PERF_TRACE_CORE_DEV_TASK_WAIT_RCV_FIRST_CALLOP_TASK,
    PERF_TRACE_CORE_DEV_TASK_CALLOP_TASK_EXEC,
    PERF_TRACE_CORE_DEV_TASK_WAIT_SYNC_STOP_NOTIFY,
    PERF_TRACE_CORE_WAIT_ALL_DEV_TASK_CALLOP_EXEC_FINISH,
    PERF_TRACE_CORE_WAIT_EXIT_NOTIFY,
    PERF_TRACE_CORE_MAX
};

struct Metrics {
  int64_t isMetricStop;
  int64_t taskCount; 
  int64_t perfTrace[PERF_TRACE_CORE_MAX][PERF_TRACE_INST_MAX_NUM_EVERY_TYPE];
  uint32_t perfTraceDevTaskId[PERF_TRACE_CORE_MAX][PERF_TRACE_INST_MAX_NUM_EVERY_TYPE];
  uint32_t perfTraceCnt[PERF_TRACE_CORE_MAX];
  TaskStat tasks[];
};

inline const char *AicorePerfTraceName[] = {
    "BEGIN",
    "INIT",
    "DEV_TASK_RCV_MODEL",
    "DEV_TASK_WAIT_RCV_FIRST_CALLOP_TASK",
    "DEV_TASK_ALL_CALLOP_TASK_EXEC",
    "DEV_TASK_WAIT_SYNC_STOP_NOTIFY",
    "WAIT_ALL_DEV_TASK_CALLOP_EXEC_FINISH",
    "WAIT_EXIT_NOTIFY"
};

struct TaskEntry {
    int32_t subGraphId;
    int32_t taskId;
    int64_t funcAddr;
    int64_t tensorAddrs;
    int64_t gmStackSize;
    int64_t gmStackBase;
    int64_t reserved2[2];
    uint32_t tensorSize;
    uint32_t reserved[1];
};

struct KernelArgs {
    int64_t shakeBuffer[8];
    int64_t shakeBufferCpuToCore[8];
    int64_t waveBufferCpuToCore[8];
    TaskEntry taskEntry;
    TaskStat taskStat[2]; // 寄存器高低32位，两个task 和 pending & running task存储： 2 * 2 个
};

static_assert(sizeof(KernelArgs) < SHARED_BUFFER_SIZE);
#endif