/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_ISA_COST_MODEL_HPP
#define PTO_ISA_COST_MODEL_HPP

#include <string>
#include <unordered_map>
#include <utility>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <pto/common/pto_tile.hpp>
#include "pto/costmodel/a2a3/TBinOp.hpp"
#include "pto/costmodel/a2a3/TBinSOp.hpp"
#include "pto/costmodel/a2a3/TUnaryOp.hpp"

namespace pto {
inline int sum_repeat_times;

// BinOp
struct AddOp {
    PTO_INTERNAL static void BinInstr(uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "AddOp, repeats:  " << repeats << std::endl;
    }

    PTO_INTERNAL static void BinInstr(uint8_t repeats, uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "AddOp, repeats:  " << repeats << std::endl;
    }
};

struct MulOp {
    PTO_INTERNAL static void BinInstr(uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "MulOp, repeats: " << repeats << std::endl;
    }

    PTO_INTERNAL static void BinInstr(uint8_t repeats, uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "MulOp, repeats: " << repeats << std::endl;
    }
};

struct SubOp {
    PTO_INTERNAL static void BinInstr(uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "SubOp, repeats: " << repeats << std::endl;
    }

    PTO_INTERNAL static void BinInstr(uint8_t repeats, uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "SubOp, repeats: " << repeats << std::endl;
    }
};

// BinSOp
struct AddSOp {
    PTO_INTERNAL static void BinSInstr(uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "AddSOp, repeats: " << repeats << std::endl;
    }

    PTO_INTERNAL static void BinSInstr(uint8_t repeats, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "AddSOp, repeats: " << repeats << std::endl;
    }
};

struct MulSOp {
    PTO_INTERNAL static void BinSInstr(uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "MulSOp, repeats: " << repeats << std::endl;
    }

    PTO_INTERNAL static void BinSInstr(uint8_t repeats, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "MulSOp, repeats: " << repeats << std::endl;
    }
};


struct MinSOp {
    PTO_INTERNAL static void BinSInstr(uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "MinSOp, repeats: " << repeats << std::endl;
    }

    PTO_INTERNAL static void BinSInstr(uint8_t repeats, uint8_t dstStride = BLOCK_MAX_PER_REPEAT, uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "MinSOp, repeats: " << repeats << std::endl;
    }
};


struct SDivOp {
    PTO_INTERNAL static void BinSInstr(uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "SDivOp, repeats: " << repeats << std::endl;
    }

    PTO_INTERNAL static void BinSInstr(uint8_t repeats, uint8_t dstStride = BLOCK_MAX_PER_REPEAT, uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "SDivOp, repeats: " << repeats << std::endl;
    }
};

struct DivSOp {
    PTO_INTERNAL static void BinSInstr(uint8_t repeats)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "DivSOp, repeats: " << repeats << std::endl;
    }

    PTO_INTERNAL static void BinSInstr(uint8_t repeats, uint8_t dstStride = BLOCK_MAX_PER_REPEAT, uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "DivSOp, repeats: " << repeats << std::endl;
    }
};


// UnaryOp
struct AbsOp {
    PTO_INTERNAL static void UnaryInstr(uint8_t repeats, uint8_t dstStride = BLOCK_MAX_PER_REPEAT, uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "AbsOp, repeats: " << repeats << std::endl;
    }
};

struct ExpOp {
    PTO_INTERNAL static void UnaryInstr(uint8_t repeats, uint8_t dstStride = BLOCK_MAX_PER_REPEAT, uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "ExpOp, repeats: " << repeats << std::endl;
    }
};

struct SqrtOp {
    PTO_INTERNAL static void UnaryInstr(uint8_t repeats, uint8_t dstStride = BLOCK_MAX_PER_REPEAT, uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        sum_repeat_times += static_cast<int>(repeats);
        std::cout << "SqrtOp, repeats: " << repeats << std::endl;
    }
};




enum class DataType {
    FP16,
    FP32,
    INT8,
    INT16,
    UNIT8,
    INT32,
    BF16
};

template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
struct TileInfo {
public:
    using DType = Element_;

    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;

    static constexpr int ValidRow = RowValid_;
    static constexpr int ValidCol = ColValid_;

    static constexpr int GetValidRow()
    {
        return ValidRow;
    }

    static constexpr int GetValidCol()
    {
        return ValidCol;
    }

    static constexpr int RowStride = Cols;
    static constexpr int ColStride = Rows;

    float cycle;
    void SetCycle(const float cycle_)
    {
        cycle = cycle_;
    }

    float GetCycle() const
    {
        return cycle;
    }
};

struct InstrTypeHash {
    size_t operator()(const std::pair<std::string, DataType>& key) const
    {
        auto hash_instr = std::hash<std::string>{}(key.first);
        auto hash_type = std::hash<int>{}(static_cast<int>(key.second));
        return hash_instr ^ (hash_type << 1);
    }
};

class CostModel {
public:
    static CostModel& GetInstance()
    {
        static CostModel instance;
        return instance;
    }

    CostModel(const CostModel&) = delete;
    CostModel& operator=(const CostModel&) = delete;

    void InitDefaultParams() {
        // TADD
        SetParam("TADD", DataType::INT16, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("TADD", DataType::INT32, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("TADD", DataType::FP16, 14.0, 19.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("TADD", DataType::FP32, 14.0, 19.0, 2.0, 18.0, 1.0, 0.0);

        // TMUL
        SetParam("TMUL", DataType::INT16, 14.0, 18.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("TMUL", DataType::INT32, 14.0, 18.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("TMUL", DataType::FP16, 14.0, 20.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("TMUL", DataType::FP32, 14.0, 20.0, 2.0, 18.0, 1.0, 0.0);

        // TSUB
        SetParam("TSUB", DataType::INT16, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("TSUB", DataType::INT32, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("TSUB", DataType::FP16, 14.0, 19.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("TSUB", DataType::FP32, 14.0, 19.0, 2.0, 18.0, 1.0, 0.0);

        // TEXP
        SetParam("TEXP", DataType::FP16, 13.0, 28.0, 4.0, 18.0, 1.0, 0.0);
        SetParam("TEXP", DataType::FP32, 13.0, 26.0, 2.0, 18.0, 1.0, 0.0);

        // TSQRT
        SetParam("TSQRT", DataType::FP16, 13.0, 29.0, 4.0, 18.0, 1.0, 0.0);
        SetParam("TSQRT", DataType::FP32, 13.0, 27.0, 2.0, 18.0, 1.0, 0.0);

        // TADDS
        SetParam("TADDS", DataType::INT16, 14.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TADDS", DataType::INT32, 14.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TADDS", DataType::FP16, 14.0, 19.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TADDS", DataType::FP32, 14.0, 19.0, 1.0, 18.0, 1.0, 0.0);

        // TABS
        SetParam("TABS", DataType::INT16, 13.0, 19.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TABS", DataType::INT32, 13.0, 19.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TABS", DataType::FP16, 13.0, 19.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TABS", DataType::FP32, 13.0, 19.0, 1.0, 18.0, 1.0, 0.0);

        // TMINS
        SetParam("TMINS", DataType::INT16, 14.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TMINS", DataType::INT32, 14.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TMINS", DataType::FP16, 14.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TMINS", DataType::FP32, 14.0, 17.0, 1.0, 18.0, 1.0, 0.0);

        // TMULS
        SetParam("TMULS", DataType::INT16, 14.0, 18.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TMULS", DataType::INT32, 14.0, 18.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TMULS", DataType::FP16, 14.0, 20.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TMULS", DataType::FP32, 14.0, 20.0, 1.0, 18.0, 1.0, 0.0);

        // TDIVS
        SetParam("TDIVS", DataType::INT16, 14.0, 18.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TDIVS", DataType::INT32, 14.0, 18.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TDIVS", DataType::FP16, 14.0, 20.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("TDIVS", DataType::FP32, 14.0, 20.0, 1.0, 18.0, 1.0, 0.0);
    }

    // TBinOp
    template <typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
    void BinOpPredictCycle(const std::string& instr_name, TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1)
    {
        sum_repeat_times = 0;
        using T = typename TileDataDst::DType;
        runBinaryOp <T, Op, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
        float resultCycles = PredictCycles<T>(instr_name);
        dst.SetCycle(resultCycles);
        DataType dtype = GetDataTypeEnum<T>();
        std::cout << "Instr: " << instr_name << " Dtype: " << static_cast<int>(dtype) << " Cycles: " << resultCycles << std::endl;
    }

    // TBinSOp
    template <typename Op, typename TileDataDst, typename TileDataSrc>
    void BinSOpPredictCycle(const std::string& instr_name, TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar)
    {
        sum_repeat_times = 0;
        using T = typename TileDataSrc::DType;
        runBinaryScalarOp <T, Op, TileDataDst, TileDataSrc>(dst, src);
        float resultCycles = PredictCycles<T>(instr_name);
        dst.SetCycle(resultCycles);
        DataType dtype = GetDataTypeEnum<T>();
        std::cout << "Instr: " << instr_name << " Dtype: " << static_cast<int>(dtype) << " Cycles: " << resultCycles << std::endl;
    }

    // TUnaryOp
    template <typename Op, typename TileDataDst, typename TileDataSrc>
    void UnaryOpPredictCycle(const std::string& instr_name, TileDataDst& dst, TileDataSrc& src)
    {
        sum_repeat_times = 0;
        using T = typename TileDataSrc::DType;
        runUnaryOp<T, Op, TileDataDst, TileDataSrc, false>(dst, src);
        float resultCycles = PredictCycles<T>(instr_name);
        dst.SetCycle(resultCycles);
        DataType dtype = GetDataTypeEnum<T>();
        std::cout << "Instr: " << instr_name << " Dtype: " << static_cast<int>(dtype) << " Cycles: " << resultCycles << std::endl;
    }

    template <typename T>
    float PredictCycles(const std::string& instr_name)
    {
        DataType dtype = GetDataTypeEnum<T>();
        auto key = std::make_pair(instr_name, dtype);

        fprintf(stdout, "[CostModel] params: <%s> <%d> \n", instr_name.c_str(), static_cast<int>(dtype));

        if (!CheckParamExist(key)) {
            fprintf(stderr, "[CostModel] Error: <%s> <%d> \n", instr_name.c_str(), static_cast<int>(dtype));
            return 0.0f;
        }

        float additional_cycles = additional_cycles_map_[key];
        std::cout << "additional_cycles: " << additional_cycles << std::endl;
        float complete_cycles = complete_cycles_map_[key];
        std::cout << "complete_cycles: " << complete_cycles << std::endl;
        float computing_cycles = computing_cycles_map_[key];
        std::cout << "computing_cycles: " << computing_cycles << std::endl;
        float interval_cycles = interval_cycles_map_[key];
        std::cout << "interval_cycles: " << interval_cycles << std::endl;
        float mask_effect = mask_effect_map_[key];
        std::cout << "mask_effect: " << mask_effect << std::endl;
        float bank_conflict_cycles = bank_conflict_cycles_map_[key];
        std::cout << "bank_conflict_cycles: " << bank_conflict_cycles << std::endl;

        std::cout << "sum_repeat_times: " << sum_repeat_times << std::endl;

        float sum_cycles = additional_cycles + complete_cycles + computing_cycles +
            ((sum_repeat_times - 1) * computing_cycles * mask_effect)
        + bank_conflict_cycles;

        return sum_cycles;
    }


private:
    CostModel() {
        InitDefaultParams();
    }

    void SetParam(const std::string& instr_name, DataType dtype,
        double head, double complete, double computing, double interval,
        double mask, double bank_conflict)
    {
        auto key = std::make_pair(instr_name, dtype);
        additional_cycles_map_[key] = head;
        complete_cycles_map_[key] = complete;
        computing_cycles_map_[key] = computing;
        interval_cycles_map_[key] = interval;
        mask_effect_map_[key] = mask;
        bank_conflict_cycles_map_[key] = bank_conflict;
    }

    bool CheckParamExist(const std::pair<std::string, DataType>& key)
    {
        return additional_cycles_map_.count(key) &&
            complete_cycles_map_.count(key) &&
            computing_cycles_map_.count(key) &&
            interval_cycles_map_.count(key) &&
            mask_effect_map_.count(key) &&
            bank_conflict_cycles_map_.count(key);
    }

    template <typename T>
    DataType GetDataTypeEnum()
    {
        if constexpr (std::is_same_v<T, __bf16>) {
            return DataType::BF16;
        } else if constexpr (std::is_same_v<T, half>) {
            return DataType::FP16;
        } else if constexpr (std::is_same_v<T, float>) {
            return DataType::FP32;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return DataType::INT16;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return DataType::INT32;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return DataType::INT8;
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            return DataType::UNIT8;
        } else {
            fprintf(stderr, "[CostModel] Warning: unknow data type, use FP16 instead.\n");
            return DataType::FP16;
        }
    }

    std::unordered_map<std::pair<std::string, DataType>, float, InstrTypeHash> additional_cycles_map_;
    std::unordered_map<std::pair<std::string, DataType>, float, InstrTypeHash> complete_cycles_map_;
    std::unordered_map<std::pair<std::string, DataType>, float, InstrTypeHash> computing_cycles_map_;
    std::unordered_map<std::pair<std::string, DataType>, float, InstrTypeHash> interval_cycles_map_;
    std::unordered_map<std::pair<std::string, DataType>, float, InstrTypeHash> mask_effect_map_;
    std::unordered_map<std::pair<std::string, DataType>, float, InstrTypeHash> bank_conflict_cycles_map_;

};

} // namespace pto

#endif