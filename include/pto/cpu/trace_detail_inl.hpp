/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRACE_DETAIL_INL_HPP
#define TRACE_DETAIL_INL_HPP

inline const char* LayoutToString(Layout layout)
{
    static constexpr std::array<std::pair<Layout, const char*>, 21> kLayoutNames = {{
        {Layout::ND, "ND"},
        {Layout::DN, "DN"},
        {Layout::NZ, "NZ"},
        {Layout::SCALE, "SCALE"},
        {Layout::MX_A_ND, "MX_A_ND"},
        {Layout::MX_A_DN, "MX_A_DN"},
        {Layout::MX_A_ZZ, "MX_A_ZZ"},
        {Layout::MX_B_ND, "MX_B_ND"},
        {Layout::MX_B_DN, "MX_B_DN"},
        {Layout::MX_B_NN, "MX_B_NN"},
        {Layout::NC1HWC0, "NC1HWC0"},
        {Layout::GNC1HWC0, "GNC1HWC0"},
        {Layout::NCHW, "NCHW"},
        {Layout::GNCHW, "GNCHW"},
        {Layout::NHWC, "NHWC"},
        {Layout::NDC1HWC0, "NDC1HWC0"},
        {Layout::NCDHW, "NCDHW"},
        {Layout::FRACTAL_Z, "FRACTAL_Z"},
        {Layout::FRACTAL_Z_S16S8, "FRACTAL_Z_S16S8"},
        {Layout::FRACTAL_Z_3D, "FRACTAL_Z_3D"},
        {Layout::MAX, "MAX"},
    }};

    for (const auto& layoutName : kLayoutNames) {
        if (layoutName.first == layout) {
            return layoutName.second;
        }
    }
    return "UNKNOWN";
}

inline const char* BLayoutToString(BLayout layout) { return layout == BLayout::RowMajor ? "row_major" : "col_major"; }

inline const char* SLayoutToString(SLayout layout)
{
    switch (layout) {
        case SLayout::NoneBox:
            return "none_box";
        case SLayout::RowMajor:
            return "row_major";
        case SLayout::ColMajor:
            return "col_major";
    }
    return "unknown";
}

inline const char* CompactModeToString(CompactMode mode)
{
    switch (mode) {
        case CompactMode::Null:
            return "null";
        case CompactMode::Normal:
            return "normal";
        case CompactMode::RowPlusOne:
            return "row_plus_one";
        case CompactMode::RowAlignedPadding:
            return "row_aligned_padding";
    }
    return "unknown";
}

template <typename T>
std::string TraceDTypeName()
{
    using Decayed = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<Decayed, bool>) {
        return "bool";
    } else if constexpr (std::is_same_v<Decayed, int8_t>) {
        return "int8";
    } else if constexpr (std::is_same_v<Decayed, uint8_t>) {
        return "uint8";
    } else if constexpr (std::is_same_v<Decayed, int16_t>) {
        return "int16";
    } else if constexpr (std::is_same_v<Decayed, uint16_t>) {
        return "uint16";
    } else if constexpr (std::is_same_v<Decayed, int32_t>) {
        return "int32";
    } else if constexpr (std::is_same_v<Decayed, uint32_t>) {
        return "uint32";
    } else if constexpr (std::is_same_v<Decayed, int64_t>) {
        return "int64";
    } else if constexpr (std::is_same_v<Decayed, uint64_t>) {
        return "uint64";
    } else if constexpr (std::is_same_v<Decayed, half> || std::is_same_v<Decayed, aclFloat16>) {
        return "float16";
    } else if constexpr (std::is_same_v<Decayed, bfloat16_t>) {
        return "bfloat16";
    } else if constexpr (std::is_same_v<Decayed, float>) {
        return "float32";
    } else if constexpr (std::is_same_v<Decayed, double>) {
        return "float64";
    } else if constexpr (std::is_same_v<Decayed, int4b_t>) {
        return "int4";
    } else if constexpr (std::is_enum_v<Decayed>) {
        return "enum";
    } else if constexpr (std::is_integral_v<Decayed>) {
        return std::string(std::is_signed_v<Decayed> ? "int" : "uint") + std::to_string(sizeof(Decayed) * 8);
    } else {
        return "unknown";
    }
}

template <typename T>
std::string ScalarValueToString(T value)
{
    using Decayed = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<Decayed, bool>) {
        return value ? "true" : "false";
    } else if constexpr (std::is_enum_v<Decayed>) {
        using Underlying = std::underlying_type_t<Decayed>;
        return std::to_string(static_cast<Underlying>(value));
    } else if constexpr (
        std::is_same_v<Decayed, half> || std::is_same_v<Decayed, aclFloat16> || std::is_same_v<Decayed, bfloat16_t>) {
        std::ostringstream os;
        os << static_cast<float>(value);
        return os.str();
    } else if constexpr (std::is_floating_point_v<Decayed>) {
        std::ostringstream os;
        os << value;
        return os.str();
    } else if constexpr (std::is_integral_v<Decayed>) {
        if constexpr (std::is_same_v<Decayed, int8_t> || std::is_same_v<Decayed, uint8_t>) {
            return std::to_string(static_cast<int32_t>(value));
        } else {
            return std::to_string(value);
        }
    } else {
        return "<unsupported>";
    }
}

template <typename T>
inline constexpr bool kTraceTileLikeOperand =
    is_tile_data_v<std::remove_cvref_t<T>> || is_conv_tile_v<std::remove_cvref_t<T>> ||
    is_global_data_v<std::remove_cvref_t<T>>;

template <typename T>
inline constexpr bool kTraceTilePointerOperand =
    std::is_pointer_v<std::remove_cvref_t<T>> && (is_tile_data_v<std::remove_pointer_t<std::remove_cvref_t<T>>> ||
                                                  is_conv_tile_v<std::remove_pointer_t<std::remove_cvref_t<T>>> ||
                                                  is_global_data_v<std::remove_pointer_t<std::remove_cvref_t<T>>>);

template <typename T>
inline constexpr bool kTraceScalarOperand =
    std::is_arithmetic_v<std::remove_cvref_t<T>> || std::is_enum_v<std::remove_cvref_t<T>>;

template <typename TileData>
std::string TileLayoutString()
{
    if constexpr (TileData::SFractal == SLayout::NoneBox && TileData::Compact == CompactMode::Null) {
        return TileData::BFractal == BLayout::RowMajor ? "ND" : "DN";
    }

    std::ostringstream os;
    bool need_sep = false;
    if constexpr (TileData::BFractal != BLayout::RowMajor) {
        os << "outer=" << BLayoutToString(TileData::BFractal);
        need_sep = true;
    }
    if constexpr (TileData::SFractal != SLayout::NoneBox) {
        if (need_sep) {
            os << ",";
        }
        os << "inner=" << SLayoutToString(TileData::SFractal);
        need_sep = true;
    }
    if constexpr (TileData::Compact != CompactMode::Null) {
        if (need_sep) {
            os << ",";
        }
        os << "compact=" << CompactModeToString(TileData::Compact);
        need_sep = true;
    }
    if constexpr (TileData::SFractal != SLayout::NoneBox) {
        if (need_sep) {
            os << ",";
        }
        os << "fractal_size=" << TileData::SFractalSize;
        need_sep = true;
    }
    if (!need_sep) {
        return TileData::BFractal == BLayout::RowMajor ? "ND" : "DN";
    }
    return os.str();
}

template <typename TileData>
TileOperandTrace CaptureTileOperand(TileData& tile)
{
    using CleanTileData = std::remove_cv_t<std::remove_reference_t<TileData>>;
    TileOperandTrace operand;
    if constexpr (requires { tile.GetAssignedAddress(); }) {
        operand.address = tile.GetAssignedAddress();
    } else {
        operand.address = reinterpret_cast<std::uintptr_t>(tile.data());
    }

    if constexpr (is_tile_data_v<CleanTileData>) {
        operand.shape = {static_cast<int64_t>(tile.GetValidRow()), static_cast<int64_t>(tile.GetValidCol())};
        operand.layout = TileLayoutString<CleanTileData>();
        operand.dtype = TraceDTypeName<typename CleanTileData::DType>();
    } else if constexpr (is_conv_tile_v<CleanTileData>) {
        operand.shape.reserve(CleanTileData::totalDimCount);
        for (int dim = 0; dim < CleanTileData::totalDimCount; ++dim) {
            operand.shape.push_back(tile.GetShape(dim));
        }
        operand.layout = LayoutToString(CleanTileData::layout);
        operand.dtype = TraceDTypeName<typename CleanTileData::DType>();
    } else {
        operand.shape.reserve(GlobalTensorDim::TOTAL_DIM);
        for (int dim = 0; dim < GlobalTensorDim::TOTAL_DIM; ++dim) {
            operand.shape.push_back(tile.GetShape(dim));
        }
        operand.layout = LayoutToString(CleanTileData::layout);
        operand.dtype = TraceDTypeName<typename CleanTileData::RawDType>();
    }

    return operand;
}

inline std::string JsonEscape(std::string_view text)
{
    std::string escaped;
    escaped.reserve(text.size() + 8);
    for (char ch : text) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += ch;
                break;
        }
    }
    return escaped;
}

inline std::string HexAddress(std::uintptr_t address)
{
    std::ostringstream os;
    os << "0x" << std::hex << address;
    return os.str();
}

inline void WriteTileOperandJson(std::ostream& os, const TileOperandTrace& operand)
{
    os << "{\"address\":\"" << HexAddress(operand.address) << "\",\"shape\":[";
    for (std::size_t dim = 0; dim < operand.shape.size(); ++dim) {
        if (dim != 0) {
            os << ",";
        }
        os << operand.shape[dim];
    }
    os << "],\"layout\":\"" << JsonEscape(operand.layout) << "\",\"dtype\":\"" << JsonEscape(operand.dtype) << "\"}";
}

inline void WriteTileOperandsJson(std::ostream& os, const std::vector<TileOperandTrace>& operands)
{
    for (std::size_t i = 0; i < operands.size(); ++i) {
        if (i != 0) {
            os << ",";
        }
        WriteTileOperandJson(os, operands[i]);
    }
}

inline void DumpInstructionTraceJson(std::ostream& os)
{
    const auto records = CopyInstructionTraceRecords();
    for (const auto& record : records) {
        os << "{\"block_idx\":" << record.block_idx << ",\"sequence_id\":" << record.sequence_id << ",\"opcode\":\""
           << JsonEscape(record.opcode) << "\",\"input_tiles\":[";
        WriteTileOperandsJson(os, record.input_tiles);
        os << "],\"scalar_inputs\":[";
        for (std::size_t i = 0; i < record.scalar_inputs.size(); ++i) {
            const auto& operand = record.scalar_inputs[i];
            if (i != 0) {
                os << ",";
            }
            os << "{\"dtype\":\"" << JsonEscape(operand.dtype) << "\",\"value\":\"" << JsonEscape(operand.value)
               << "\"}";
        }
        os << "],\"output_tiles\":[";
        WriteTileOperandsJson(os, record.output_tiles);
        os << "]}\n";
    }
}

class PtoInstrTraceScope {
public:
    template <typename... Args>
    explicit PtoInstrTraceScope(std::string_view opcode, std::size_t output_tile_count, Args&&... args)
    {
        if constexpr (!kInstructionTraceEnabled) {
            (void)opcode;
            (void)output_tile_count;
            (void)sizeof...(Args);
        } else {
            Initialize(opcode);
            remaining_output_tiles_ = output_tile_count;
            (VisitArg(std::forward<Args>(args)), ...);
        }
    }

    template <typename... Args>
    explicit PtoInstrTraceScope(std::string_view opcode, std::string_view roles, Args&&... args)
    {
        if constexpr (!kInstructionTraceEnabled) {
            (void)opcode;
            (void)roles;
            (void)sizeof...(Args);
        } else {
            Initialize(opcode);
            std::size_t arg_index = 0;
            (VisitArgWithRole(arg_index < roles.size() ? roles[arg_index++] : 'A', std::forward<Args>(args)), ...);
        }
    }

    ~PtoInstrTraceScope()
    {
        if constexpr (!kInstructionTraceEnabled) {
            return;
        } else if (active_) {
            for (const auto& capture : output_tile_captures_) {
                capture(record_);
            }
            AppendInstructionTraceRecord(std::move(record_));
        }
    }

    PtoInstrTraceScope(const PtoInstrTraceScope&) = delete;
    PtoInstrTraceScope& operator=(const PtoInstrTraceScope&) = delete;

private:
    void Initialize(std::string_view opcode)
    {
        active_ = true;
        record_.block_idx = ::get_block_idx();
        record_.sequence_id = ReserveInstructionTraceSequenceId();
        record_.opcode = std::string(opcode);
    }

    template <typename TileData>
    void AddOutputCapture(TileData& tile)
    {
        output_tile_captures_.push_back(
            [&tile](InstructionTraceRecord& record) { record.output_tiles.push_back(CaptureTileOperand(tile)); });
    }

    template <typename TileData>
    void AddOutputCapture(TileData* tile)
    {
        output_tile_captures_.push_back([tile](InstructionTraceRecord& record) {
            if (tile != nullptr) {
                record.output_tiles.push_back(CaptureTileOperand(*tile));
            }
        });
    }

    template <typename T>
    void VisitArg(T&& arg)
    {
        using Arg = std::remove_cvref_t<T>;
        if constexpr (kTraceTileLikeOperand<Arg>) {
            if (remaining_output_tiles_ > 0) {
                AddOutputCapture(arg);
                --remaining_output_tiles_;
            } else {
                record_.input_tiles.push_back(CaptureTileOperand(arg));
            }
        } else if constexpr (kTraceTilePointerOperand<Arg>) {
            if (arg == nullptr) {
                return;
            }
            if (remaining_output_tiles_ > 0) {
                AddOutputCapture(arg);
                --remaining_output_tiles_;
            } else {
                record_.input_tiles.push_back(CaptureTileOperand(*arg));
            }
        } else if constexpr (kTraceScalarOperand<Arg>) {
            record_.scalar_inputs.push_back({TraceDTypeName<Arg>(), ScalarValueToString(arg)});
        }
    }

    template <typename T>
    void VisitArgWithRole(char role, T&& arg)
    {
        switch (role) {
            case 'O':
                VisitOutputArg(std::forward<T>(arg));
                break;
            case 'I':
                VisitInputArg(std::forward<T>(arg));
                break;
            case 'B':
                VisitInputArg(std::forward<T>(arg));
                VisitOutputArg(std::forward<T>(arg));
                break;
            case 'X':
                break;
            default:
                VisitArg(std::forward<T>(arg));
                break;
        }
    }

    template <typename T>
    void VisitInputArg(T&& arg)
    {
        using Arg = std::remove_cvref_t<T>;
        if constexpr (kTraceTileLikeOperand<Arg>) {
            record_.input_tiles.push_back(CaptureTileOperand(arg));
        } else if constexpr (kTraceTilePointerOperand<Arg>) {
            if (arg != nullptr) {
                record_.input_tiles.push_back(CaptureTileOperand(*arg));
            }
        } else if constexpr (kTraceScalarOperand<Arg>) {
            record_.scalar_inputs.push_back({TraceDTypeName<Arg>(), ScalarValueToString(arg)});
        }
    }

    template <typename T>
    void VisitOutputArg(T&& arg)
    {
        using Arg = std::remove_cvref_t<T>;
        if constexpr (kTraceTileLikeOperand<Arg>) {
            AddOutputCapture(arg);
        } else if constexpr (kTraceTilePointerOperand<Arg>) {
            if (arg != nullptr) {
                AddOutputCapture(arg);
            }
        } else if constexpr (kTraceScalarOperand<Arg>) {
            record_.scalar_inputs.push_back({TraceDTypeName<Arg>(), ScalarValueToString(arg)});
        }
    }

    bool active_ = false;
    std::size_t remaining_output_tiles_ = 0;
    InstructionTraceRecord record_{};
    std::vector<std::function<void(InstructionTraceRecord&)>> output_tile_captures_;
};

#endif // TRACE_DETAIL_INL_HPP
