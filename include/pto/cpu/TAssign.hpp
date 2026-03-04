#ifndef TTILE_ASSIGN
#define TTILE_ASSIGN
#include <cstdint>
#include <cstring>
#include <pto/common/pto_tile.hpp>
#include <pto/cpu/TileBufferManager.hpp>

namespace pto {

template <typename T, typename AddrType>
PTO_INTERNAL void TASSIGN_IMPL(T &obj, AddrType addr)
{
    if constexpr (is_tile_data_v<T>) {
        using DType = typename T::DType;
        constexpr std::size_t numElements = T::Rows * T::Cols;
        std::size_t ubOffset = static_cast<std::size_t>(addr);
        
        // Check if this is the first TASSIGN to this offset this phase
        bool isFirst = TileBufferManager::Instance().IsFirstAssign(ubOffset);
        
        // Get shared buffer
        DType* shared = TileBufferManager::Instance().GetSharedBuffer<DType>(ubOffset, numElements);
        
        // Only copy data on first assign (source tile has the data we want to share)
        if (isFirst && obj.data() != nullptr && obj.data() != shared) {
            std::memcpy(shared, obj.data(), numElements * sizeof(DType));
        }
        
        // Redirect tile to shared buffer
        obj.data() = shared;
    } else {
        static_assert(is_global_data_v<T>, "Only Tile and GlobalTensor data types are supported.");
        static_assert(std::is_pointer_v<AddrType>, "GlobalTensor can only be assigned with address of pointer type.");
        static_assert(std::is_same_v<std::remove_cv_t<std::remove_pointer_t<AddrType>>, typename T::DType>,
                      "GlobalTensor can only be assigned with pointer of same data type.");
        obj.SetAddr(addr);
    }
}

// Call at start of each loop iteration to reset TASSIGN first-assign tracking
inline void TASSIGN_RESET_PHASE() {
    TileBufferManager::Instance().ResetPhase();
}

} // namespace pto
#endif
