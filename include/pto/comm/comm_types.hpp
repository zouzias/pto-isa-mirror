#ifndef PTO_COMM_COMM_TYPES_HPP
#define PTO_COMM_COMM_TYPES_HPP

#include <cstdint>
#include <type_traits>

#include "pto/common/type.hpp"

namespace pto {
namespace comm {

// 2D拷贝描述，单位均为元素个数
struct Copy2DParams {
    uint32_t repeat {1};         // 行数或重复次数
    uint32_t lenElems {0};       // 每次拷贝的元素数
    uint32_t srcStrideElems {0}; // 源步长（元素数）
    uint32_t dstStrideElems {0}; // 目的步长（元素数）

    static Copy2DParams Contiguous(uint32_t elemCount, uint32_t repeatCount = 1)
    {
        Copy2DParams p;
        p.repeat = repeatCount;
        p.lenElems = elemCount;
        p.srcStrideElems = elemCount;
        p.dstStrideElems = elemCount;
        return p;
    }
};

// ParallelGroup: 用于把参与集合通信的 GlobalTensor 组成一个 "组"(team)。
//
// 说明：
// - 这里仅做轻量"视图"封装：不在设备侧动态分配内存，避免引入 `std::vector` 等不支持的容器。
// - 组内的每个元素一般表示"该 team rank 对应的 GlobalTensor 视图"（通常通过 SetRank 映射到 world rank）。
template <typename GlobalData>
struct ParallelGroup {
    using value_type = GlobalData;  // 类型别名，用于类型萃取
    
    GlobalData **tensors {nullptr}; // 指向外部数组：tensors[teamRank] -> GlobalTensor*
    int nranks {0};
    int my_rank {-1};

    constexpr ParallelGroup() = default;
    AICORE constexpr ParallelGroup(GlobalData **tensorPtrs, int size, int rank_id) : tensors(tensorPtrs), nranks(size), my_rank(rank_id) {}

    AICORE constexpr int size() const { return nranks; }
    AICORE constexpr bool empty() const { return nranks == 0; }

    AICORE constexpr int GetRank() const { return my_rank; }
    AICORE constexpr int GetSize() const { return nranks; }

    AICORE constexpr GlobalData &operator[](int teamRank) { return *tensors[teamRank]; }
    AICORE constexpr const GlobalData &operator[](int teamRank) const { return *tensors[teamRank]; }
};

// 类型萃取：从 ParallelGroup<GlobalData> 中提取 GlobalData 类型
template <typename T>
struct ParallelGroupTraits {
    // 如果不是 ParallelGroup，这里会触发编译错误
    static_assert(std::is_same_v<T, void>, 
                  "TALLREDUCE: T must be ParallelGroup<GlobalData>");
};

template <typename GlobalData>
struct ParallelGroupTraits<ParallelGroup<GlobalData>> {
    using GlobalDataType = GlobalData;
};

// 后端枚举，便于后续扩展
enum class BackendKind : uint8_t {
    Shmem = 0,
};

} // namespace comm
} // namespace pto

#endif // PTO_COMM_COMM_TYPES_HPP

