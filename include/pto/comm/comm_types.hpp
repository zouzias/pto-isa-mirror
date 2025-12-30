#ifndef PTO_COMM_COMM_TYPES_HPP
#define PTO_COMM_COMM_TYPES_HPP

#include <cstdint>

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

// 后端枚举，便于后续扩展
enum class BackendKind : uint8_t {
    Shmem = 0,
};

} // namespace comm
} // namespace pto

#endif // PTO_COMM_COMM_TYPES_HPP

