#ifndef PTO_COMM_TALLREDUCE_HPP
#define PTO_COMM_TALLREDUCE_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/backend/backend_selector.hpp"

namespace pto {
namespace comm {


// 主实现：使用类型萃取，保持简单的调用接口
// 用户调用时只需传入 ParallelGroup<GlobalData>，编译器会自动推导类型
template <BackendKind backend = BackendKind::Shmem, typename ParallelGroupType, typename GlobalDstData>
AICORE void TALLREDUCE_IMPL(ParallelGroupType &pg, GlobalDstData &dstGlobal)
{
    // 使用类型萃取获取 GlobalData 类型（如果不是 ParallelGroup<...>，这里会触发编译错误）
    using GlobalData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    
    // Check PG size 
    PTO_ASSERT(pg.size() > 1, "ParallelGroup size must be greater than 1!");
    
    // Check PG tensors have the same shape (运行时检查)
    for (int i = 1; i < pg.size(); i++) {
        PTO_ASSERT(pg[i].GetShape(0) == pg[0].GetShape(0) &&
                   pg[i].GetShape(1) == pg[0].GetShape(1) &&
                   pg[i].GetShape(2) == pg[0].GetShape(2) &&
                   pg[i].GetShape(3) == pg[0].GetShape(3) &&
                   pg[i].GetShape(4) == pg[0].GetShape(4),
                   "All tensors in ParallelGroup must have the same shape!");
    }
    
    BackendSelector<backend>::type::template AllReduce(pg, dstGlobal);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TALLREDUCE_HPP
