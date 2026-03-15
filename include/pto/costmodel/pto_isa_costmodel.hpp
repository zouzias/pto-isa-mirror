#ifndef PTO_ISA_COST_MODEL_HPP
#define PTO_ISA_COST_MODEL_HPP

#include <type_traits>
#include <pto/common/pto_tile.hpp>

namespace pto {
inline int sum_repeat_times;

struct AddOp {
    PTO_INTERNAL static void BinInstr(uint8_t repeats){
        sum_repeat_times += static_cast<int>(repeats);
    }
}
} // namespace pto

#endif