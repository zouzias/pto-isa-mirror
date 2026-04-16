#ifndef PTO_MOCKER_COMMON_ARCH_SELECT_HPP
#define PTO_MOCKER_COMMON_ARCH_SELECT_HPP

#if !defined(__NPU_ARCH__)
#error "__NPU_ARCH__ must be defined for PTO costmodel."
#elif (__NPU_ARCH__ == 2201)
#include <pto/costmodel/a2a3/cce_costmodel.hpp>
#else
#error "PTO costmodel only supports __NPU_ARCH__ == 2201 (A2/A3)."
#endif

#endif
