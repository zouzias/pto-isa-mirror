#ifndef PTO_MOCKER_COMMON_ARCH_SELECT_HPP
#define PTO_MOCKER_COMMON_ARCH_SELECT_HPP

#if !defined(__NPU_ARCH__)
#error "__NPU_ARCH__ must be defined for PTO costmodel."
#elif (__NPU_ARCH__ == 2201)
#include <pto/costmodel/a2a3/cce_stub.hpp>
#elif (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 3510)
#include <pto/costmodel/a5/cce_stub.hpp>
#else
#error "Unsupported __NPU_ARCH__ for PTO costmodel."
#endif

#endif
