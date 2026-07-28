#ifndef PTO_CPU_MEMORY_HPP
#define PTO_CPU_MEMORY_HPP

#if defined(__CPU_SIM)
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <pto/common/utils.hpp>

namespace pto::cpu {

#if defined(__CPU_SIM)
PTO_INTERNAL bool IsMappedAddress(const void* addr)
{
    if (addr == nullptr) {
        return false;
    }
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0) {
        return true;
    }
    const auto raw = reinterpret_cast<std::uintptr_t>(addr);
    void* page = reinterpret_cast<void*>(raw & ~(static_cast<std::uintptr_t>(pageSize) - 1));
    unsigned char vec = 0;
    errno = 0;
    if (mincore(page, static_cast<std::size_t>(pageSize), &vec) == 0) {
        return true;
    }
    return errno != ENOMEM;
}
#endif

} // namespace pto::cpu

#endif
