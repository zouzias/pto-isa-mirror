/**
 * Compatibility shim: some files include "pto/comm/async/async_types.hpp"
 * while the canonical header lives under async_common/. Provide a thin
 * forwarding header to avoid fragile include-path workarounds.
 */
#ifndef PTO_COMM_ASYNC_ASYNC_TYPES_SHIM_HPP
#define PTO_COMM_ASYNC_ASYNC_TYPES_SHIM_HPP

#include "pto/comm/async_common/async_types.hpp"

#endif // PTO_COMM_ASYNC_ASYNC_TYPES_SHIM_HPP
