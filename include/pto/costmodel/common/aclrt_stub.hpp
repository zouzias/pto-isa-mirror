/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_COMMON_ACLRT_STUB_HPP
#define PTO_MOCKER_COMMON_ACLRT_STUB_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <pto/costmodel/common/qualifiers.hpp>

// cpu_stub.hpp keeps these names as empty macros for CPU_SIM. The cost-model
// runtime needs return values, so replace only the macros visible in this
// translation unit with Host-side functions.
#ifdef aclInit
#undef aclInit
#endif
#ifdef aclrtSetDevice
#undef aclrtSetDevice
#endif
#ifdef aclrtCreateStream
#undef aclrtCreateStream
#endif

namespace pto::costmodel::aclrt_stub {

inline constexpr aclError kInvalidParameter = 145000;
inline constexpr aclError kAllocationFailed = 207001;

inline int& CurrentDevice()
{
    static thread_local int deviceId = 0;
    return deviceId;
}

} // namespace pto::costmodel::aclrt_stub

inline aclError aclInit(const char*) { return ACL_SUCCESS; }

inline aclError aclFinalize() { return ACL_SUCCESS; }

inline const char* aclGetRecentErrMsg() { return ""; }

inline aclError aclrtSetDevice(int deviceId)
{
    pto::costmodel::aclrt_stub::CurrentDevice() = deviceId;
    return ACL_SUCCESS;
}

inline int aclrtGetDevice(int* deviceId)
{
    if (deviceId == nullptr) {
        return pto::costmodel::aclrt_stub::kInvalidParameter;
    }
    *deviceId = pto::costmodel::aclrt_stub::CurrentDevice();
    return ACL_SUCCESS;
}

inline aclError aclrtGetDeviceCount(uint32_t* count)
{
    if (count == nullptr) {
        return pto::costmodel::aclrt_stub::kInvalidParameter;
    }
    *count = 1;
    return ACL_SUCCESS;
}

inline aclError aclrtResetDevice(int deviceId)
{
    if (pto::costmodel::aclrt_stub::CurrentDevice() == deviceId) {
        pto::costmodel::aclrt_stub::CurrentDevice() = 0;
    }
    return ACL_SUCCESS;
}

inline aclError aclrtGetCurrentContext(aclrtContext* ctx)
{
    if (ctx == nullptr) {
        return pto::costmodel::aclrt_stub::kInvalidParameter;
    }
    *ctx = nullptr;
    return ACL_SUCCESS;
}

inline aclError aclrtCreateStream(aclrtStream* stream)
{
    if (stream == nullptr) {
        return pto::costmodel::aclrt_stub::kInvalidParameter;
    }
    *stream = std::calloc(1, 1);
    return *stream == nullptr ? pto::costmodel::aclrt_stub::kAllocationFailed : ACL_SUCCESS;
}

inline aclError aclrtCreateStreamWithConfig(aclrtStream* stream, uint32_t, uint32_t)
{
    return aclrtCreateStream(stream);
}

inline aclError aclrtDestroyStream(aclrtStream stream)
{
    std::free(stream);
    return ACL_SUCCESS;
}

inline aclError aclrtStreamGetId(aclrtStream stream, int32_t* streamId)
{
    if (stream == nullptr || streamId == nullptr) {
        return pto::costmodel::aclrt_stub::kInvalidParameter;
    }
    *streamId = 0;
    return ACL_SUCCESS;
}

inline aclError aclrtMallocHost(void** ptr, size_t size)
{
    if (ptr == nullptr || size == 0) {
        return pto::costmodel::aclrt_stub::kInvalidParameter;
    }
    *ptr = std::calloc(1, size);
    return *ptr == nullptr ? pto::costmodel::aclrt_stub::kAllocationFailed : ACL_SUCCESS;
}

inline aclError aclrtMalloc(void** ptr, size_t size, int) { return aclrtMallocHost(ptr, size); }

inline aclError aclrtFree(void* ptr)
{
    std::free(ptr);
    return ACL_SUCCESS;
}

inline aclError aclrtFreeHost(void* ptr)
{
    std::free(ptr);
    return ACL_SUCCESS;
}

inline aclError aclrtMemcpy(void* dst, size_t dstSize, const void* src, size_t count, int)
{
    if (count == 0) {
        return ACL_SUCCESS;
    }
    if (dst == nullptr || src == nullptr || count > dstSize) {
        return pto::costmodel::aclrt_stub::kInvalidParameter;
    }
    std::memcpy(dst, src, count);
    return ACL_SUCCESS;
}

inline aclError aclrtMemset(void* dst, size_t dstSize, int value, size_t count)
{
    if (count == 0) {
        return ACL_SUCCESS;
    }
    if (dst == nullptr || count > dstSize) {
        return pto::costmodel::aclrt_stub::kInvalidParameter;
    }
    std::fill_n(static_cast<uint8_t*>(dst), count, static_cast<uint8_t>(value));
    return ACL_SUCCESS;
}

inline aclError aclrtSynchronizeStream(aclrtStream) { return ACL_SUCCESS; }

inline aclError aclrtSetStreamAttribute(aclrtStream, int, const void*) { return ACL_SUCCESS; }

inline aclError aclrtCmoAsync(void*, size_t, int, aclrtStream) { return ACL_SUCCESS; }

#endif
