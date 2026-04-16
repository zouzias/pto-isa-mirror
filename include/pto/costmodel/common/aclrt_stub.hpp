#ifndef PTO_MOCKER_COMMON_ACLRT_STUB_HPP
#define PTO_MOCKER_COMMON_ACLRT_STUB_HPP

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <pto/costmodel/common/qualifiers.hpp>

inline int aclInit(...) { return 0; }
inline int aclFinalize(...) { return 0; }
inline int aclrtSetDevice(...) { return 0; }
inline int aclrtResetDevice(...) { return 0; }
inline int aclrtGetDevice(int *deviceId)
{
    if (deviceId != nullptr) {
        *deviceId = 0;
    }
    return 0;
}
inline int aclrtGetCurrentContext(aclrtContext *ctx)
{
    if (ctx != nullptr) {
        *ctx = nullptr;
    }
    return 0;
}
inline int aclrtCreateStream(aclrtStream *stream)
{
    if (stream != nullptr) {
        *stream = nullptr;
    }
    return 0;
}
inline int aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t, uint32_t)
{
    if (stream != nullptr) {
        *stream = nullptr;
    }
    return 0;
}
inline int aclrtDestroyStream(aclrtStream)
{
    return 0;
}
inline int aclrtSynchronizeStream(aclrtStream)
{
    return 0;
}
inline int aclrtStreamGetId(aclrtStream, uint32_t *streamId)
{
    if (streamId != nullptr) {
        *streamId = 0;
    }
    return 0;
}
inline int aclrtSetStreamAttribute(aclrtStream, int, const void *)
{
    return 0;
}
inline int aclrtMallocHost(void **ptr, size_t size)
{
    *ptr = std::malloc(size);
    return (*ptr == nullptr) ? 1 : 0;
}
inline int aclrtMalloc(void **ptr, size_t size, uint32_t)
{
    return aclrtMallocHost(ptr, size);
}
inline int aclrtMemcpy(void *dst, size_t dstSize, const void *src, size_t srcSize, int)
{
    const size_t bytes = (srcSize < dstSize) ? srcSize : dstSize;
    std::memcpy(dst, src, bytes);
    return 0;
}
inline int aclrtMemset(void *dst, size_t dstSize, int value, size_t count)
{
    const size_t bytes = (count < dstSize) ? count : dstSize;
    std::memset(dst, value, bytes);
    return 0;
}
inline int aclrtFree(void *ptr)
{
    std::free(ptr);
    return 0;
}
inline int aclrtFreeHost(void *ptr)
{
    std::free(ptr);
    return 0;
}
inline int aclrtCmoAsync(void *, size_t, int, aclrtStream)
{
    return 0;
}

#endif
