/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aicore_print.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "aicore_data.h"

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#ifdef __TILE_FWK_HOST__
#include <string>
#include <sstream>
#include <securec.h>
#endif

enum NodeTy { END, NORMAL, FLOAT, INT, CHAR, STRING, POINTER };

struct LogContext {
    void (*PrintInt)(LogContext *ctx, __gm__ const char **fmt, int64_t val);
    void (*PrintFloat)(LogContext *ctx, __gm__ const char **fmt, float val);
    void (*Print)(LogContext *ctx, __gm__ const char *fmt);
};

template <typename T>
INLINE void __AiCorePrint(LogContext *ctx, __gm__ const char **fmt, T val) {
    if constexpr (std::is_integral_v<T>) {
        ctx->PrintInt(ctx, fmt, static_cast<int64_t>(val));
    } else if constexpr (std::is_floating_point_v<T>) {
        ctx->PrintFloat(ctx, fmt, static_cast<float>(val));
    } else if constexpr (std::is_pointer_v<T>) {
        ctx->PrintInt(ctx, fmt, reinterpret_cast<int64_t>(val));
    }
}

template <typename... Ts>
INLINE void AiCoreLogF(LogContext *ctx, __gm__ const char *fmt, Ts... Args) {
    if (ctx && fmt) {
        (__AiCorePrint(ctx, &fmt, Args), ...);
        ctx->Print(ctx, fmt);
    }
}

struct AicoreLogger {
    struct Remote {
        int64_t head_;
        int64_t tail_;
    };

    static __aicore__ void __PrintInt(LogContext *ctx, __gm__ const char **fmt, int64_t val) {
        auto self = reinterpret_cast<AicoreLogger *>(ctx);
        if (self) {
            self->PrintInt(fmt, val);
        }
    }

    static __aicore__ void __PrintFloat(LogContext *ctx, __gm__ const char **fmt, float val) {
        auto self = reinterpret_cast<AicoreLogger *>(ctx);
        if (self) {
            self->PrintFloat(fmt, val);
        }
    }

    static __aicore__ void __Print(LogContext *ctx, __gm__ const char *fmt) {
        auto self = reinterpret_cast<AicoreLogger *>(ctx);
        if (self) {
            self->Print(fmt);
        }
    }

    __aicore__ void Init(__gm__ uint8_t *buf, size_t n) {
        remote_ = reinterpret_cast<volatile __gm__ Remote *>(buf);
        remote_->head_ = remote_->tail_ = 0;
        head_ = tail_ = 0;
        size_ = n - sizeof(Remote);
        data_ = buf + sizeof(Remote);
        ctx.PrintInt = __PrintInt;
        ctx.PrintFloat = __PrintFloat;
        ctx.Print = __Print;
    }

    __aicore__ __gm__ uint8_t *GetBuffer()  {
        return data_ - sizeof(Remote);
    }

    __aicore__ void PrintInt(__gm__ const char **fmt, int64_t val) {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);
        if (idx == -1) {
            return;
        }
        switch (curFmt[idx++]) {
            case 's': {
                auto tmp = reinterpret_cast<__gm__ const char *>(val);
                if (tmp == nullptr) {
                    tmp = "<null>";
                }
                Encode(STRING, reinterpret_cast<__gm__ const uint8_t *>(tmp), Length(tmp), *fmt, idx);
                break;
            }
            case 'd':
            case 'i':
            case 'x':
            case 'X':
            case 'o':
            case 'u': {
                Encode(INT, reinterpret_cast<uint8_t *>(&val), sizeof(val), *fmt, idx);
                break;
            }
            case 'p': {
                Encode(POINTER, reinterpret_cast<uint8_t *>(&val), sizeof(val), *fmt, idx);
                break;
            }
            case 'c': {
                char c = static_cast<char>(val);
                Encode(CHAR, reinterpret_cast<uint8_t *>(&c), 1, *fmt, idx);
                break;
            }
            default: Encode(NORMAL, static_cast<uint8_t *>(nullptr), 0, *fmt, idx); break;
        }

        *fmt = *fmt + idx;
    }

    __aicore__ void PrintFloat(__gm__ const char **fmt, float val) {
        auto curFmt = *fmt;
        auto idx = ParseNextFormat(*fmt);
        if (idx == -1) {
            return;
        }
        switch (curFmt[idx++]) {
            case 'u': {
                Encode(FLOAT, reinterpret_cast<uint8_t *>(&val), sizeof(val), *fmt, idx);
                break;
                default: Encode(NORMAL, static_cast<uint8_t *>(nullptr), 0, *fmt, idx); break;
            }
        }
        *fmt = *fmt + idx;
    }

    __aicore__ void Print(__gm__ const char *str) {
        auto n = Length(str);
        if (n) {
            Encode(NORMAL, reinterpret_cast<const __gm__ uint8_t *>(str), n, str, n);
        }
        Encode(END);
        Sync();
    }

    __aicore__ void Sync() {
#ifndef __TILE_FWK_HOST__
        int64_t delta = (int64_t)(&data_[remote_->head_ % size_]) & (CACHE_LINE_SIZE -1);
        int64_t off = remote_->head_ - delta;
        while (off < head_) {
            dcci(&data_[off % size_], SINGLE_CACHE_LINE, CACHELINE_OUT);
            off += CACHE_LINE_SIZE;
        }
        remote_->head_ = head_;
        remote_->tail_ = tail_;
        dcci(remote_, SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
        remote_->head_ = head_;
        remote_->tail_ = tail_;
#endif
    }

    INLINE LogContext *context() { return &ctx; }

#ifdef __TILE_FWK_HOST__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
    int Read(char *buf, size_t maxSize) {
        size_t size = 0;
        head_ = remote_->head_;
        if (tail_ < remote_->tail_) {
            // lose some data
            tail_ = remote_->tail_;
        }
        while (tail_ != head_) {
            auto type = Read<uint8_t>(tail_++);
            if (type == END) {
                if (size == 0)
                    continue;
                else
                    return size;
            } else if (maxSize == 0) {
                continue;
            }

            auto valOff = tail_ + sizeof(short);
            tail_ += Read<short>(tail_) + sizeof(short);
            auto fmtOff = tail_ + sizeof(short);
            std::string fmt = ReadString(fmtOff);
            tail_ += Read<short>(tail_) + sizeof(short);
            int n = 0;
            switch (type) {
                case NORMAL: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), 0); break;
                case FLOAT: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), Read<float>(valOff)); break;
                case INT: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), Read<int64_t>(valOff)); break;
                case CHAR: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), Read<char>(valOff)); break;
                case STRING: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), ReadString(valOff).c_str()); break;
                case POINTER: n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), Read<int64_t>(valOff)); break;
                default: if (n) { buf[0] = '?'; n = 1;} break;
            }
            buf += n;
            size += n;
            maxSize -= n;
        }
        return 0;
    }
#pragma GCC diagnostic pop
#endif

private:
    __aicore__ int64_t ParseNextFormat(__gm__ const char *fmt) {
        int64_t idx = 0;
        while (fmt[idx]) {
            if (fmt[idx] == '%') {
                if (fmt[idx + 1] == '%') {
                    idx += 2;
                } else {
                    break;
                }
            } else {
                idx++;
            }
        }

        if (!fmt[idx]) {
            return -1;
        }

        idx++;

        // skip fmt
        while (fmt[idx]) {
            if (fmt[idx] != '0' && fmt[idx] != '+' && fmt[idx] != '-' && fmt[idx] != ' ' && fmt[idx] != '#') {
                break;
            }
            idx++;
        }

        // width
        while (IsDigit(fmt[idx])) {
            idx++;
        }

        // precision
        if (fmt[idx] == '.') {
            idx++;
            while (IsDigit(fmt[idx])) {
                idx++;
            }
        }

        // Length
        if (fmt[idx] == 'l' || fmt[idx] == 'z' || fmt[idx] == 'h') {
            idx++;
            if (fmt[idx] == 'l')
                idx++;
        }

        return fmt[idx] ? idx : -1;
    }

    template <typename T>
    INLINE T Read(int64_t off) {
        T val;
        char tmp[sizeof(T)];
        for (size_t i = 0; i < sizeof(T); i++) {
            tmp[i] = data_[(off + i) % size_];
        }
        val = *reinterpret_cast<T *>(tmp);
        return val;
    }

#ifdef __TILE_FWK_HOST__
    std::string ReadString(int64_t off) {
        std::stringstream ss;
        while (off < head_) {
            auto c = Read<char>(off++);
            if (c == '\0') break;
            ss << c;
        }
        return ss.str();
    }
#endif

    __aicore__ void Encode(uint8_t val) {
        if (head_ == tail_ + size_) {
            while (Read<uint8_t>(tail_) != END) {
                tail_++;
                tail_ += Read<short>(tail_) + sizeof(short);
                tail_ += Read<short>(tail_) + sizeof(short);
            }
            tail_++;
        }
        volatile __gm__ uint8_t *p = &data_[head_++ % size_];
        *p = val;
    }

    template<typename T>
    __aicore__ void Encode(NodeTy ty, const T *val, short valLen, __gm__ const char *fmt, int fmtLen) {
        Encode(ty);

        auto bytes = reinterpret_cast<uint8_t *>(&valLen);
        Encode(bytes[0]);
        Encode(bytes[1]);
        for (auto i = 0; i < valLen; i++) {
            Encode(val[i]);
        }

        fmtLen += 1; // pad '\0'
        bytes = reinterpret_cast<uint8_t *>(&fmtLen);
        Encode(bytes[0]);
        Encode(bytes[1]);
        for (auto i = 0; i < fmtLen - 1; i++) {
            Encode(fmt[i]);
        }
        Encode('\0');
    }

    INLINE size_t Length(__gm__ const char *str) {
        size_t n = 0;
        while (*str++) {
            n++;
        }
        return n;
    }

    INLINE bool IsDigit(char c) {
        return c >= '0' && c <= '9';
    }

private:
    LogContext ctx;
    int64_t head_;
    int64_t tail_;
    int64_t size_;
    volatile __gm__ Remote *remote_;
    __gm__ uint8_t *data_;
};
