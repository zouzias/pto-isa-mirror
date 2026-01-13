/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_PTO_PIPE_HPP
#define PTO_COMMON_PTO_PIPE_HPP

#include <cstdint>
#include <type_traits>

#include <pto/common/event.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

namespace pto {

// -----------------------------------------------------------------------------
// Low-level sync wrappers
// -----------------------------------------------------------------------------

// Pipeline-pair event that maps 1:1 to {set_flag, wait_flag} usage.
// Useful when the dependency is about concrete pipelines rather than Op classes.
template <pipe_t SrcPipe, pipe_t DstPipe, bool AutoToken = true, event_t EventID = EVENT_ID0>
struct PipeEvent {
  static constexpr pipe_t srcPipe = SrcPipe;
  static constexpr pipe_t dstPipe = DstPipe;

#if defined(__CPU_SIM) || !defined(__CCE_AICORE__)
  PTO_INTERNAL PipeEvent& Wait() { return *this; }
  PTO_INTERNAL PipeEvent& Record() { return *this; }
  PTO_INTERNAL PipeEvent& operator=(RecordEvent) { return Record(); }
#else
#ifdef PTO_FLAG_TEST
  CceEventIdType token = {};
#else
  const event_t token = AutoToken ? EventIdCounter<srcPipe, dstPipe>::GetNextId() : EventID;
#endif

  PTO_INTERNAL PipeEvent& Wait() {
#ifdef PTO_FLAG_TEST
    __pto_wait_flag((pipe_t)srcPipe, (pipe_t)dstPipe, token);
#else
    wait_flag((pipe_t)srcPipe, (pipe_t)dstPipe, token);
#endif
    return *this;
  }

  PTO_INTERNAL PipeEvent& Record() {
#ifdef PTO_FLAG_TEST
    token = __pto_set_flag((pipe_t)srcPipe, (pipe_t)dstPipe);
#else
    set_flag((pipe_t)srcPipe, (pipe_t)dstPipe, token);
#endif
    return *this;
  }

  PTO_INTERNAL PipeEvent& operator=(RecordEvent) { return Record(); }
#endif
};

// Runtime-token variant of PipeEvent (for code that uses dynamic event ids).
template <pipe_t SrcPipe, pipe_t DstPipe>
struct DynPipeEvent {
  static constexpr pipe_t srcPipe = SrcPipe;
  static constexpr pipe_t dstPipe = DstPipe;
  event_t token;

  PTO_INTERNAL explicit DynPipeEvent(event_t token_) : token(token_) {}

#if defined(__CPU_SIM) || !defined(__CCE_AICORE__)
  PTO_INTERNAL DynPipeEvent& Wait() { return *this; }
  PTO_INTERNAL DynPipeEvent& Record() { return *this; }
  PTO_INTERNAL DynPipeEvent& operator=(RecordEvent) { return Record(); }
#else
  PTO_INTERNAL DynPipeEvent& Wait() {
#ifdef PTO_FLAG_TEST
    __pto_wait_flag((pipe_t)srcPipe, (pipe_t)dstPipe, token);
#else
    wait_flag((pipe_t)srcPipe, (pipe_t)dstPipe, token);
#endif
    return *this;
  }

  PTO_INTERNAL DynPipeEvent& Record() {
#ifdef PTO_FLAG_TEST
    token = __pto_set_flag((pipe_t)srcPipe, (pipe_t)dstPipe);
#else
    set_flag((pipe_t)srcPipe, (pipe_t)dstPipe, token);
#endif
    return *this;
  }

  PTO_INTERNAL DynPipeEvent& operator=(RecordEvent) { return Record(); }
#endif
};

// Cross-core CV flag event wrapper around {ffts_cross_core_sync, wait_flag_dev}.
// This is intentionally independent of Op/pipe mapping.
enum class CvSyncMode : uint16_t {
  CV_CORE_SYNC = 2,
};

PTO_INTERNAL constexpr uint16_t GetCvFftsMsg(CvSyncMode mode, uint16_t flag_id, uint16_t base_const = 0x1) {
  return static_cast<uint16_t>((base_const & 0xf) + ((static_cast<uint16_t>(mode) & 0x3) << 4) +
                               ((flag_id & 0xf) << 8));
}

template <pipe_t SignalPipe, uint16_t FlagId, CvSyncMode Mode = CvSyncMode::CV_CORE_SYNC>
struct CvFlagEvent {
#if defined(__CPU_SIM) || !defined(__CCE_AICORE__)
  PTO_INTERNAL CvFlagEvent& Wait() { return *this; }
  PTO_INTERNAL CvFlagEvent& Record() { return *this; }
  PTO_INTERNAL CvFlagEvent& operator=(RecordEvent) { return Record(); }
#else
  PTO_INTERNAL CvFlagEvent& Wait() {
    wait_flag_dev(static_cast<uint8_t>(FlagId));
    return *this;
  }
  PTO_INTERNAL CvFlagEvent& Record() {
    ffts_cross_core_sync(SignalPipe, GetCvFftsMsg(Mode, FlagId));
    return *this;
  }
  PTO_INTERNAL CvFlagEvent& operator=(RecordEvent) { return Record(); }
#endif
};

// Runtime-id variant for CV cross-core flags.
template <pipe_t SignalPipe, CvSyncMode Mode = CvSyncMode::CV_CORE_SYNC>
struct DynCvFlagEvent {
  uint16_t flag_id;
  PTO_INTERNAL explicit DynCvFlagEvent(uint16_t flag_id_) : flag_id(flag_id_) {}

#if defined(__CPU_SIM) || !defined(__CCE_AICORE__)
  PTO_INTERNAL DynCvFlagEvent& Wait() { return *this; }
  PTO_INTERNAL DynCvFlagEvent& Record() { return *this; }
  PTO_INTERNAL DynCvFlagEvent& operator=(RecordEvent) { return Record(); }
#else
  PTO_INTERNAL DynCvFlagEvent& Wait() {
    wait_flag_dev(static_cast<uint8_t>(flag_id));
    return *this;
  }
  PTO_INTERNAL DynCvFlagEvent& Record() {
    ffts_cross_core_sync(SignalPipe, GetCvFftsMsg(Mode, flag_id));
    return *this;
  }
  PTO_INTERNAL DynCvFlagEvent& operator=(RecordEvent) { return Record(); }
#endif
};

// -----------------------------------------------------------------------------
// Cross-core GM FIFO pipe
// -----------------------------------------------------------------------------

enum class CoreType {
  CUBE,
  VECTOR,
};

template <CoreType>
struct PipeStageTraits;

// Default mapping: chosen to match typical GM FIFO traffic for Cube/Vec roles.
template <>
struct PipeStageTraits<CoreType::CUBE> {
#if defined(__CCE_AICORE__)
  static constexpr pipe_t ProdSignalPipe = PIPE_FIX;
  static constexpr pipe_t ConsSignalPipe = PIPE_MTE2;
#else
  static constexpr pipe_t ProdSignalPipe = (pipe_t)0;
  static constexpr pipe_t ConsSignalPipe = (pipe_t)0;
#endif
};

template <>
struct PipeStageTraits<CoreType::VECTOR> {
#if defined(__CCE_AICORE__)
  static constexpr pipe_t ProdSignalPipe = PIPE_MTE3;
  static constexpr pipe_t ConsSignalPipe = PIPE_MTE2;
#else
  static constexpr pipe_t ProdSignalPipe = (pipe_t)0;
  static constexpr pipe_t ConsSignalPipe = (pipe_t)0;
#endif
};

// Pipe: a ring-buffer synchronization helper for cross-core GM FIFOs.
//
// ReadyFlag:    producer -> consumer
// ConsumedFlag: consumer -> producer
// Depth:        ring buffer depth (>= 1)
// Period:       reduce flag traffic; sync once every Period items (1 = every item)
template <unsigned ReadyFlag, unsigned ConsumedFlag, int Depth, int Period, CoreType ProdRole, CoreType ConsRole>
struct Pipe {
  static_assert(Depth >= 1, "Pipe Depth must be >= 1");
  static_assert(Period >= 1, "Pipe Period must be >= 1");
  static_assert(Period <= Depth, "Pipe Period must be <= Depth");
  static_assert(ReadyFlag < 16, "ReadyFlag must fit in 4 bits (FFTS CV flag id).");
  static_assert(ConsumedFlag < 16, "ConsumedFlag must fit in 4 bits (FFTS CV flag id).");

  static constexpr int PipeDepth = Depth;
  static constexpr int SyncPeriod = Period;

  static constexpr pipe_t ProdPipe = PipeStageTraits<ProdRole>::ProdSignalPipe;
  static constexpr pipe_t ConsPipe = PipeStageTraits<ConsRole>::ConsSignalPipe;

  struct Producer {
    static constexpr int PipeDepth = Depth;
    int iter = 0;

    PTO_INTERNAL void alloc() {
#if defined(__CCE_AICORE__)
      if (iter >= Depth && (iter % Period) == 0) {
        wait_flag_dev(static_cast<uint8_t>(ConsumedFlag));
      }
#endif
    }

    PTO_INTERNAL void record() {
#if defined(__CCE_AICORE__)
      ffts_cross_core_sync(ProdPipe, GetCvFftsMsg(CvSyncMode::CV_CORE_SYNC, static_cast<uint16_t>(ReadyFlag)));
#endif
      iter++;
    }
  };

  struct Consumer {
    static constexpr int PipeDepth = Depth;
    int iter = 0;

    PTO_INTERNAL void wait() {
#if defined(__CCE_AICORE__)
      wait_flag_dev(static_cast<uint8_t>(ReadyFlag));
#endif
    }

    PTO_INTERNAL void free() {
#if defined(__CCE_AICORE__)
      const bool is_sync_step = ((iter + 1) % Period) == 0;
      if (iter >= Depth - Period && is_sync_step) {
        ffts_cross_core_sync(ConsPipe, GetCvFftsMsg(CvSyncMode::CV_CORE_SYNC, static_cast<uint16_t>(ConsumedFlag)));
      }
#endif
      iter++;
    }
  };
};

// -----------------------------------------------------------------------------
// TPUSH/TPOP: GM FIFO tile transfer helpers
// -----------------------------------------------------------------------------

namespace detail {
template <typename T, typename = void>
struct has_wait : std::false_type {};
template <typename T>
struct has_wait<T, std::void_t<decltype(std::declval<T&>().wait())>> : std::true_type {};

template <typename T, typename = void>
struct has_free : std::false_type {};
template <typename T>
struct has_free<T, std::void_t<decltype(std::declval<T&>().free())>> : std::true_type {};

template <typename T, typename = void>
struct has_alloc : std::false_type {};
template <typename T>
struct has_alloc<T, std::void_t<decltype(std::declval<T&>().alloc())>> : std::true_type {};

template <typename T, typename = void>
struct has_record : std::false_type {};
template <typename T>
struct has_record<T, std::void_t<decltype(std::declval<T&>().record())>> : std::true_type {};
} // namespace detail

template <typename T>
constexpr bool is_pipe_consumer_v = detail::has_wait<T>::value && detail::has_free<T>::value;

template <typename T>
constexpr bool is_pipe_producer_v = detail::has_alloc<T>::value && detail::has_record<T>::value;

template <typename PipeCons, typename TileData, typename DType>
PTO_INTERNAL RecordEvent TPOP_IMPL(PipeCons &cons, TileData &tile, __gm__ DType *fifo_base) {
  static_assert(is_pipe_consumer_v<PipeCons>, "TPOP: PipeCons must provide wait() and free().");

  cons.wait();

  const int buffer_idx = cons.iter % PipeCons::PipeDepth;
  __gm__ DType *addr = fifo_base + buffer_idx * TileData::Numel;

  using GShape = TileShape2D<DType, TileData::Rows, TileData::Cols, Layout::ND>;
  using GStride = BaseShape2D<DType, TileData::Rows, TileData::Cols, Layout::ND>;
  using GT = GlobalTensor<DType, GShape, GStride, Layout::ND>;
  GT global_tensor(addr);

  RecordEvent ev = TLOAD(tile, global_tensor);
  cons.free();
  return ev;
}

template <typename PipeProd, typename TileData, typename DType>
PTO_INTERNAL RecordEvent TPUSH_IMPL(PipeProd &prod, TileData &tile, __gm__ DType *fifo_base) {
  static_assert(is_pipe_producer_v<PipeProd>, "TPUSH: PipeProd must provide alloc() and record().");

  prod.alloc();

  const int buffer_idx = prod.iter % PipeProd::PipeDepth;
  __gm__ DType *addr = fifo_base + buffer_idx * TileData::Numel;

  using GShape = TileShape2D<DType, TileData::Rows, TileData::Cols, Layout::ND>;
  using GStride = BaseShape2D<DType, TileData::Rows, TileData::Cols, Layout::ND>;
  using GT = GlobalTensor<DType, GShape, GStride, Layout::ND>;
  GT global_tensor(addr);

  RecordEvent ev = TSTORE(global_tensor, tile);
  prod.record();
  return ev;
}

template <typename PipeCons, typename TileData, typename DType, typename... WaitEvents>
PTO_INST RecordEvent TPOP(PipeCons &cons, TileData &dst, __gm__ DType *fifo_base, WaitEvents &...events) {
  TSYNC(events...);
  return TPOP_IMPL(cons, dst, fifo_base);
}

template <typename PipeProd, typename TileData, typename DType, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(PipeProd &prod, TileData &src, __gm__ DType *fifo_base, WaitEvents &...events) {
  TSYNC(events...);
  return TPUSH_IMPL(prod, src, fifo_base);
}

} // namespace pto

#endif // PTO_COMMON_PTO_PIPE_HPP
