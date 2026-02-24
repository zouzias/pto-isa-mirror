# AscendC Add vs PTO tadd — why PTO is shorter

This note explains **why PTO tadd is shorter** and what it encapsulates compared to a minimal AscendC pipeline implementation.

## What PTO encapsulates

### 1) Pipeline + sync management
- **PTO**: `TLOAD/TADD/TSTORE` + `Event<Op::...>` give correct ordering between **MTE2 → VEC → MTE3**.
- **AscendC**: you must explicitly build the pipeline with **TPipe + TQue** and manage **EnQue/DeQue/Free** to keep MTE2/VEC/MTE3 synchronized.

### 2) UB buffer allocation + lifetime
- **PTO**: `Tile` + `TASSIGN` implicitly allocates UB regions and ties them to tiles.
- **AscendC**: you must `InitBuffer()` and allocate local tensors from queues (`AllocTensor`), then free them after use.

### 3) Shape/stride + memory layout
- **PTO**: `Tile` + `GlobalTensor` infer layout/stride from template params; no manual stride.
- **AscendC**: you must pass **rows/cols/stride** and ensure `DataCopy` length matches the layout.

## Minimal AscendC pipeline version (what PTO hides)

```cpp
class KernelTAdd {
public:
    __aicore__ inline void Init(__gm__ float *src0, __gm__ float *src1, __gm__ float *dst,
                                int rows, int cols, int rowStride)
    {
        src0Global.SetGlobalBuffer(src0);
        src1Global.SetGlobalBuffer(src1);
        dstGlobal.SetGlobalBuffer(dst);
        totalRows = rows; totalCols = cols; stride = rowStride;
        int dataSize = rows * rowStride;
        pipe.InitBuffer(inQueue0, 1, dataSize * sizeof(float));
        pipe.InitBuffer(inQueue1, 1, dataSize * sizeof(float));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(float));
    }

    __aicore__ inline void Process() { CopyIn(); Compute(); CopyOut(); }

private:
    __aicore__ inline void CopyIn() {
        auto s0 = inQueue0.AllocTensor<float>();
        auto s1 = inQueue1.AllocTensor<float>();
        AscendC::DataCopy(s0, src0Global, totalRows * stride);
        AscendC::DataCopy(s1, src1Global, totalRows * stride);
        inQueue0.EnQue(s0); inQueue1.EnQue(s1);
    }
    __aicore__ inline void Compute() {
        auto s0 = inQueue0.DeQue<float>();
        auto s1 = inQueue1.DeQue<float>();
        auto d  = outQueue.AllocTensor<float>();
        AscendC::Add(d, s0, s1, totalRows * stride);
        outQueue.EnQue(d);
        inQueue0.FreeTensor(s0); inQueue1.FreeTensor(s1);
    }
    __aicore__ inline void CopyOut() {
        auto d = outQueue.DeQue<float>();
        AscendC::DataCopy(dstGlobal, d, totalRows * stride);
        outQueue.FreeTensor(d);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue0, inQueue1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::GlobalTensor<float> src0Global, src1Global, dstGlobal;
    int totalRows = 0, totalCols = 0, stride = 0;
};
```

**Summary:** PTO is shorter because it encapsulates **(a) pipeline sync**, **(b) UB buffer management**, and **(c) shape/stride bookkeeping** into `Tile + GlobalTensor + Events + TLOAD/TADD/TSTORE`.
