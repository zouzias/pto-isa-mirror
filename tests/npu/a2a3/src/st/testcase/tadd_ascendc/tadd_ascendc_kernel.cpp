#include "kernel_operator.h"

// AscendC TADD example with full enqueue/dequeue + copyin/out
// Pattern based on WholeReduceSum example (TPipe + TQue)

class KernelTAdd {
public:
    __aicore__ inline KernelTAdd() {}

    __aicore__ inline void Init(__gm__ uint8_t *src0, __gm__ uint8_t *src1, __gm__ uint8_t *dstGm,
                                int rows, int cols, int rowStride)
    {
        // Global tensors
        src0Global.SetGlobalBuffer((__gm__ float *)src0);
        src1Global.SetGlobalBuffer((__gm__ float *)src1);
        dstGlobal.SetGlobalBuffer((__gm__ float *)dstGm);

        // shape info
        totalRows = rows;
        totalCols = cols;
        stride = rowStride;

        // UB buffers (queue)
        // size = rows*rowStride elements
        int dataSize = rows * rowStride;
        pipe.InitBuffer(inQueue0, 1, dataSize * sizeof(float));
        pipe.InitBuffer(inQueue1, 1, dataSize * sizeof(float));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor src0Local = inQueue0.AllocTensor();
        AscendC::LocalTensor src1Local = inQueue1.AllocTensor();

        // Copy GM -> UB
        AscendC::DataCopy(src0Local, src0Global, totalRows * stride);
        AscendC::DataCopy(src1Local, src1Global, totalRows * stride);

        inQueue0.EnQue(src0Local);
        inQueue1.EnQue(src1Local);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor src0Local = inQueue0.DeQue();
        AscendC::LocalTensor src1Local = inQueue1.DeQue();
        AscendC::LocalTensor dstLocal = outQueue.AllocTensor();

        // Elementwise add
        // Set mask to valid columns and repeat across rows
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask(0, totalCols);

        // repeatTimes = rows - 1
        int repeatTimes = totalRows - 1;
        int repStride = stride / 8; // 32B blocks
        AscendC::Add(dstLocal, src0Local, src1Local,
                     repeatTimes, 1, 1, 1, repStride, repStride, repStride);

        outQueue.EnQue(dstLocal);
        inQueue0.FreeTensor(src0Local);
        inQueue1.FreeTensor(src1Local);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor dstLocal = outQueue.DeQue();
        AscendC::DataCopy(dstGlobal, dstLocal, totalRows * stride);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue inQueue0;
    AscendC::TQue inQueue1;
    AscendC::TQue outQueue;
    AscendC::GlobalTensor src0Global;
    AscendC::GlobalTensor src1Global;
    AscendC::GlobalTensor dstGlobal;

    int totalRows = 0;
    int totalCols = 0;
    int stride = 0;
};

extern "C" __global__ __aicore__ void tadd_ascendc_kernel(__gm__ uint8_t *src0,
                                                          __gm__ uint8_t *src1,
                                                          __gm__ uint8_t *dst)
{
    // Example: 64x64, rowStride=64
    KernelTAdd op;
    op.Init(src0, src1, dst, /*rows=*/64, /*cols=*/64, /*rowStride=*/64);
    op.Process();
}
