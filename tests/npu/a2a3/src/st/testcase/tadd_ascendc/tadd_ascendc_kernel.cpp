#include "kernel_operator.h"

class KernelTAdd {
public:
    __aicore__ inline KernelTAdd() {}

    __aicore__ inline void Init(__gm__ uint8_t *src0, __gm__ uint8_t *src1, __gm__ uint8_t *dstGm,
                                int rows, int cols, int rowStride)
    {
        src0Global.SetGlobalBuffer((__gm__ float *)src0);
        src1Global.SetGlobalBuffer((__gm__ float *)src1);
        dstGlobal.SetGlobalBuffer((__gm__ float *)dstGm);

        totalRows = rows;
        totalCols = cols;
        stride = rowStride;

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
        AscendC::LocalTensor<float> src0Local = inQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> src1Local = inQueue1.AllocTensor<float>();

        AscendC::DataCopy(src0Local, src0Global, totalRows * stride);
        AscendC::DataCopy(src1Local, src1Global, totalRows * stride);

        inQueue0.EnQue(src0Local);
        inQueue1.EnQue(src1Local);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> src0Local = inQueue0.DeQue<float>();
        AscendC::LocalTensor<float> src1Local = inQueue1.DeQue<float>();
        AscendC::LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();

        AscendC::Add(dstLocal, src0Local, src1Local, totalRows * stride);

        outQueue.EnQue(dstLocal);
        inQueue0.FreeTensor(src0Local);
        inQueue1.FreeTensor(src1Local);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> dstLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(dstGlobal, dstLocal, totalRows * stride);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue0;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::GlobalTensor<float> src0Global;
    AscendC::GlobalTensor<float> src1Global;
    AscendC::GlobalTensor<float> dstGlobal;

    int totalRows = 0;
    int totalCols = 0;
    int stride = 0;
};

extern "C" __global__ __aicore__ void tadd_ascendc_kernel(__gm__ uint8_t *src0,
                                                          __gm__ uint8_t *src1,
                                                          __gm__ uint8_t *dst)
{
    KernelTAdd op;
    op.Init(src0, src1, dst, /*rows=*/64, /*cols=*/64, /*rowStride=*/64);
    op.Process();
}
