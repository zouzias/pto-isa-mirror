codex resume 019d7641-d6d9-7482-9ec1-8038e35faffc
fill in the RecordCceCall

- 去掉raw_cee的考量，没有raw cce call
- 统一pto_instr的入口为PTO_TRACE_CALL，只保留一个；后期可以在PTO_TRACE_CALL的宏定义中对不同运行模式进行宏展开

process instruction one by one:
1. pick instruction `t*`
2. read /home/lc/pto-isa-costmodel/tests/npu/a2a3/src/st/testcase/t*
3. write st for costmodel according to step 2
4. expect accuracy set to 0, only make sure it can compile and no runtime error
**if encounter uncertainty, ask me, DO NOT INFER BY YOURSELF**
if st already exists, proceed to next

TARGETs:
TSUB, TMUL, TEXP, TSQRT, TROWSUM, TROWMAX, TROWEXPAND, TCOLSUM, TCOLMAX, TADDS, TDIVS, TMINS, TMULS, TCOPY, TSEL, TCVT, TEXTRACT, TTRANS, TSQRT32, TMRGSORT, TLOAD, TMATMUL, TMATMUL_ACC, TMOV

### DSV3
TSUB, TMUL, TEXP, TSQRT, TROWSUM, TROWMAX, TROWEXPAND, TCOLSUM, TCOLMAX, TADDS, TDIVS, TMINS, TMULS, TCOPY, TSEL, TCVT, TEXTRACT, TTRANS, TSQRT32, TMRGSORT, TLOAD, TMATMUL, TMATMUL_ACC, TMOV

### Qwen3
TDIV, TMAX, TMAXS, TMIN, TCMP, TNEG, TRECP, TRSQRT, TSUBS, TABS, TCOLMIN, TROWMIN, TGATHER, TSELS, TMAXRANGE


### Simplify logic:
Discard the costmodel catogorization of cce instructions, write costmodel formula for every cce instruction.
`cce_costmodel.hpp`:
contains all cce instructions, return costmodel latency prediction
input parameters are identical to original cce inputs
```cpp
namespace CceCostmodel{
    uint64_t vadd(...) {
        // head_cycles + computing_cycles * repeat + interval_cycles
        cycles = 13 + 2 * repeat + 18;
        return cycles
    }

    uint64_t vdiv(...) {
        // head_cycles + computing_cycles * repeat + interval_cycles
        cycles = 13 + 8 * repeat + 25;
        return cycles
    }

    uint64_t copy_gm_to_cbuf(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto gmGap, auto l1Gap, auto pad) {
        datasize = nBurst * lenBurst * BlockSize;
        cycles = int(datasize / (arch_config.bandwidth['gm_to_l1'] * 1024**3) * arch_config.frequency)
        return cycles
    }

    ...
}
```

- remove names of parameters in cce record, only record sequences of parameters, and it will be input to cce costmodel.
- shorten dispatch logic with higher level dispatcher, now codes are too clumsy.

remove helper functions
EstimateCopyGmToUbufAlignCycles
EstimateCopyUbufToGmAlignCycles
EstimateImg2colv2Cycles
EstimateLoadCbufCycles
EstimateVconvCycles

combine cce_stub.hpp and cce_costmodel.hpp to simplify code structure
- interface are still like cce_costmodel, expand every instruction function, make it flat 
- now every cce instruction function do two things:
    1. record this cce instruction name and its parameter sequence in a simple way 
    2. calculate latency according to costmodel
- the wrapper of pto instruction in pto_instr.hpp sum up all cce latency into ths dst tile.
DO NOT OVER ENGINEER OR OVER WRAP, use a flat and simple way, original structure can be discarded if a much simpler structure is available

need to clean cce_costmodel.hpp
- namespace too long, remove it
- EvaluateLinear function define default value, remove those with two inputs which are guessed. 

codex resume 019d8abd-a8c0-7042-9f1f-12916dd23fb1

1. cycle into tile
2. cce head/tail controled by if first/last. last PIPE_V, tail

Maintain a queue of various pipes (Vector, Cube, GM_TO_L1, L1_TO_L0A, etc.). Each CCE
instruction is executed on one of these pipes, and the CCE queues executed on these pipes
are recorded.
For head: if <pipe> is empty, add head into cce cycles.
For tail: trace a global variable LastCceTail, which is the tail cycle of last cce
instruction, and is assigned in EstimateLinearCycles: <pipe>::LastCceTail=tail. When
encountering pipe_barrier(<pipe>) or wait_flag(srcPipe=<pipe>), add <pipe>::LastCceTail into total pto cycles.

Try to resuse pipe defined in arch config, too much pipe names in different files.
RecordCceCall use same, put pipe as input, <pipe>.append(<cce>) in RecordCceCall

add InjectTileCycles helper function in pto_instr.hpp, 

› /home/lc/pto-isa-costmodel/include/pto/common, /home/lc/pto-isa-costmodel/include/pto/
  costmodel is a new costmodel realization based on an old pto-isa version. I want to
  imply this on latest version /home/lc/pto-isa. /home/lc/pto-isa/include/pto/common
  should be revised and /home/lc/pto-isa/include/pto/costmodel should be replaced.