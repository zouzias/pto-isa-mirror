OK I also learned a lot about FA code that you can use too.

first of all, the vector side and cube side compile separately. I think it even outputs two differnet binaries that get combined after.
so memory allocation and auto sync stuff see those sides separate.

For FA, 

we have some big computations:

1. compute_qk happens in cube,
2. compute_p (softmax) happens in vector
3. compute_pv happens in cube,
4. compute_gu happens in vector

each one is dependant on the previos.

to tell a bit more about architecture,

we have multiple ai cores,
each ai core has a cube and vector unit
vector unit also has 2 subblocks that can do differnt computation. the memory transfer in cube, and vector has multiple lanes. like you can start multiple transfers happen simultanously from l1 to cube or vector without waiting for previous one to finish.
there are some cube/vector communication (cv communication).

when doing computation, we can have smaller tiles and compute sequentially over them. there's also this possiblity to do multi-buffering for each computation.

sometimes when there are two different computation on the same resource, we might be able to actually make a pipeline for them. for example there are these compute_qk and compute_pv that happen in cube. so maybe we can load compute_qk, and then compute it and whlie doing that load the compute_pv in cube.

and by doing that we have some problems, like for example the first few iterations of compute_qk need to be done before doing this pipeline thing of doing compute_qk with the compute_pv. so there's a preload stage that only does compute_qk. same thing can happen in vector side too.

Also another thing for copule last or first iterations of each compute, we might need to do syncronization differently (becaues of multi-buffering) thats why we have prelouge, and epilough phases that compiler detect them and compile them differently

also regarding cv communication, right now we are using some pragmas, and those are a hacky solution but that's the way for now. might change later but will tell you.
pto auto mode is dumb about having multiple ai cores so those ones use ffts instructions and pto automode cant replace them too. for the two vector subblokc thing, that is also done manually i think and pto auto mode doesnt have control on that too.

i have to provide you with the way ew do pipelineing and multi buffering with code example so you understand.


ok now actually lets see how some of these tecniques, specially multi buffering and pipelining and multiple phasing can be done.

1. using cube and vector differently:

we usually have these guards in the code:

// Inline macro used for small, performance-sensitive functions
#ifndef PTO_INLINE
#define PTO_INLINE __attribute__((always_inline)) inline
#endif

// Detect build-time macros and expose as constexpr flags for clearer conditionals
#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

cube code should be written separately than vector code.
    if constexpr (DAV_CUBE) or   if constexpr (DAV_VEC) 
because the compilation would be different

2.multi-buffering

the code is similar to this 
```cpp
using InnerMultiBuffered = MultiBuffered<kMatTNBuffers>::NestedLoopInvoker<Range<kTileFactor>>;
        mb.loop<Range<qkPreloadNum, kTileFactor>>([&](auto ctxOuter, InnerMultiBuffered inner) {
            int tile_id = ctxOuter.iter;
            inner.loop([&](auto ctxInner) {
                int sub_tile = ctxInner.iter;
                TileMatKData kMatTile;
                qkPipe.prod.setTileId(tile_id, sub_tile);
                compute_qk<QKPipe, S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, INTERMEDIATE_CHECK, CAUSAL_MASK>(
                    qkPipe, tile_id, sub_tile, q_block, k, qMatTile, kMatTile, qkAccTile, block_idx);
            });
        });
```
where kMatTNBuffers is the number of buffers. this will normally unroll inner loop to do multi buffering but  only if the inner iteration is 1, it will unroll outer loop. 

3. pipelining
sometimes two types of computation needs to happen in the same resource (cube). 
for example doing compute_qk and cmopute_pv can be done after preload of several tiles of compute_qk.

here we do something like this :

```cpp
constexpr int NumStages = 2;
        MultiStaged<NumStages> qk_pv_stages;
        for (int tile_id = 0; tile_id < num_tiles_s1 - qkPreloadNum; ++tile_id) {
            for (int sub_tile = 0; sub_tile < kTileFactor; ++sub_tile) {
                qk_pv_stages.run(
                    [&]() {
                        qkPipe.prod.setTileId(tile_id + qkPreloadNum, sub_tile);
                        compute_qk<QKPipe, S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, INTERMEDIATE_CHECK,
                                   CAUSAL_MASK>(qkPipe, tile_id + qkPreloadNum, sub_tile, q_block, k, qMatTile,
                                                kMatTile, qkAccTile, block_idx);
                    },
                    [&]() {
                        pPipe.cons.setTileId(tile_id, sub_tile);
                        pvPipe.prod.setTileId(tile_id, sub_tile);
                        compute_pv<PPipe, PVPipe, S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, INTERMEDIATE_CHECK,
                                   CAUSAL_MASK>(pPipe, pvPipe, tile_id, sub_tile, v, pMatTile, vMatTile, pvAccTile, block_idx);
                    });
            }
        }
```
here normally multi-buffering is not done because, we are already maximizing the resources so it's just not worth it doing.

Also remember when doing two different computation with same resource, if they are dependant to each other you should do some precompute for the first tiles of the dependee and some post compute for the latest tiles of dependant


4.multiple phases:

we have some phases like Epilogue, prolouge and main. these besaically are fo the compiler pass to treat them differnetly . ususallly the first couple iterations or last couple iterations need differnet sync flags, so thats about it 
```cpp
if constexpr (GU_Phase == Phase::Epilogue) {
            using GlobalOutT = GlobalTensor<float, pto::Shape<1, 1, 1, CUBE_S0 / VEC_CORES, HEAD_SIZE>,
                                            pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
            GlobalOutT outGlobal((__gm__ float *)(o_out + subblock_base_rows * HEAD_SIZE));
            TSTORE(outGlobal, runningOTile);
        
        }
```

6. pragmas:

you will see in the code something like this 
```cpp
#pragma pto v_loop_barrier 
```
this is to hint the compiler about syncronization between cube and vector i believe. right now this is the hack to make it work but in future this will change.

7. multipe ai cores.

so far everything was happening in the same ai core. but we have multiple ai cores in each soc. so in this situation we have to first partition our computation but more important than that, we have to sync between them. this sync is not currently supported in auto mode compiler so we have to have the `ffts_cross_core_sync` flags. 
