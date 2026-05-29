bash run.sh -r sim -v Ascend950PR_9599 -n 0 --cases "128,${1},${1},128,128,128" --qk-preload 2 --mode 1 
#bash run.sh -r sim -v Ascend950PR_9599 -n 0 --cases "64,${1},${1},128,256,256" --qk-preload 2 --mode 1

#python3 scripts/compute_buffer_usage_fa3_fp16_dn.py --cube_s0 128 --cube_s1 128 --head_size 128 --tile_s1 128 
#python3 scripts/compute_buffer_usage_fa3_fp16_dn.py --cube_s0 256 --cube_s1 128 --head_size 64 --tile_s1 128  --union
