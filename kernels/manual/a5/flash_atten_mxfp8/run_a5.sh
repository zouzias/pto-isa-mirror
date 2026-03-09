MODE=${1}
bash run.sh -r npu -v Ascend910_9599 -n 0 --cases "128,16384,16384,128,128" --qk-preload 2 --mode_dn --mode ${MODE} 
