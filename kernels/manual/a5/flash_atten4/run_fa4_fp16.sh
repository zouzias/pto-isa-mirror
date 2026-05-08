MODE=${2}
bash run.sh -r sim -v Ascend950PR_9599 -n 0 --cases "128,${1},${1},128,128" --qk-preload 2 --mode ${MODE}
