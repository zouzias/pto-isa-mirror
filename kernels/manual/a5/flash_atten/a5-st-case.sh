export ASCEND_HOME_PATH=/usr/local/Ascend/cann
source /usr/local/Ascend/cann/set_env.sh
bash run.sh -r sim -v Ascend910_9599 --cases "128,128,1024,128,128" --qk-preload 4 --intermediate