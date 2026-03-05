DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

g++ -std=c++17 -g \
    -I $ASCEND_HOME_PATH/include \
    -I $ASCEND_HOME_PATH/aarch64-linux/include/experiment/runtime \
    -I $ASCEND_HOME_PATH/aarch64-linux/include/experiment/msprof \
    -L $ASCEND_HOME_PATH/lib64 \
    -l runtime -l ascendcl main.cpp -o main
    
ccec -c -x cce fa_auto.cpp -std=c++17 -O2 \
    --cce-aicore-only --cce-aicore-arch=dav-c220 \
    -I $DIR/../../../../include \
    -I $DIR/../../../../include/pto \
    --cce-enable-pto-passes -fenable-matrix \
    -DMEMORY_BASE -D__davinci -D__DAV_V220 \
    -o attention.o

python3 gen_data.py
./main attention.o attention