export ASCEND_HOME_PATH=/usr/local/Ascend/cann
source /usr/local/Ascend/cann/set_env.sh
#python3 /home/ssreddy/pto/upload/pto-isa-ttrans/tests/script/run_st.py -r sim -v a5 -t ttrans -g TTRANSTest.case_float_8x8_2x8_2x8
bash run.sh -r sim -v Ascend910_9599 --cases "128,128,1024,128,128" --qk-preload 4

#python3 tests/script/run_st.py -r sim -v a5 -t ttrans -g TTRANSTest.case_uint8_32x32_32x32_32x32
#python3 tests/script/run_st.py -r sim -v a5 -t ttrans -g TTRANSTest.case_uint8_64x64_64x64_22x63


#python3 tests/script/run_st.py -r sim -v a5 -t ttrans -g TTRANSTest.case_float_8x8_1x8_1x8
#python3 tests/script/run_st.py -r sim -v a5 -t ttrans -g TTRANSTest.case_float_8x8_2x8_2x8
#python3 tests/script/run_st.py -r sim -v a5 -t ttrans -g TTRANSTest.case_float_16x8_1x16_1x16
#python3 tests/script/run_st.py -r sim -v a5 -t ttrans -g TTRANSTest.case_float_8x8_4x8_4x8
#python3 tests/script/run_st.py -r sim -v a5 -t ttrans -g TTRANSTest.case_float_8x8_8x8_8x8