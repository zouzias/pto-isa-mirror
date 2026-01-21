export ASCEND_HOME_PATH=/usr/local/Ascend/cann
source /usr/local/Ascend/cann/set_env.sh
python3 tests/script/run_st.py -r sim -v a5 -t tinsert_custom -g TInsertCustomTest.case1
python3 tests/script/run_st.py -r sim -v a5 -t tinsert_custom -g TInsertCustomTest.case2
python3 tests/script/run_st.py -r sim -v a5 -t tinsert_custom -g TInsertCustomTest.case3
python3 tests/script/run_st.py -r sim -v a5 -t tinsert_custom -g TInsertCustomTest.case4
python3 tests/script/run_st.py -r sim -v a5 -t tinsert_custom -g TInsertCustomTest.case5
python3 tests/script/run_st.py -r sim -v a5 -t tinsert_custom -g TInsertCustomTest.case6
python3 tests/script/run_st.py -r sim -v a5 -t tinsert_custom -g TInsertCustomTest.case7
