1. please read the skills in "/home/jlchen/workspace/PTO/npu_skills/pto-isa/", and then replace line 804 ~ 816 to "copy_ubuf_to_gm_align_v2" which need store the data from nzConBuff (NZ+1) to p_fifo as "output_p_nz_plus_one.bin"
2. And then change the main.cpp to compare the p_fifo data with golden data "p_t_nz_plus_one.bin".
3. run testcase to check if the p_fifo can compare successfully, "bash run.sh -r sim -v Ascend950PR_9599 -n 0 --cases "128,128,128,128,128,128" --qk-preload 2 --mode 1 -i"
4. the intermedicate check in step 3 is pass, however, it is failed for testcase "bash run.sh -r sim -v Ascend950PR_9599 -n 0 --cases "128,512,512,128,128,128" --qk-preload 2 --mode 1 -i" when open "#define SOFTMAX_S064_4VSSTB", please help to update and fix the bugs.

