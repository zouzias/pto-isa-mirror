# A5 TADD 单 ISA 指令测试:验证 PTO 入口 + VF 结算。
# repo include 用 st_a5 源码根往上 3 级(st_a5 -> costmodel -> tests -> repo)。
get_filename_component(_repo_inc "${CMAKE_SOURCE_DIR}/../../../include" ABSOLUTE)
add_executable(tadd main.cpp)
target_include_directories(tadd PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${_repo_inc})
target_enable_a5_vf_mock(tadd)
add_test(NAME tadd COMMAND tadd)
