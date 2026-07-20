# A5 多 VF 测试:验证一条 PTO 含多 VF 时 vf_infos 攒 vector + 后端统一算。
get_filename_component(_repo_inc "${CMAKE_SOURCE_DIR}/../../../include" ABSOLUTE)
add_executable(multi_vf main.cpp)
target_include_directories(multi_vf PRIVATE ${_repo_inc})
target_enable_a5_vf_mock(multi_vf)
add_test(NAME multi_vf COMMAND multi_vf)
