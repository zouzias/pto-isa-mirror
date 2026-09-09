get_filename_component(_repo_root "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)
set(_cpu_sim_test_dir "${_repo_root}/tests/cpu/st/testcase/tadd")

add_executable(full_host_tadd
    "${_cpu_sim_test_dir}/main.cpp"
    "${_cpu_sim_test_dir}/tadd_kernel.cpp")
target_include_directories(full_host_tadd PRIVATE
    "${_repo_root}/include"
    "${_repo_root}/tests/common")
target_link_libraries(full_host_tadd PRIVATE
    GTest::gtest_main
    Threads::Threads)
pto_enable_costmodel(full_host_tadd ARCH A5)

add_executable(full_host_tadd_fixture fixture_writer.cpp)

set(_run_dir "${CMAKE_CURRENT_BINARY_DIR}/run")
set(_case_dir "${CMAKE_CURRENT_BINARY_DIR}/TADDTest.case_float_64x64_64x64_64x64")
file(MAKE_DIRECTORY "${_run_dir}")

add_test(NAME full_host_tadd_fixture COMMAND full_host_tadd_fixture "${_case_dir}")
set_tests_properties(full_host_tadd_fixture PROPERTIES FIXTURES_SETUP full_host_tadd_data)

add_test(
    NAME full_host_tadd
    COMMAND full_host_tadd --gtest_filter=TADDTest.case_float_64x64_64x64_64x64)
set_tests_properties(full_host_tadd PROPERTIES
    FIXTURES_REQUIRED full_host_tadd_data
    WORKING_DIRECTORY "${_run_dir}")
