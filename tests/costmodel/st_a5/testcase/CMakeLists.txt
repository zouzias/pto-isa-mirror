function(pto_a5_vfsim_tileop_test NAME)
    get_filename_component(_repo_inc "${CMAKE_SOURCE_DIR}/../../../include" ABSOLUTE)
    add_executable(${NAME} main.cpp)
    target_include_directories(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${_repo_inc})
    target_link_libraries(${NAME} PRIVATE
        GTest::gtest_main
        Threads::Threads)
    pto_enable_costmodel(${NAME} ARCH A5)
    add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

function(pto_a5_vfsim_adapter_test NAME)
    get_filename_component(_repo_inc "${CMAKE_SOURCE_DIR}/../../../include" ABSOLUTE)
    get_filename_component(_repo_pkg_inc "${CMAKE_SOURCE_DIR}/../../../pkg_inc" ABSOLUTE)
    get_filename_component(_vfsim_root "${_repo_pkg_inc}/pto/costmodel/vfsim" ABSOLUTE)
    _pto_a5_build_vfsim()
    add_executable(${NAME} main.cpp)
    target_include_directories(${NAME} PRIVATE
        ${_repo_inc}
        ${_repo_pkg_inc}
        ${_vfsim_root})
    target_compile_definitions(${NAME} PRIVATE PTO_VFSIM_TEST_SOURCE_ROOT="${_vfsim_root}")
    pto_enable_costmodel(${NAME} ARCH A5)
    target_link_libraries(${NAME} PRIVATE
        pto_a5_vfsim
        GTest::gtest_main
        Threads::Threads)
    add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

set(ALL_TESTCASES
    adapter_golden
    full_host_tadd
    host_runtime
    memory_sync
    tadd
    tadds
    tsub
    tsubs
    tmul
    tmuls
    tdiv
    tmax
    tmin
    tmins
    tand
    tshls
    tshrs
)

foreach(TESTCASE ${ALL_TESTCASES})
    if(NOT DEFINED TEST_CASE OR TEST_CASE STREQUAL "${TESTCASE}")
        add_subdirectory(${TESTCASE})
    endif()
endforeach()

if(TARGET tadd)
    get_filename_component(_vfsim_configs "${CMAKE_SOURCE_DIR}/../../../pkg_inc/pto/costmodel/vfsim/configs" ABSOLUTE)
    get_filename_component(_repo_root "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)
    add_test(
        NAME vfsim_relocated_config
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:tadd>
            -DVFSIM_CONFIGS=${_vfsim_configs}
            -DRELOCATED_ROOT=${CMAKE_BINARY_DIR}/relocated_vfsim
            -P ${CMAKE_SOURCE_DIR}/verify_relocated_vfsim.cmake)
    add_test(
        NAME vfsim_external_project
        COMMAND ${CMAKE_COMMAND}
            -DPTO_ISA_ROOT=${_repo_root}
            -DPTO_A5_LLVM_CONFIG=${PTO_A5_LLVM_CONFIG}
            -DPTO_A5_CLANGXX=${PTO_A5_CLANGXX}
            -DCPU_TADD_DIR=${_repo_root}/tests/cpu/st/testcase/tadd
            -DCOSTMODEL_TADD_KERNEL=${CMAKE_CURRENT_SOURCE_DIR}/full_host_tadd/tadd_kernel.cpp
            -DTEST_COMMON_DIR=${_repo_root}/tests/common
            -DFIXTURE_WRITER=${CMAKE_CURRENT_SOURCE_DIR}/full_host_tadd/fixture_writer.cpp
            -DINSTALL_ARCH_DIR=${CMAKE_SYSTEM_PROCESSOR}-linux
            -DEXTERNAL_ROOT=${CMAKE_BINARY_DIR}/external_vfsim_project
            -P ${CMAKE_SOURCE_DIR}/verify_external_vfsim_project.cmake)
endif()
