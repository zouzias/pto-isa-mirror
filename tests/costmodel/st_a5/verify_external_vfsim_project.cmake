# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(NOT PTO_ISA_ROOT OR NOT EXTERNAL_ROOT OR NOT CPU_TADD_DIR OR NOT TEST_COMMON_DIR OR NOT FIXTURE_WRITER OR
   NOT INSTALL_ARCH_DIR)
    message(FATAL_ERROR "external VfSim project test is missing an input path")
endif()

set(_source_dir "${EXTERNAL_ROOT}/src")
set(_build_dir "${EXTERNAL_ROOT}/build")
set(_install_build_dir "${EXTERNAL_ROOT}/pto_install_build")
set(_install_prefix "${EXTERNAL_ROOT}/install_prefix")
set(_relocated_prefix "${EXTERNAL_ROOT}/relocated_prefix")
set(_install_root "${_relocated_prefix}/${INSTALL_ARCH_DIR}")
set(_case_dir "${EXTERNAL_ROOT}/TADDTest.case_float_64x64_64x64_64x64")
file(REMOVE_RECURSE "${EXTERNAL_ROOT}")
file(MAKE_DIRECTORY "${_source_dir}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${PTO_ISA_ROOT}" -B "${_install_build_dir}"
            -DCMAKE_BUILD_TYPE=Debug
            -DCMAKE_INSTALL_PREFIX=${_install_prefix}
    RESULT_VARIABLE _pto_configure_result
    OUTPUT_VARIABLE _pto_configure_output
    ERROR_VARIABLE _pto_configure_error)
if(NOT _pto_configure_result EQUAL 0)
    message(FATAL_ERROR "PTO-ISA install configure failed:\n${_pto_configure_output}\n${_pto_configure_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${_install_build_dir}" --target version_pto-isa_info
    RESULT_VARIABLE _version_result
    OUTPUT_VARIABLE _version_output
    ERROR_VARIABLE _version_error)
if(NOT _version_result EQUAL 0)
    message(FATAL_ERROR "PTO-ISA version info generation failed:\n${_version_output}\n${_version_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${_install_build_dir}" --prefix "${_install_prefix}" --component pto-isa
    RESULT_VARIABLE _install_result
    OUTPUT_VARIABLE _install_output
    ERROR_VARIABLE _install_error)
if(NOT _install_result EQUAL 0)
    message(FATAL_ERROR "PTO-ISA install failed:\n${_install_output}\n${_install_error}")
endif()

if(NOT EXISTS "${_install_prefix}/${INSTALL_ARCH_DIR}/include" OR
   NOT EXISTS "${_install_prefix}/${INSTALL_ARCH_DIR}/pkg_inc")
    message(FATAL_ERROR "PTO-ISA install did not create the expected ${INSTALL_ARCH_DIR}/include and pkg_inc layout")
endif()

file(MAKE_DIRECTORY "${_relocated_prefix}")
file(RENAME "${_install_prefix}/${INSTALL_ARCH_DIR}" "${_install_root}")

file(COPY "${CPU_TADD_DIR}/main.cpp" "${CPU_TADD_DIR}/tadd_kernel.cpp" DESTINATION "${_source_dir}")
file(COPY "${TEST_COMMON_DIR}/test_common.h" DESTINATION "${_source_dir}")
file(COPY "${FIXTURE_WRITER}" DESTINATION "${_source_dir}")

file(WRITE "${_source_dir}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.18)
include(\"${_install_root}/pkg_inc/pto/costmodel/vfsim/cmake/a5_vf_mock.cmake\")
_pto_a5_find_llvm()
set(CMAKE_CXX_COMPILER \${PTO_A5_CLANGXX})
project(pto_a5_vfsim_external CXX)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(external_fixture_writer fixture_writer.cpp)
add_executable(external_full_host_tadd
    \"\${CMAKE_CURRENT_SOURCE_DIR}/main.cpp\"
    \"\${CMAKE_CURRENT_SOURCE_DIR}/tadd_kernel.cpp\"
    \"\${CMAKE_CURRENT_SOURCE_DIR}/vfsim_log_initializer.cpp\")
target_include_directories(external_full_host_tadd PRIVATE
    \"${_install_root}/include\"
    \"\${CMAKE_CURRENT_SOURCE_DIR}\")
target_compile_definitions(external_full_host_tadd PRIVATE
    __COSTMODEL
    __NPU_ARCH__=3101
    PTO_COMM_NOT_SUPPORTED)
target_link_libraries(external_full_host_tadd PRIVATE GTest::gtest_main Threads::Threads)
target_enable_a5_vf_mock(external_full_host_tadd)
")

file(WRITE "${_source_dir}/vfsim_log_initializer.cpp"
"#include \"pto/costmodel/a5/cce_costmodel/vf_cost.hpp\"
namespace {
struct Initializer {
    Initializer()
    {
        ::pto::mocker::vf::SetVfPredictionOptions(
            {::pto::mocker::vf::VfSimLogLevel::SUMMARY, {}});
    }
};
Initializer g_initializer;
} // namespace
")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${_source_dir}" -B "${_build_dir}" -DCMAKE_BUILD_TYPE=Debug
            -DPTO_A5_LLVM_CONFIG=${PTO_A5_LLVM_CONFIG}
            -DPTO_A5_CLANGXX=${PTO_A5_CLANGXX}
    RESULT_VARIABLE _configure_result
    OUTPUT_VARIABLE _configure_output
    ERROR_VARIABLE _configure_error)
if(NOT _configure_result EQUAL 0)
    message(FATAL_ERROR "external VfSim project configure failed:\n${_configure_output}\n${_configure_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${_build_dir}"
    RESULT_VARIABLE _build_result
    OUTPUT_VARIABLE _build_output
    ERROR_VARIABLE _build_error)
if(NOT _build_result EQUAL 0)
    message(FATAL_ERROR "external VfSim project build failed:\n${_build_output}\n${_build_error}")
endif()

file(GLOB_RECURSE _link_files "${_build_dir}/CMakeFiles/external_full_host_tadd.dir/link.txt")
if(NOT _link_files)
    message(FATAL_ERROR "external VfSim project link line was not found")
endif()
list(GET _link_files 0 _link_file)
file(READ "${_link_file}" _link_line)
if(_link_line MATCHES "libvfsim_native_core\\.a")
    message(FATAL_ERROR "external VfSim project linked libvfsim_native_core.a directly:\n${_link_line}")
endif()
if(NOT _link_line MATCHES "libpto_a5_vfsim\\.a")
    message(FATAL_ERROR "external VfSim project did not link libpto_a5_vfsim.a:\n${_link_line}")
endif()

execute_process(
    COMMAND "${_build_dir}/external_fixture_writer" "${_case_dir}"
    RESULT_VARIABLE _fixture_result
    OUTPUT_VARIABLE _fixture_output
    ERROR_VARIABLE _fixture_error)
if(NOT _fixture_result EQUAL 0)
    message(FATAL_ERROR "external VfSim project fixture setup failed:\n${_fixture_output}\n${_fixture_error}")
endif()

execute_process(
    COMMAND "${_build_dir}/external_full_host_tadd" --gtest_filter=TADDTest.case_float_64x64_64x64_64x64
    WORKING_DIRECTORY "${_build_dir}"
    RESULT_VARIABLE _run_result
    OUTPUT_VARIABLE _run_output
    ERROR_VARIABLE _run_error)
if(NOT _run_result EQUAL 0 OR NOT _run_error MATCHES "\\[VfSim\\] status=VfSimHit")
    message(FATAL_ERROR "external VfSim project run failed or did not report VfSimHit:\n${_run_output}\n${_run_error}")
endif()
