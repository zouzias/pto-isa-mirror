# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-1.0

include_guard(GLOBAL)

get_filename_component(_pto_a5_helper_dir "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
get_filename_component(_pto_a5_vfsim_dir "${_pto_a5_helper_dir}/.." ABSOLUTE)
get_filename_component(_pto_a5_root_dir "${_pto_a5_helper_dir}/../../../../.." ABSOLUTE)
set(_pto_a5_include_dir "${_pto_a5_root_dir}/include")
set(_pto_a5_pkg_inc_dir "${_pto_a5_root_dir}/pkg_inc")
set(_pto_a5_pass_source "${_pto_a5_include_dir}/pto/costmodel/a5/PtoLoopTracePass.cpp")
set(_pto_a5_stub_dir "${_pto_a5_include_dir}/pto/costmodel/stubs")

function(_pto_a5_require_installation_path path description)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR
            "Incomplete PTO-ISA A5 VfSim costmodel installation: missing ${description}:\n"
            "  ${path}\n"
            "Expected PTO-ISA root derived from helper:\n"
            "  ${_pto_a5_root_dir}")
    endif()
endfunction()

_pto_a5_require_installation_path("${_pto_a5_include_dir}/pto/pto-inst.hpp" "public PTO header")
_pto_a5_require_installation_path("${_pto_a5_pass_source}" "PtoLoopTracePass source")
_pto_a5_require_installation_path("${_pto_a5_stub_dir}" "costmodel stubs")
_pto_a5_require_installation_path("${_pto_a5_vfsim_dir}/native/CMakeLists.txt" "VfSim native CMake entry")
_pto_a5_require_installation_path(
    "${_pto_a5_vfsim_dir}/pto_adapter/vfsim_cost_model.cpp" "PTO VfSim adapter source")
_pto_a5_require_installation_path(
    "${_pto_a5_vfsim_dir}/pto_adapter/pto_canonical_lowering.cpp" "PTO canonical lowering source")
_pto_a5_require_installation_path(
    "${_pto_a5_vfsim_dir}/pto_adapter/pto_canonical_lowering.hpp" "PTO canonical lowering header")
_pto_a5_require_installation_path("${_pto_a5_vfsim_dir}/configs" "VfSim configuration directory")
_pto_a5_require_installation_path("${_pto_a5_vfsim_dir}/configs/isa.json" "VfSim ISA configuration")
_pto_a5_require_installation_path("${_pto_a5_vfsim_dir}/configs/uarch.json" "VfSim uarch configuration")

function(_pto_a5_add_vfsim_config_sync_target)
    if(TARGET pto_a5_vfsim_configs)
        return()
    endif()

    get_filename_component(_config_source "${_pto_a5_vfsim_dir}/configs" REALPATH)
    get_filename_component(
        _config_destination
        "${CMAKE_BINARY_DIR}/pkg_inc/pto/costmodel/vfsim/configs"
        ABSOLUTE)
    get_filename_component(_binary_root "${CMAKE_BINARY_DIR}" REALPATH)
    file(RELATIVE_PATH _config_destination_relative "${_binary_root}" "${_config_destination}")

    if(_config_destination_relative MATCHES "^\\.\\.(/|$)" OR
       IS_ABSOLUTE "${_config_destination_relative}")
        message(FATAL_ERROR
            "a5_vf_mock: refusing to synchronize VfSim configs outside CMAKE_BINARY_DIR:\n"
            "  ${_config_destination}")
    endif()

    if(EXISTS "${_config_destination}")
        get_filename_component(_config_destination_real "${_config_destination}" REALPATH)
        if(_config_destination_real STREQUAL _config_source)
            message(FATAL_ERROR
                "a5_vf_mock: VfSim config source and build destination are the same directory. "
                "Use an out-of-source build to prevent removal of source configurations:\n"
                "  ${_config_source}")
        endif()
    elseif(_config_destination STREQUAL _config_source)
        message(FATAL_ERROR
            "a5_vf_mock: VfSim config source and build destination are the same directory. "
            "Use an out-of-source build to prevent removal of source configurations:\n"
            "  ${_config_source}")
    endif()

    add_custom_target(pto_a5_vfsim_configs ALL
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_config_destination}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_config_destination}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${_config_source}" "${_config_destination}"
        COMMENT "Synchronizing A5 VfSim runtime configurations"
        VERBATIM)
endfunction()

function(_pto_a5_build_vfsim)
    if(TARGET pto_a5_vfsim)
        return()
    endif()
    set(VFSIM_ENABLE_MLIR_PLANNER OFF CACHE BOOL "Build the optional VfSim MLIR planner" FORCE)
    _pto_a5_add_vfsim_config_sync_target()
    add_subdirectory(
        "${_pto_a5_vfsim_dir}/native"
        "${CMAKE_CURRENT_BINARY_DIR}/pto_a5_vfsim_native"
        EXCLUDE_FROM_ALL)

    add_library(pto_a5_vfsim STATIC
        "${_pto_a5_vfsim_dir}/pto_adapter/pto_canonical_lowering.cpp"
        "${_pto_a5_vfsim_dir}/pto_adapter/vfsim_cost_model.cpp")
    target_include_directories(pto_a5_vfsim PRIVATE
        "${_pto_a5_include_dir}"
        "${_pto_a5_pkg_inc_dir}"
        "${_pto_a5_vfsim_dir}")
    target_link_libraries(pto_a5_vfsim PRIVATE vfsim::native_core_objects)
    target_compile_features(pto_a5_vfsim PUBLIC cxx_std_20)
    add_dependencies(pto_a5_vfsim pto_a5_vfsim_configs)
endfunction()

macro(_pto_a5_find_llvm)
    if(NOT PTO_A5_LLVM_CONFIG)
        if(DEFINED ENV{LLVM_CONFIG} AND EXISTS "$ENV{LLVM_CONFIG}")
            set(PTO_A5_LLVM_CONFIG "$ENV{LLVM_CONFIG}")
        else()
            file(GLOB _pto_a5_bindirs /usr/lib/llvm-*/bin
                 /usr/local/opt/llvm@*/bin /usr/local/opt/llvm/bin)
            find_program(PTO_A5_LLVM_CONFIG
                NAMES llvm-config
                PATHS ${_pto_a5_bindirs} /usr/local/bin /usr/bin)
            if(NOT PTO_A5_LLVM_CONFIG)
                file(GLOB _pto_a5_versioned_llvm_configs
                    /usr/local/bin/llvm-config-* /usr/bin/llvm-config-*)
                set(_pto_a5_best_llvm_version "0")
                foreach(_pto_a5_llvm_config_candidate IN LISTS _pto_a5_versioned_llvm_configs)
                    execute_process(COMMAND ${_pto_a5_llvm_config_candidate} --version
                        OUTPUT_VARIABLE _pto_a5_llvm_config_candidate_version
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        RESULT_VARIABLE _pto_a5_llvm_config_candidate_result)
                    if(_pto_a5_llvm_config_candidate_result EQUAL 0 AND
                       _pto_a5_llvm_config_candidate_version VERSION_GREATER _pto_a5_best_llvm_version)
                        set(_pto_a5_best_llvm_version "${_pto_a5_llvm_config_candidate_version}")
                        set(PTO_A5_LLVM_CONFIG "${_pto_a5_llvm_config_candidate}")
                    endif()
                endforeach()
            endif()
        endif()
    endif()
    if(NOT PTO_A5_LLVM_CONFIG)
        message(FATAL_ERROR "a5_vf_mock: llvm-config was not found. "
            "Install Clang/LLVM or set PTO_A5_LLVM_CONFIG or LLVM_CONFIG.")
    endif()
    execute_process(COMMAND ${PTO_A5_LLVM_CONFIG} --version
        OUTPUT_VARIABLE PTO_A5_LLVM_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PTO_A5_LLVM_CONFIG} --bindir
        OUTPUT_VARIABLE PTO_A5_LLVM_BINDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "^([0-9]+)" PTO_A5_LLVM_MAJOR "${PTO_A5_LLVM_VERSION}")
    if(NOT PTO_A5_LLVM_MAJOR)
        message(FATAL_ERROR "a5_vf_mock: unable to determine the LLVM major version from "
            "'${PTO_A5_LLVM_VERSION}' (${PTO_A5_LLVM_CONFIG})")
    endif()
    if(NOT PTO_A5_CLANGXX OR NOT EXISTS "${PTO_A5_CLANGXX}")
        unset(PTO_A5_CLANGXX CACHE)
        find_program(PTO_A5_CLANGXX NAMES clang++-${PTO_A5_LLVM_MAJOR} clang++
            PATHS ${PTO_A5_LLVM_BINDIR} /usr/bin /usr/local/bin NO_DEFAULT_PATH)
        if(NOT PTO_A5_CLANGXX)
            find_program(PTO_A5_CLANGXX NAMES clang++-${PTO_A5_LLVM_MAJOR} clang++)
        endif()
    endif()
    if(NOT PTO_A5_CLANGXX)
        message(FATAL_ERROR "a5_vf_mock: matching clang++ for LLVM ${PTO_A5_LLVM_MAJOR} not found")
    endif()
    execute_process(COMMAND ${PTO_A5_CLANGXX} --version
        OUTPUT_VARIABLE PTO_A5_CLANGXX_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "clang version ([0-9]+)" _pto_a5_clang_version_match "${PTO_A5_CLANGXX_VERSION}")
    set(PTO_A5_CLANGXX_MAJOR "${CMAKE_MATCH_1}")
    if(NOT PTO_A5_CLANGXX_MAJOR)
        message(FATAL_ERROR
            "a5_vf_mock: unable to determine the clang++ major version from '${PTO_A5_CLANGXX_VERSION}' "
            "at ${PTO_A5_CLANGXX}")
    endif()
    if(NOT PTO_A5_CLANGXX_MAJOR STREQUAL PTO_A5_LLVM_MAJOR)
        message(FATAL_ERROR
            "a5_vf_mock: clang++ and LLVM must use the same major version; found clang++ "
            "${PTO_A5_CLANGXX_MAJOR} at ${PTO_A5_CLANGXX}, but ${PTO_A5_LLVM_CONFIG} provides LLVM "
            "${PTO_A5_LLVM_VERSION}")
    endif()
    execute_process(COMMAND ${PTO_A5_LLVM_CONFIG} --cxxflags
        OUTPUT_VARIABLE PTO_A5_LLVM_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    separate_arguments(PTO_A5_LLVM_CXXFLAGS NATIVE_COMMAND "${PTO_A5_LLVM_CXXFLAGS}")
    message(STATUS "a5_vf_mock: LLVM ${PTO_A5_LLVM_VERSION} (${PTO_A5_LLVM_CONFIG})")
endmacro()

macro(_pto_a5_build_pass)
    if(NOT TARGET PtoLoopTracePass)
    _pto_a5_find_llvm()
    add_library(PtoLoopTracePass MODULE "${_pto_a5_pass_source}")
    target_compile_options(PtoLoopTracePass PRIVATE ${PTO_A5_LLVM_CXXFLAGS}
        -Wno-unused-command-line-argument -Wno-unknown-warning-option)
    if(APPLE)
        file(GLOB _sdks "/Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk")
        set(_sdk "")
        foreach(s IN LISTS _sdks)
            if(EXISTS "${s}/usr/include/c++/v1" AND EXISTS "${s}/usr/lib/libc++.tbd")
                get_filename_component(_n "${s}" NAME)
                string(REGEX MATCH "MacOSX([0-9]+\\.[0-9]+)\\.sdk" _ "${_n}")
                if(CMAKE_MATCH_1 AND CMAKE_MATCH_1 VERSION_LESS 16)
                    set(_sdk "${s}")
                endif()
            endif()
        endforeach()
        if(_sdk)
            target_compile_options(PtoLoopTracePass PRIVATE -isysroot ${_sdk})
            target_link_options(PtoLoopTracePass PRIVATE -undefined dynamic_lookup -nostdlib++
                -Wl,-syslibroot,${_sdk})
        else()
            target_link_options(PtoLoopTracePass PRIVATE -undefined dynamic_lookup -nostdlib++)
        endif()
    endif()
    endif()
endmacro()

function(target_enable_a5_vf_mock target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "a5_vf_mock: target '${target}' does not exist")
    endif()
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR
            "a5_vf_mock: target '${target}' must be compiled with Clang; found "
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} "
            "(${CMAKE_CXX_COMPILER})")
    endif()
    _pto_a5_build_pass()
    string(REGEX MATCH "^([0-9]+)" _pto_a5_target_clang_version "${CMAKE_CXX_COMPILER_VERSION}")
    set(_pto_a5_target_clang_major "${CMAKE_MATCH_1}")
    if(NOT _pto_a5_target_clang_major STREQUAL PTO_A5_LLVM_MAJOR)
        message(FATAL_ERROR
            "a5_vf_mock: target '${target}' uses Clang ${CMAKE_CXX_COMPILER_VERSION}, but the pass "
            "plugin uses LLVM ${PTO_A5_LLVM_VERSION}; their major versions must match")
    endif()
    _pto_a5_build_vfsim()
    add_dependencies(${target} PtoLoopTracePass)
    target_include_directories(${target} PRIVATE
        "${_pto_a5_stub_dir}"
        "${_pto_a5_pkg_inc_dir}")
    target_link_libraries(${target} PRIVATE pto_a5_vfsim)
    target_compile_options(${target} PRIVATE
        -O0 -g
        -Wno-macro-redefined
        -Wno-ignored-attributes
        "SHELL:-include stdint.h"
        "SHELL:-include stddef.h"
        -fpass-plugin=$<TARGET_FILE:PtoLoopTracePass>)
    message(STATUS "a5_vf_mock: ${target} enabled pass instrumentation (-O0 -g -fpass-plugin)")
endfunction()
