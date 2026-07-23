# a5_vf_mock.cmake — A5 CCE mock 的 pass 工具链封装。
# target_enable_a5_vf_mock(target):一行给 A5 测试 target 启用 LLVM pass 插桩(拦截 __VEC_SCOPE__ 内的 for)。
# 内部:探测本机 LLVM(≥14)+ 编 PtoLoopTracePass(MODULE)+ 给 target 加 -O0 -g -fpass-plugin。Mac SDK/ld 坑在此处理。
# 用法:include 本文件后 add_executable(tadd main.cpp) + target_enable_a5_vf_mock(tadd)。

# 缓存本文件目录:macro/function 执行时 CMAKE_CURRENT_LIST_DIR 会变成调用者目录(不指向本文件),
# 导致 _pass_src 拼错。在文件作用域(include 时)固化。
set(_pto_a5_self_dir "${CMAKE_CURRENT_LIST_DIR}")
set(_pto_a5_repo_dir "${CMAKE_CURRENT_LIST_DIR}/..")

function(_pto_a5_build_vfsim)
    if(TARGET pto_a5_vfsim)
        return()
    endif()
    set(_vfsim_dir "${_pto_a5_repo_dir}/include/pto/costmodel/a5/VfSim")
    add_library(pto_a5_vfsim STATIC
        ${_vfsim_dir}/VfInfo.cpp
        ${_vfsim_dir}/JsonVfInfoAdapter.cpp
        ${_vfsim_dir}/Json.cpp
        ${_vfsim_dir}/ParamDB.cpp
        ${_vfsim_dir}/ISATraits.cpp
        ${_vfsim_dir}/ProgramAnalysis.cpp
        ${_vfsim_dir}/ProgramCanonicalization.cpp
        ${_vfsim_dir}/ProgramVregLiveRangeNormalization.cpp
        ${_vfsim_dir}/ProgramFlatten.cpp
        ${_vfsim_dir}/IFU.cpp
        ${_vfsim_dir}/IDU.cpp
        ${_vfsim_dir}/OOO.cpp
        ${_vfsim_dir}/SimulatorRunner.cpp
        ${_vfsim_dir}/VfSimCostModel.cpp)
    target_include_directories(pto_a5_vfsim PUBLIC "${_pto_a5_repo_dir}/include")
    target_compile_features(pto_a5_vfsim PUBLIC cxx_std_20)
    target_compile_definitions(pto_a5_vfsim PRIVATE PTO_VFSIM_SOURCE_ROOT="${_vfsim_dir}")
endfunction()

# 探测 LLVM + 配对 clang。llvm-config 路径可缓存，但 version/bindir/cxxflags 必须
# 每次 configure 重新查询，因为它们不是 cache 变量。
macro(_pto_a5_find_llvm)
    if(NOT PTO_A5_LLVM_CONFIG)
        if(DEFINED ENV{LLVM_CONFIG} AND EXISTS "$ENV{LLVM_CONFIG}")
            set(PTO_A5_LLVM_CONFIG "$ENV{LLVM_CONFIG}")
        else()
            file(GLOB _pto_a5_bindirs /usr/lib/llvm-*/bin
                 /usr/local/opt/llvm@*/bin /usr/local/opt/llvm/bin)
            find_program(PTO_A5_LLVM_CONFIG
                NAMES llvm-config llvm-config-20 llvm-config-19 llvm-config-18
                      llvm-config-17 llvm-config-16 llvm-config-15 llvm-config-14
                PATHS ${_pto_a5_bindirs} /usr/local/bin /usr/bin)
        endif()
    endif()
    if(NOT PTO_A5_LLVM_CONFIG)
        message(FATAL_ERROR "a5_vf_mock: llvm-config 未找到。Mac: brew install llvm@18; "
            "Linux: sudo apt install clang-N llvm-N-dev(N>=14)。或设 LLVM_CONFIG env。")
    endif()
    execute_process(COMMAND ${PTO_A5_LLVM_CONFIG} --version
        OUTPUT_VARIABLE PTO_A5_LLVM_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PTO_A5_LLVM_CONFIG} --bindir
        OUTPUT_VARIABLE PTO_A5_LLVM_BINDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "^([0-9]+)" PTO_A5_LLVM_MAJOR "${PTO_A5_LLVM_VERSION}")
    if(NOT PTO_A5_CLANGXX OR NOT EXISTS "${PTO_A5_CLANGXX}")
        unset(PTO_A5_CLANGXX CACHE)
        find_program(PTO_A5_CLANGXX NAMES clang++-${PTO_A5_LLVM_MAJOR} clang++
            PATHS ${PTO_A5_LLVM_BINDIR} /usr/bin /usr/local/bin NO_DEFAULT_PATH)
        if(NOT PTO_A5_CLANGXX)
            find_program(PTO_A5_CLANGXX NAMES clang++-${PTO_A5_LLVM_MAJOR} clang++)
        endif()
    endif()
    if(NOT PTO_A5_CLANGXX)
        message(FATAL_ERROR "a5_vf_mock: 未找到与 LLVM ${PTO_A5_LLVM_MAJOR} 匹配的 clang++")
    endif()
    execute_process(COMMAND ${PTO_A5_LLVM_CONFIG} --cxxflags
        OUTPUT_VARIABLE PTO_A5_LLVM_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    separate_arguments(PTO_A5_LLVM_CXXFLAGS NATIVE_COMMAND "${PTO_A5_LLVM_CXXFLAGS}")
    message(STATUS "a5_vf_mock: LLVM ${PTO_A5_LLVM_VERSION} (${PTO_A5_LLVM_CONFIG})")
endmacro()

# 编 PtoLoopTracePass(MODULE,只做一次)。
macro(_pto_a5_build_pass)
    if(NOT TARGET PtoLoopTracePass)
    _pto_a5_find_llvm()
    set(_pass_src "${_pto_a5_self_dir}/../include/pto/costmodel/a5/PtoLoopTracePass.cpp")
    add_library(PtoLoopTracePass MODULE ${_pass_src})
    target_compile_options(PtoLoopTracePass PRIVATE ${PTO_A5_LLVM_CXXFLAGS}
        -Wno-unused-command-line-argument -Wno-unknown-warning-option)
    if(APPLE)
        # Mac:CommandLineTools 老 ld 解析不了新 SDK 的 tbd v4,选 <16 的 SDK,插件借宿主 C++ 运行时。
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

# 公开:给 target 启用 A5 VF pass 插桩。
function(target_enable_a5_vf_mock target)
    _pto_a5_build_pass()
    _pto_a5_build_vfsim()
    add_dependencies(${target} PtoLoopTracePass)
    target_link_libraries(${target} PRIVATE pto_a5_vfsim)
    target_compile_options(${target} PRIVATE
        -O0 -g -fpass-plugin=$<TARGET_FILE:PtoLoopTracePass>)
    message(STATUS "a5_vf_mock: ${target} 已启用 pass 插桩(-O0 -g -fpass-plugin)")
endfunction()
