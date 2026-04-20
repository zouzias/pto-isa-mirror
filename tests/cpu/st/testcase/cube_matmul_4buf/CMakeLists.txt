# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
# --------------------------------------------------------------------------------

# Custom function for this test (includes both non-preload and preload kernels)
set(FILES_LIST 
    main.cpp 
    cube_matmul_4buf_kernel.cpp 
    cube_matmul_4buf_preload_kernel.cpp
    cube_matmul_4buf_bufid_kernel.cpp
)

add_executable(cube_matmul_4buf ${FILES_LIST})
target_compile_options(cube_matmul_4buf PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS} -std=c++${CMAKE_CXX_STANDARD})
target_include_directories(cube_matmul_4buf PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include/
    ${PROJECT_SOURCE_DIR}/../../../include/common
    ${PROJECT_SOURCE_DIR}/../stubs
    ${PROJECT_SOURCE_DIR}/../../common
)

target_link_libraries(cube_matmul_4buf PRIVATE
    ${PTO_CPU_SIM_GTEST_MAIN_TARGET}
    Threads::Threads
)
if (UNIX)
    target_link_libraries(cube_matmul_4buf PRIVATE m)
endif()
if (UNIX AND NOT APPLE)
    target_link_libraries(cube_matmul_4buf PRIVATE dl)
endif()
