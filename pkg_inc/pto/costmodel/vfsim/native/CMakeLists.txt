# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.20)
project(VfSimNative LANGUAGES C CXX)

option(VFSIM_ENABLE_MLIR_PLANNER "Build MLIR IR planner entry point" OFF)

set(VFSIM_NATIVE_CORE_SOURCES
  ../api/native/CanonicalVfInfo.cpp
  ../api/native/CanonicalVfInfoValidation.cpp
  ../api/native/LegacyVfInfoAdapter.cpp
  ../api/native/InstructionCatalog.cpp
  ../api/native/UarchOverrideSchema.cpp
  ../api/native/VfInfo.cpp
  ../api/native/JsonVfInfoAdapter.cpp
  Json.cpp
  ControlUnit.cpp
  ParamCompat.cpp
  ParamDB.cpp
  ISATraits.cpp
  ProgramAnalysis.cpp
  ProgramCanonicalization.cpp
  ProgramVregLiveRangeNormalization.cpp
  CanonicalProgramLowering.cpp
  ProgramFlatten.cpp
  IFU.cpp
  IDU.cpp
  OOO.cpp
  SimulatorRunner.cpp
)

add_library(vfsim_native_core_objects OBJECT
  ${VFSIM_NATIVE_CORE_SOURCES}
)

add_library(vfsim::native_core_objects ALIAS vfsim_native_core_objects)

target_include_directories(vfsim_native_core_objects PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(vfsim_native_core_objects PUBLIC cxx_std_17)

# Keep the standalone native archive available to native-only consumers. PTO's
# A5 adapter embeds the same object files directly so its users only link one
# static archive.
add_library(vfsim_native_core STATIC)
add_library(vfsim::native_core ALIAS vfsim_native_core)
target_link_libraries(vfsim_native_core PUBLIC vfsim::native_core_objects)

if(VFSIM_ENABLE_MLIR_PLANNER)
  if(NOT TARGET MLIRIR)
    find_package(MLIR REQUIRED CONFIG)
  endif()

  add_library(vfsim_ir_planner STATIC
    IRPlanner.cpp
  )

  add_library(vfsim::ir_planner ALIAS vfsim_ir_planner)

  target_link_libraries(vfsim_ir_planner PUBLIC
    vfsim::native_core
    MLIRIR
    MLIRSupport
  )
  target_include_directories(vfsim_ir_planner PUBLIC
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
  )
endif()
