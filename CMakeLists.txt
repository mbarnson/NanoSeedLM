# NanoSeedLM on Windows (MSVC) and Linux with NVIDIA CUDA.  The macOS / Metal build is the Makefile.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ctest --test-dir build
#
# Binaries land in build/bin.  Every C translation unit is compiled without FMA contraction: the scalar, vector and
# GPU paths of the seed search must agree bit for bit (as -ffp-contract=off in the Makefile).
cmake_minimum_required(VERSION 3.24)
project(NanoSeedLM LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 89)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

if(MSVC)
  set(COMPAT_DIR ${CMAKE_SOURCE_DIR}/win/compat)
  # /fp:precise without /fp:contract: no FMA contraction in host code.  /utf-8: sources and string literals are UTF-8.
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8> $<$<COMPILE_LANGUAGE:C,CXX>:/fp:precise>
                      $<$<COMPILE_LANGUAGE:C,CXX>:/W3> $<$<COMPILE_LANGUAGE:C,CXX>:/wd4244> $<$<COMPILE_LANGUAGE:C,CXX>:/wd4267>
                      $<$<COMPILE_LANGUAGE:C,CXX>:/wd4305> $<$<COMPILE_LANGUAGE:C,CXX>:/wd4996> $<$<COMPILE_LANGUAGE:C,CXX>:/wd4146>
                      $<$<COMPILE_LANGUAGE:C,CXX>:/FI${COMPAT_DIR}/posix_compat.h>
                      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
                      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/FI${COMPAT_DIR}/posix_compat.h>)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE NOMINMAX WIN32_LEAN_AND_MEAN)
  add_library(nslm_compat STATIC win/compat/posix_compat.c)
  target_include_directories(nslm_compat PUBLIC ${COMPAT_DIR})
  set(PLATFORM_LIBS nslm_compat ws2_32)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-ffp-contract=off> $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>)
  set(PLATFORM_LIBS pthread m)
endif()
# device code: the kernels decide where FMA is allowed (__fmul_rn / __fadd_rn where it is not)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -lineinfo")

# ---- the C99 library ------------------------------------------------------------------------------------------------
file(GLOB NSLM_LIB_SRC ${CMAKE_SOURCE_DIR}/nslm/lib_*.c)
add_library(nslm STATIC ${NSLM_LIB_SRC})
target_include_directories(nslm PUBLIC nslm harness)
target_link_libraries(nslm PUBLIC ${PLATFORM_LIBS})

# ---- tests (pure C) -------------------------------------------------------------------------------------------------
enable_testing()
foreach(t affine lfsr model_st moe mova_cfg search search4 sha256)
  add_executable(test_${t} tests/test_${t}.c)
  target_link_libraries(test_${t} nslm)
  add_test(NAME test_${t} COMMAND test_${t} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()
set_tests_properties(test_affine PROPERTIES SKIP_RETURN_CODE 77)
# tests that read a model folder (config.json, ...): -DNSLM_MODEL_DIR=<a K2-Horizon-MoVA folder>
set(NSLM_MODEL_DIR "" CACHE PATH "a NanoSeedLM model folder for the tests that need one")
if(NSLM_MODEL_DIR)
  set_tests_properties(test_mova_cfg PROPERTIES ENVIRONMENT "MOVA_DIR=${NSLM_MODEL_DIR}")
endif()
foreach(t chat_template tool_calls)
  add_executable(test_${t} tests/test_${t}.c harness/chat_template.c harness/tool_calls.c)
  target_link_libraries(test_${t} nslm)
  add_test(NAME test_${t} COMMAND test_${t} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()
