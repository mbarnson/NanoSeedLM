# NanoSeedLM on Windows (MSVC) and Linux with NVIDIA CUDA.  The macOS / Metal build is the Makefile.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ctest --test-dir build
#
# Binaries land in build/bin.  Every C translation unit is compiled without FMA contraction: the scalar, vector and
# GPU paths of the seed search must agree bit for bit (as -ffp-contract=off in the Makefile).
cmake_minimum_required(VERSION 3.24)
# GPUs (BF16 tensor cores need sm_80 or newer): machine code for Ampere (sm_80, 86), Ada (sm_89) and, with CUDA 12.8 or
# newer, consumer Blackwell (sm_120); compute_89 PTX for the rest (Hopper sm_90, datacenter Blackwell sm_100), which the
# driver compiles at first use.  -DCMAKE_CUDA_ARCHITECTURES=native (or a list) overrides this.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80-real CACHE STRING "CUDA architectures (80-real: the default list below)")
endif()
project(NanoSeedLM LANGUAGES C CXX CUDA)
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "80-real")
  set(CMAKE_CUDA_ARCHITECTURES 80-real 86-real 89)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 120-real)
  endif()
  message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION}: architectures ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

if(MSVC)
  set(COMPAT_DIR ${CMAKE_SOURCE_DIR}/win/compat)
  # /fp:precise without /fp:contract: no FMA contraction in host code.  /utf-8: sources and string literals are UTF-8.
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8> $<$<COMPILE_LANGUAGE:C,CXX>:/fp:precise> $<$<AND:$<COMPILE_LANGUAGE:C>,$<C_COMPILER_ID:MSVC>>:/experimental:c11atomics>
                      $<$<COMPILE_LANGUAGE:C,CXX>:/W3> $<$<COMPILE_LANGUAGE:C,CXX>:/wd4244> $<$<COMPILE_LANGUAGE:C,CXX>:/wd4267>
                      $<$<COMPILE_LANGUAGE:C,CXX>:/wd4305> $<$<COMPILE_LANGUAGE:C,CXX>:/wd4996> $<$<COMPILE_LANGUAGE:C,CXX>:/wd4146>
                      $<$<COMPILE_LANGUAGE:C,CXX>:/FI${COMPAT_DIR}/posix_compat.h>
                      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
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
# The CPU seed kernels (the CPU expert path, plan D4) need Clang vectors and pthread_once (win/compat has it): any Clang C
# compiler, clang-cl on Windows too (-DCMAKE_C_COMPILER=clang-cl; nvcc keeps cl as its host compiler).  MSVC cl and GCC
# build the portable stub.
if(NOT CMAKE_C_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
  list(FILTER NSLM_LIB_SRC EXCLUDE REGEX "/lib_cpu_seed4\\.c$")
  list(APPEND NSLM_LIB_SRC ${CMAKE_SOURCE_DIR}/nslm/cpu_seed4_none.c)   # cpu_seed4_on() is 0: no CPU expert path
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  if(MSVC)   # clang-cl: the driver flags through /clang:
    set_source_files_properties(${CMAKE_SOURCE_DIR}/nslm/lib_cpu_seed4.c PROPERTIES COMPILE_OPTIONS "/clang:-mavx2;/clang:-mfma")
  else()
    set_source_files_properties(${CMAKE_SOURCE_DIR}/nslm/lib_cpu_seed4.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  endif()
endif()
add_library(nslm STATIC ${NSLM_LIB_SRC})
target_include_directories(nslm PUBLIC nslm harness)
target_link_libraries(nslm PUBLIC ${PLATFORM_LIBS})

# ---- tests (pure C) -------------------------------------------------------------------------------------------------
enable_testing()
foreach(t affine emulate gguf kvq lfsr linalg model_st moe ple qwen4_ref search search4 searchc searchp sha256)
  add_executable(test_${t} tests/test_${t}.c)
  target_link_libraries(test_${t} nslm)
  add_test(NAME test_${t} COMMAND test_${t} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()
set_tests_properties(test_affine PROPERTIES SKIP_RETURN_CODE 77)
if(CMAKE_C_COMPILER_ID MATCHES "^(Clang|AppleClang)$")   # the CPU seed kernels exist only in Clang builds
  add_executable(test_cpu_seed4 tests/test_cpu_seed4.c)
  target_link_libraries(test_cpu_seed4 nslm)
  add_test(NAME test_cpu_seed4 COMMAND test_cpu_seed4 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
# tests that read a model folder (config.json, ...): -DNSLM_MODEL_DIR=<a K2-Horizon-MoVA folder>
set(NSLM_MODEL_DIR "" CACHE PATH "a NanoSeedLM model folder for the tests that need one")
add_executable(test_mova_cfg tests/test_mova_cfg.c)
target_link_libraries(test_mova_cfg nslm)
add_test(NAME test_mova_cfg COMMAND test_mova_cfg WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_mova_cfg PROPERTIES SKIP_RETURN_CODE 77)   # 77: no model folder to read
if(NSLM_MODEL_DIR)
  set_tests_properties(test_mova_cfg PROPERTIES ENVIRONMENT "MOVA_DIR=${NSLM_MODEL_DIR}")
endif()
foreach(t chat_template tool_calls qwen4_template qwen4_tools)
  add_executable(test_${t} tests/test_${t}.c harness/chat_template.c harness/chat_qwen.c harness/tool_calls.c)
  target_link_libraries(test_${t} nslm)
  add_test(NAME test_${t} COMMAND test_${t} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()

# ---- the CUDA engine ------------------------------------------------------------------------------------------------
add_library(nslm_engine STATIC engine/mova_cuda_named.c engine/qwen4_cuda.c engine/qwen4_api.c engine/engine.c engine/kernels_moe.cu
            engine/kernels_qwen4.cu)
target_include_directories(nslm_engine PUBLIC engine nslm)
target_link_libraries(nslm_engine PUBLIC nslm CUDA::cudart_static)
set_target_properties(nslm_engine PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# ---- the tools ------------------------------------------------------------------------------------------------------
add_library(nslm_harness STATIC harness/platform.c harness/tokenizer.c harness/chat_template.c harness/chat_qwen.c harness/chat_model.c harness/tool_calls.c harness/kv_disk.c
            $<IF:$<BOOL:${APPLE}>,harness/telemetry.m,harness/telemetry_nvml.c>)
target_include_directories(nslm_harness PUBLIC harness)
if(WIN32)
  target_link_libraries(nslm_harness PUBLIC nslm icu shell32 bcrypt ws2_32)
elseif(APPLE)
  target_link_libraries(nslm_harness PUBLIC nslm icucore "-framework CoreFoundation" "-framework IOKit")
else()   # Linux: ICU's headers and libraries (Ubuntu / Debian: libicu-dev), found at configure time
  find_package(ICU COMPONENTS uc i18n)
  if(NOT ICU_FOUND)
    message(FATAL_ERROR "ICU not found: the tokenizer (harness/tokenizer.c) needs ICU's headers and libraries "
                        "(Ubuntu / Debian: apt install libicu-dev; Fedora: dnf install libicu-devel)")
  endif()
  target_link_libraries(nslm_harness PUBLIC nslm ICU::uc ICU::i18n)
endif()

foreach(t nslm-chat nslm-serve nslm-mova-bench nslm-mova-gen nslm-mova-score nslm-mova-plcheck nslm-mova-smoke nslm-mova-kbench
          nslm-mova-refcheck nslm-mova-routes nslm-mova-mlacapture nslm-qwen4-gen
          nslm-qwen4-kld nslm-handoff-bench)
  add_executable(${t} harness/${t}.c)
  target_link_libraries(${t} nslm_engine nslm_harness)
endforeach()

# ---- the compressor's GPU seed search (bit-identical to the C reference: no FMA contraction) -------------------------
add_library(nslm_search_gpu STATIC nslm/search.cu nslm/search4.cu nslm/searchc.cu nslm/searchp.cu)
target_include_directories(nslm_search_gpu PUBLIC nslm)
target_link_libraries(nslm_search_gpu PUBLIC nslm CUDA::cudart_static)
target_compile_options(nslm_search_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
add_executable(test_search4_gpu tests/test_search4_gpu.c)
target_link_libraries(test_search4_gpu nslm_search_gpu)
add_test(NAME test_search4_gpu COMMAND test_search4_gpu WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_executable(test_searchp_gpu tests/test_searchp_gpu.c)
target_link_libraries(test_searchp_gpu nslm_search_gpu)
add_test(NAME test_searchp_gpu COMMAND test_searchp_gpu WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_executable(test_searchc_gpu tests/test_searchc_gpu.c)   # CUDA only (no Metal kernel yet)
target_link_libraries(test_searchc_gpu nslm_search_gpu)
add_test(NAME test_searchc_gpu COMMAND test_searchc_gpu WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_executable(test_search_gpu tests/test_search_gpu.c)
target_link_libraries(test_search_gpu nslm_search_gpu)
add_test(NAME test_search_gpu COMMAND test_search_gpu WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_search_gpu test_search4_gpu test_searchc_gpu test_searchp_gpu PROPERTIES SKIP_RETURN_CODE 77)   # 77: no GPU

# the compressor: seed search (nslm-moe, nslm-dense), the packer, the bits probe
add_executable(nslm-moe nslm/moe.c)
target_link_libraries(nslm-moe nslm_search_gpu)
add_executable(nslm-dense nslm/dense.c)
target_link_libraries(nslm-dense nslm_search_gpu)
add_test(NAME test_dense_help COMMAND nslm-dense -h)
set_tests_properties(test_dense_help PROPERTIES WILL_FAIL TRUE)   # usage exits 2; a startup crash still fails
add_executable(nslm-mova-pack nslm/mova_pack.c)
target_link_libraries(nslm-mova-pack nslm)
add_executable(nslm-qwen4-pack nslm/qwen4_pack.c)
target_link_libraries(nslm-qwen4-pack nslm)
add_executable(nslm-qwen4-emulate nslm/qwen4_emulate.c)
target_link_libraries(nslm-qwen4-emulate nslm)
add_executable(nslm-bits-probe nslm/bits_probe.c)
target_link_libraries(nslm-bits-probe nslm)

# ---- engine tests (the same sources the Metal build runs) -----------------------------------------------------------
add_executable(test_engine tests/test_engine.c)
target_link_libraries(test_engine nslm_engine)
add_test(NAME test_engine COMMAND test_engine WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_engine PROPERTIES SKIP_RETURN_CODE 77)
add_executable(test_engine_formats tests/test_engine_formats.c)   # unsupported CUDA weight formats fail at load
target_link_libraries(test_engine_formats nslm_engine)
add_test(NAME test_engine_formats COMMAND test_engine_formats WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_engine_formats PROPERTIES SKIP_RETURN_CODE 77)
add_executable(test_mla_head_groups tests/test_mla_head_groups.c)
target_include_directories(test_mla_head_groups PRIVATE engine)
target_link_libraries(test_mla_head_groups nslm)
add_test(NAME test_mla_head_groups COMMAND test_mla_head_groups)
add_executable(test_mla_resident_head_policy tests/test_mla_resident_head_policy.c)
target_include_directories(test_mla_resident_head_policy PRIVATE engine)
add_test(NAME test_mla_resident_head_policy COMMAND test_mla_resident_head_policy)
add_executable(test_mova_kernels tests/test_mova_kernels.c tests/kernel_backend_cuda.c)   # the kernels one by one
target_link_libraries(test_mova_kernels nslm_engine)
add_test(NAME test_mova_kernels COMMAND test_mova_kernels WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_mova_kernels PROPERTIES SKIP_RETURN_CODE 77)
add_executable(test_qwen4_kernels tests/test_qwen4_kernels.c tests/kernel_backend_cuda.c)   # kernels vs nslm/qwen4_ops.h
target_link_libraries(test_qwen4_kernels nslm_engine)
add_test(NAME test_qwen4_kernels COMMAND test_qwen4_kernels WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_qwen4_kernels PROPERTIES SKIP_RETURN_CODE 77)
add_executable(test_qwen4_engine tests/test_qwen4_engine.c)   # the qwen4 CUDA engine against mlx-vlm on the tiny model
target_link_libraries(test_qwen4_engine nslm_engine)
add_test(NAME test_qwen4_engine COMMAND test_qwen4_engine WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_qwen4_engine PROPERTIES SKIP_RETURN_CODE 77)
add_executable(test_qwen4_api tests/test_qwen4_api.c)   # engine_api.h over the qwen4 engine (engine/qwen4_api.c)
target_link_libraries(test_qwen4_api nslm_engine)
add_test(NAME test_qwen4_api COMMAND test_qwen4_api WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_qwen4_api PROPERTIES SKIP_RETURN_CODE 77)
add_executable(test_qwen4_mtp tests/test_qwen4_mtp.c)   # the qwen4 CUDA engine's MTP head: drafts, verify rows, greedy decode
target_link_libraries(test_qwen4_mtp nslm_engine)
add_test(NAME test_qwen4_mtp COMMAND test_qwen4_mtp WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_qwen4_mtp PROPERTIES SKIP_RETURN_CODE 77)
add_executable(test_engine_dispatch tests/test_engine_dispatch.c)   # engine/engine.c: qwen4 or MoVA by config.json
target_link_libraries(test_engine_dispatch nslm_engine)
add_test(NAME test_engine_dispatch COMMAND test_engine_dispatch WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_engine_dispatch PROPERTIES SKIP_RETURN_CODE 77)
add_executable(test_tokenizer tests/test_tokenizer.c)
target_link_libraries(test_tokenizer nslm_harness)
add_test(NAME test_tokenizer COMMAND test_tokenizer WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_tokenizer PROPERTIES SKIP_RETURN_CODE 77)
if(NSLM_MODEL_DIR)
  set_tests_properties(test_tokenizer PROPERTIES ENVIRONMENT "MOVA_DIR=${NSLM_MODEL_DIR}")
endif()
add_executable(test_kv_disk tests/test_kv_disk.c)
target_link_libraries(test_kv_disk nslm_harness)
add_test(NAME test_kv_disk COMMAND test_kv_disk WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_executable(test_serve_splitter tests/test_serve_splitter.c)
target_link_libraries(test_serve_splitter nslm_engine nslm_harness)
add_test(NAME test_serve_splitter COMMAND test_serve_splitter WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# ---- Windows: every executable runs in the UTF-8 code page (paths from plat_init's UTF-8 argv) ----------------------
if(MSVC)
  get_property(NSLM_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  foreach(t ${NSLM_TARGETS})
    get_target_property(ty ${t} TYPE)
    if(ty STREQUAL "EXECUTABLE")
      target_sources(${t} PRIVATE ${CMAKE_SOURCE_DIR}/win/utf8.manifest)
    endif()
  endforeach()
endif()

add_executable(test_mla_full_resident_budget tests/test_mla_full_resident_budget.c)
target_include_directories(test_mla_full_resident_budget PRIVATE ${CMAKE_SOURCE_DIR}/engine)
add_test(NAME test_mla_full_resident_budget COMMAND test_mla_full_resident_budget)
