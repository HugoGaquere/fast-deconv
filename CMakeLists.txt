#-------------------------------------------------------------------#
# Project configuration
#-------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.26)

# Selects include/fast_deconv/backend/<name> (headers) and
# src/fast_deconv/backend/<name> (implementations). Read before project()
# because it decides whether CUDA is enabled as a language at all.
set(FAST_DECONV_BACKEND "cuda" CACHE STRING "Compute backend: cuda | host")
set_property(CACHE FAST_DECONV_BACKEND PROPERTY STRINGS cuda host)

if(NOT FAST_DECONV_BACKEND MATCHES "^(cuda|host)$")
  message(FATAL_ERROR "FAST_DECONV_BACKEND must be 'cuda' or 'host', got '${FAST_DECONV_BACKEND}'")
endif()

if(FAST_DECONV_BACKEND STREQUAL "cuda")
  set(FAST_DECONV_WITH_CUDA ON)
else()
  set(FAST_DECONV_WITH_CUDA OFF)
endif()

set(_fd_languages CXX)
if(FAST_DECONV_WITH_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES "native")
  list(APPEND _fd_languages CUDA)
endif()

project(fast_deconv LANGUAGES ${_fd_languages})

if(FAST_DECONV_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# Build all objects as PIC so executables link cleanly on toolchains that
# default to PIE (conda-forge, recent Debian/Ubuntu/Fedora).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(
  CONAN_ARGS
  "" CACHE STRING
  "Additional conan args"
)

if(SKBUILD)
  set(CONAN_INSTALL_ARGS "--build=missing;--options=fast_deconv/*:python_module=True;${CONAN_ARGS}")
endif()

#-------------------------------------------------------------------#
# Host sanitizers (ASan + UBSan)
#-------------------------------------------------------------------#
# Instruments HOST code only — device code is out of reach for these
# sanitizers (use compute-sanitizer for kernels). Run the resulting test
# binaries with:
#   ASAN_OPTIONS=protect_shadow_gap=0 \
#   LSAN_OPTIONS=suppressions=<repo>/tests/lsan_suppressions.txt
option(FAST_DECONV_SANITIZE "Build host code with ASan + UBSan" OFF)

if(FAST_DECONV_SANITIZE)
  # nvcc splits -Xcompiler arguments on commas, so -fsanitize=address,undefined
  # must be passed as separate flags.
  set(_fd_san_flags
    -fsanitize=address
    -fsanitize=undefined
    -fno-sanitize-recover=undefined  # make UBSan findings fail the test
    -fno-omit-frame-pointer
  )
  set(_fd_san_flags_cuda "")
  foreach(_flag IN LISTS _fd_san_flags)
    list(APPEND _fd_san_flags_cuda "-Xcompiler=${_flag}")
  endforeach()

  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:${_fd_san_flags}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:${_fd_san_flags_cuda}>"
  )
  # CMake host-links CUDA targets with the CXX compiler, so plain flags work
  # for both link languages.
  add_link_options(-fsanitize=address -fsanitize=undefined)
endif()

#-------------------------------------------------------------------#
# Dependencies
#-------------------------------------------------------------------#
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(emu CONFIG REQUIRED)

# Mark third-party imported targets as SYSTEM so their headers come through
# as -isystem in compile_commands.json. clangd then suppresses diagnostics
# from these headers while still indexing them for completion/hover.
foreach(_tp_target IN ITEMS
    fmt::fmt
    spdlog::spdlog
    emu::emu
    emu::core
    CUDA::cublas
    CUDA::cufft
    CUDA::cudart
    CUDA::cuda_driver)
  if(TARGET ${_tp_target})
    set_target_properties(${_tp_target} PROPERTIES SYSTEM TRUE)
  endif()
endforeach()

#-------------------------------------------------------------------#
# Core library
#-------------------------------------------------------------------#

# Implementations for the selected backend. Globbed because the file set churns
# while the matrix/linalg modules migrate; swap for an explicit list once it settles.
file(GLOB_RECURSE FAST_DECONV_BACKEND_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fast_deconv/backend/${FAST_DECONV_BACKEND}/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fast_deconv/backend/${FAST_DECONV_BACKEND}/*.cpp
)

add_library(fast_deconv STATIC
        src/fast_deconv/algorithm/ddmsc_cycles.cpp
        src/fast_deconv/algorithm/ddmsc.cpp
        src/fast_deconv/common/convergence.cpp
        ${FAST_DECONV_BACKEND_SOURCES}
)

target_link_libraries(fast_deconv PUBLIC
  fmt::fmt
  spdlog::spdlog
)

if(FAST_DECONV_WITH_CUDA)
  target_link_libraries(fast_deconv PUBLIC emu::emu CUDA::cublas CUDA::cufft)
else()
  # emu::emu aggregates emu::cuda, which links cudart; emu::core is the CUDA-free half.
  target_link_libraries(fast_deconv PUBLIC emu::core)
  # nvtx3 ships with the CUDA toolkit, so the probes cannot compile without it.
  target_compile_definitions(fast_deconv PUBLIC FD_NVTX_DISABLE)
endif()

target_include_directories(fast_deconv PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/fast_deconv/backend/${FAST_DECONV_BACKEND}>
)

target_compile_features(fast_deconv PUBLIC cxx_std_20)

target_compile_definitions(fast_deconv PUBLIC
  $<$<CONFIG:Release>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO> # SPDLOG_LEVEL_OFF
  $<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG>
)

target_compile_options(fast_deconv PUBLIC
  $<$<COMPILE_LANGUAGE:CUDA>:
  --extended-lambda
  --expt-relaxed-constexpr
  --compiler-options=-Wall
    $<$<CONFIG:Release>:
      # --use_fast_math
      # -dlto
      --restrict
      --generate-line-info
    >
    $<$<CONFIG:Debug>:
      -g  # host debug
      # -G  # device debug
    >
  >
)


#-------------------------------------------------------------------#
# Python extension
#-------------------------------------------------------------------#

if (SKBUILD AND NOT FAST_DECONV_WITH_CUDA)
  message(FATAL_ERROR "The python extension has CUDA-only sources; build it with FAST_DECONV_BACKEND=cuda")
endif()

if (SKBUILD)
  # Let pybind11 find the python interpreter by itself.
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_fast_deconv
    src/python/fast_deconv_module.cu
    src/python/bindings/core.cu
    src/python/bindings/ddmsc.cu
  )

  target_link_libraries(_fast_deconv PRIVATE fast_deconv)

  # We only install the module and library. No need to install the headers for fast_deconv and _fast_deconv.
  install(TARGETS _fast_deconv DESTINATION ${SKBUILD_PROJECT_NAME})

  # We install all the libs next to each other in the fast_deconv package.
  # Let them simply link with each other when installed.
  set(CMAKE_INSTALL_RPATH "$ORIGIN")

  # We also copy/bundle every shared libraries to the destination
  # install(TARGETS fast_deconv DESTINATION ${SKBUILD_PROJECT_NAME})
  file(GLOB SHARED_LIBS "${CMAKE_BINARY_DIR}/[!_]*.so*")
  install(FILES ${SHARED_LIBS} DESTINATION ${SKBUILD_PROJECT_NAME})
endif()

#-------------------------------------------------------------------#
# Example executable
#-------------------------------------------------------------------#
# Sources here are all .cu, so they only build for the cuda backend.
if (NOT SKBUILD AND FAST_DECONV_WITH_CUDA)
  add_executable(fast_deconv_app src/main.cu)
  target_link_libraries(fast_deconv_app PRIVATE fast_deconv)

  add_executable(example_ddmsc src/example_ddmsc.cu)
  target_link_libraries(example_ddmsc PRIVATE fast_deconv)

  add_executable(bench_ddmsc src/bench_ddmsc.cu)
  target_link_libraries(bench_ddmsc PRIVATE fast_deconv)

  add_executable(bench_tiled_argmax src/bench_tiled_argmax.cu)
  target_link_libraries(bench_tiled_argmax PRIVATE fast_deconv)

  if(FAST_DECONV_SANITIZE)
    # nvcc's host compiler (g++-15 here) has no sanitizer link files on
    # Fedora — they ship only for the default gcc. Host-link the CUDA
    # executables with the CXX driver instead, like the test binaries.
    set_target_properties(
      fast_deconv_app example_ddmsc bench_ddmsc bench_tiled_argmax
      PROPERTIES LINKER_LANGUAGE CXX
    )
  endif()
endif()

#-------------------------------------------------------------------#
# Tests
#-------------------------------------------------------------------#
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS AND NOT SKBUILD)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

#-------------------------------------------------------------------#
# Archives, not used for now
#-------------------------------------------------------------------#

#-------------------------------------------------------------------#
# MathDx configuration
#-------------------------------------------------------------------#
# if(FAST_DECONV_HAS_MATHDX)
#   target_compile_definitions(fast_deconv INTERFACE FAST_DECONV_HAS_MATHDX)
# endif()
# if(DEFINED ENV{MATHDX_ROOT})
#   set(MATHDX_ROOT "$ENV{MATHDX_ROOT}")
# endif()
#
# if(DEFINED MATHDX_ROOT)
#   message(STATUS "Looking for MathDx in ${MATHDX_ROOT}")
#   find_package(mathdx
#     COMPONENTS cublasdx cufftdx
#     CONFIG
#     PATHS "${MATHDX_ROOT}"
#   )
#   if(mathdx_FOUND)
#     message(STATUS "MathDx found: ${mathdx_VERSION}")
#     set(FAST_DECONV_HAS_MATHDX ON)
#   else()
#     message(WARNING "MathDx not found at ${MATHDX_ROOT}")
#   endif()
# endif()
#-------------------------------------------------------------------#
# cuTensor configuration
#-------------------------------------------------------------------#
# if(NOT DEFINED ENV{CUTENSOR_ROOT} AND NOT DEFINED CUTENSOR_ROOT)
#   message(FATAL_ERROR "CUTENSOR_ROOT not set!")
# else()
#   if(DEFINED ENV{CUTENSOR_ROOT})
#     set(CUTENSOR_ROOT "$ENV{CUTENSOR_ROOT}")
#   endif()
#   message("-- Looking for cuTENSOR in ${CUTENSOR_ROOT}")
#   if(NOT EXISTS ${CUTENSOR_ROOT})
#     message(FATAL_ERROR "Cannot find CUTENSOR_ROOT")
#   endif()
# endif()
#
# if(NOT TARGET cutensor)
#   add_library(cutensor SHARED IMPORTED)
#   set(LIB_DIR "/lib64/libcutensor/${CUDA_VERSION_MAJOR}")
#   set(CUTENSOR_LIBRARY_NAME "libcutensor.so")
#   set_target_properties(cutensor PROPERTIES
#     IMPORTED_LOCATION "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_NAME}"
#     IMPORTED_IMPLIB "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_DEF}"
#     INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_ROOT}/include")
# endif()
