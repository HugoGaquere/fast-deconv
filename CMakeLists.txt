#-------------------------------------------------------------------#
# Project configuration
#-------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.26)

set(CMAKE_CUDA_ARCHITECTURES "native")

project(fast_deconv LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

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

add_library(fast_deconv STATIC
        src/fast_deconv/algorithm/scales.cu
        src/fast_deconv/algorithm/wscms.cu
        src/fast_deconv/algorithm/wscms_class.cu
        src/fast_deconv/common/mask.cu
        src/fast_deconv/common/gain.cu
        src/fast_deconv/common/clean.cu
        src/fast_deconv/common/multi_frequency.cu
        src/fast_deconv/common/convergence.cpp
        src/fast_deconv/linalg/fft.cu
        src/fast_deconv/linalg/linalg.cu
        src/fast_deconv/linalg/pseudo_inverse.cu
        src/fast_deconv/matrix/argmax.cu
        src/fast_deconv/matrix/max.cu
        src/fast_deconv/matrix/rms.cu
        src/fast_deconv/matrix/stats.cu
        src/fast_deconv/morphology/dilation.cu
        src/fast_deconv/morphology/roi.cu
)

target_link_libraries(fast_deconv PUBLIC
  fmt::fmt
  spdlog::spdlog
  emu::emu
  CUDA::cublas
  CUDA::cufft
)

target_include_directories(fast_deconv PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
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

if (SKBUILD)
  # Let pybind11 find the python interpreter by itself.
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_fast_deconv
    src/python/fast_deconv_module.cu
    src/python/bindings/core.cu
    src/python/bindings/wscms.cu
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
if (NOT SKBUILD)
  add_executable(fast_deconv_app src/main.cu)
  target_link_libraries(fast_deconv_app PRIVATE fast_deconv)

  add_executable(example_wscms src/example_wscms.cu)
  target_link_libraries(example_wscms PRIVATE fast_deconv)

  add_executable(example_peer_access src/example_peer_access.cu)
  target_link_libraries(example_peer_access PRIVATE fast_deconv)

  add_executable(probe_fft_workspace src/probe_fft_workspace.cu)
  target_link_libraries(probe_fft_workspace PRIVATE CUDA::cufft)
endif()

#-------------------------------------------------------------------#
# Tests
#-------------------------------------------------------------------#
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS AND NOT SKBUILD)
  enable_testing()
  find_package(GTest CONFIG REQUIRED)

  foreach(_tp_target IN ITEMS GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
    if(TARGET ${_tp_target})
      set_target_properties(${_tp_target} PROPERTIES SYSTEM TRUE)
    endif()
  endforeach()

  add_executable(fast_deconv_tests
    tests/cpp/test_main.cpp
    tests/cpp/test_scale_selection.cu
    tests/cpp/morphology.cu
    tests/cpp/test_mask.cu
  )

  target_link_libraries(fast_deconv_tests PRIVATE
    fast_deconv
    GTest::gtest
    GTest::gtest_main
  )

  target_include_directories(fast_deconv_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp
  )

  include(GoogleTest)
  gtest_discover_tests(fast_deconv_tests)
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
