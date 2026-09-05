#-------------------------------------------------------------------#
# Project configuration
#-------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.26)

# Selects include/fast_deconv/backend/<name> (headers) and src/fast_deconv/backend/<name> (implementations).
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
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to build")
  endif()
  list(APPEND _fd_languages CUDA)
endif()

project(fast_deconv LANGUAGES ${_fd_languages})

# The provider reads CMAKE_CXX_STANDARD when constructing its Conan profile.
# Keep the target feature below too, so consumers inherit the API requirement.
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FAST_DECONV_BUILD_TOOLS "Build the replay executable" OFF)
option(FAST_DECONV_BUILD_BENCHMARKS "Build CUDA benchmarks" OFF)
if(SKBUILD)
  # Disable C++ tests for Python builds, even if the cache previously enabled them.
  set(BUILD_TESTING OFF)
endif()
include(CTest)

set(CONAN_ARGS "" CACHE STRING "Additional Conan arguments (semicolon-separated)")
# The dependency provider runs on the first find_package(), including toolkit
# discovery. Set all graph options before any such call.
if(SKBUILD)
  set(_fd_python_module True)
else()
  set(_fd_python_module False)
endif()
if(BUILD_TESTING)
  set(_fd_with_tests True)
else()
  set(_fd_with_tests False)
endif()
list(APPEND CONAN_INSTALL_ARGS
  ${CONAN_ARGS}
  "--options=fast-deconv/*:python_module=${_fd_python_module}"
  "--options=fast-deconv/*:backend=${FAST_DECONV_BACKEND}"
  "--options=fast-deconv/*:with_tests=${_fd_with_tests}"
)
if(BUILD_TESTING)
  list(APPEND CONAN_INSTALL_ARGS "--conf=tools.build:skip_test=False")
endif()

#-------------------------------------------------------------------#
# Host sanitizers (ASan + UBSan)
#-------------------------------------------------------------------#
# Instruments host code; use compute-sanitizer for CUDA device code.
# Environment workarounds for CUDA sanitizer test runs, when needed:
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
  # Sanitized executables and the Python module link with the CXX driver;
  # CUDA benchmarks explicitly select that driver below.
  add_link_options(-fsanitize=address -fsanitize=undefined)
endif()

#-------------------------------------------------------------------#
# Dependencies
#-------------------------------------------------------------------#
if(FAST_DECONV_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(emu CONFIG REQUIRED)

if(NOT FAST_DECONV_WITH_CUDA)
  find_package(pocketfft CONFIG REQUIRED)
  find_package(nvtx CONFIG REQUIRED)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

#-------------------------------------------------------------------#
# Core library
#-------------------------------------------------------------------#

# Collect the selected backend's implementations. CONFIGURE_DEPENDS makes
# additions and removals trigger CMake regeneration during the build.
file(GLOB_RECURSE FAST_DECONV_BACKEND_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fast_deconv/backend/${FAST_DECONV_BACKEND}/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fast_deconv/backend/${FAST_DECONV_BACKEND}/*.cpp
)

add_library(fast_deconv STATIC
        src/fast_deconv/algorithm/ddmsc_cycles.cpp
        src/fast_deconv/algorithm/ddmsc.cpp
        src/fast_deconv/algorithm/conv_psf_cache.cpp
        src/fast_deconv/algorithm/scales.cpp
        src/fast_deconv/common/convergence.cpp
        ${FAST_DECONV_BACKEND_SOURCES}
)

target_link_libraries(fast_deconv PUBLIC
  fmt::fmt
  spdlog::spdlog
)

if(FAST_DECONV_WITH_CUDA)
  target_link_libraries(fast_deconv PUBLIC emu::emu CUDA::cublas CUDA::cufft)
  # CUDA builds use the toolkit's NVTX3 headers.
  target_link_libraries(fast_deconv PUBLIC CUDA::nvtx3)
else()
  # Use only the backend-neutral emu component for host builds.
  target_link_libraries(fast_deconv PUBLIC emu::core)
  # PocketFFT headers are needed only when compiling backend sources.
  target_link_libraries(fast_deconv PRIVATE pocketfft::pocketfft)
  # Static-library link dependencies still reach consumers with PRIVATE scope.
  target_link_libraries(fast_deconv PRIVATE OpenMP::OpenMP_CXX)
  # PUBLIC: core/nvtx.hpp is a public header, so consumers need the include dir too.
  target_link_libraries(fast_deconv PUBLIC nvtx::nvtx)
endif()

target_include_directories(fast_deconv PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/fast_deconv/backend/${FAST_DECONV_BACKEND}>
)

# The static core is linked into the Python shared module.
set_target_properties(fast_deconv PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(fast_deconv PUBLIC cxx_std_20)
if(FAST_DECONV_WITH_CUDA)
  target_compile_features(fast_deconv PUBLIC cuda_std_20)
endif()

target_compile_definitions(fast_deconv PUBLIC
  $<$<CONFIG:Release>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO>
  $<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG>
)

# Propagate CUDA language-extension flags to consumers compiling CUDA sources.
target_compile_options(fast_deconv PUBLIC
  "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr>"
)

# Apply build preferences to our CUDA targets without exporting them to consumers.
function(fast_deconv_cuda_options target)
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--restrict;--generate-line-info>"
  )
endfunction()
fast_deconv_cuda_options(fast_deconv)

#-------------------------------------------------------------------#
# Python extension
#-------------------------------------------------------------------#

if(SKBUILD)
  # Use pybind11's CMake FindPython mode.
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)

  # Plain C++: the bindings only touch the backend-agnostic public API.
  pybind11_add_module(_fast_deconv
    src/python/fast_deconv_module.cpp
    src/python/bindings/core.cpp
    src/python/bindings/ddmsc.cpp
  )

  target_link_libraries(_fast_deconv PRIVATE fast_deconv)

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(_fast_deconv PROPERTIES INSTALL_RPATH "$ORIGIN")
    # Resolve the built module's dependency graph at install time. Keep the
    # platform C/C++ runtime and NVIDIA driver external; bundle other shared
    # dependencies (including the OpenMP and CUDA math runtimes) beside it.
    install(TARGETS _fast_deconv LIBRARY DESTINATION "${SKBUILD_PROJECT_NAME}")
    install(CODE "
      set(_fd_module \"$<TARGET_FILE:_fast_deconv>\")
      set(_fd_runtime_destination \"\${CMAKE_INSTALL_PREFIX}/${SKBUILD_PROJECT_NAME}\")
    ")
    install(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_runtime_dependencies.cmake")
  else()
    if(APPLE)
      set_target_properties(_fast_deconv PROPERTIES INSTALL_RPATH "@loader_path")
    endif()
    # Non-Linux dependency bundling is left to external wheel-repair tooling.
    install(TARGETS _fast_deconv DESTINATION "${SKBUILD_PROJECT_NAME}")
  endif()
endif()

#-------------------------------------------------------------------#
# Executables
#-------------------------------------------------------------------#
# Plain C++ driving the public API, so it builds for both backends.
if(FAST_DECONV_BUILD_TOOLS AND NOT SKBUILD)
  add_executable(replay_ddmsc src/replay_ddmsc.cpp)
  target_link_libraries(replay_ddmsc PRIVATE fast_deconv)
endif()

# The benchmarks are .cu, so they only build for the cuda backend.
if(FAST_DECONV_BUILD_BENCHMARKS AND NOT SKBUILD AND FAST_DECONV_WITH_CUDA)
  add_executable(bench_ddmsc src/bench_ddmsc.cu)
  target_link_libraries(bench_ddmsc PRIVATE fast_deconv)
  fast_deconv_cuda_options(bench_ddmsc)

  add_executable(bench_tiled_argmax src/bench_tiled_argmax.cu)
  target_link_libraries(bench_tiled_argmax PRIVATE fast_deconv)
  fast_deconv_cuda_options(bench_tiled_argmax)

  if(FAST_DECONV_SANITIZE)
    # Use the CXX driver's sanitizer runtime, like the test binaries.
    # It may differ from nvcc's host compiler and its available runtimes.
    set_target_properties(
      bench_ddmsc bench_tiled_argmax
      PROPERTIES LINKER_LANGUAGE CXX
    )
  endif()
endif()

#-------------------------------------------------------------------#
# Tests
#-------------------------------------------------------------------#
if(BUILD_TESTING AND NOT SKBUILD)
  add_subdirectory(tests/cpp)
endif()
