#-------------------------------------------------------------------#
# Project configuration
#-------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.26)

set(CMAKE_CUDA_ARCHITECTURES "native")

project(fast_deconv LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(
  CONNAN_ARGS
  "" CACHE STRING
  "Additional conan args"
)

# Only when building the python module
if(SKBUILD)
  set(CONAN_INSTALL_ARGS "--build=missing;--options=fast_deconv/*:python_module=True;${CONAN_ARGS}")
endif()

# Set build type to release if nothing was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

#-------------------------------------------------------------------#
# Cuda configuration
#-------------------------------------------------------------------#
set(CMAKE_CUDA_FLAGS "-Wall")

#-------------------------------------------------------------------#
# Dependencies
#-------------------------------------------------------------------#
find_package(fmt CONFIG REQUIRED)
find_package(emu CONFIG REQUIRED)

#-------------------------------------------------------------------#
# MathDx configuration (optional)
#-------------------------------------------------------------------#
if(DEFINED ENV{MATHDX_ROOT})
  set(MATHDX_ROOT "$ENV{MATHDX_ROOT}")
endif()

if(DEFINED MATHDX_ROOT)
  message(STATUS "Looking for MathDx in ${MATHDX_ROOT}")
  find_package(mathdx
    COMPONENTS cublasdx cufftdx
    CONFIG
    PATHS "${MATHDX_ROOT}"
  )
  if(mathdx_FOUND)
    message(STATUS "MathDx found: ${mathdx_VERSION}")
    set(FAST_DECONV_HAS_MATHDX ON)
  else()
    message(WARNING "MathDx not found at ${MATHDX_ROOT}")
  endif()
endif()

#-------------------------------------------------------------------#
# Core library
#-------------------------------------------------------------------#

add_library(fast_deconv INTERFACE)

target_include_directories(fast_deconv INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(fast_deconv INTERFACE
  fmt::fmt
  emu::emu
  CUDA::cublas
  CUDA::cufft
  # CUDA::thrust
  # $<TARGET_NAME_IF_EXISTS:mathdx::cublasdx>
  $<TARGET_NAME_IF_EXISTS:mathdx::cufftdx>
)

# Set the targeting architectures
set_target_properties(fast_deconv PROPERTIES
                CUDA_ARCHITECTURES native
)

target_compile_features(fast_deconv INTERFACE cxx_std_20)

target_compile_options(fast_deconv INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:
  --extended-lambda
  --expt-relaxed-constexpr
  >
)

if(FAST_DECONV_HAS_MATHDX)
  target_compile_definitions(fast_deconv INTERFACE FAST_DECONV_HAS_MATHDX)
endif()

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
    src/python/bindings/debug.cu
    src/python/bindings/matrix.cu
    src/python/bindings/wscms.cu
    src/python/bindings/linalg.cu
  )

  target_link_libraries(_fast_deconv PRIVATE fast_deconv)

  # We only install the module and library. No need to install the headers for fast_deconv and _fast_deconv.
  install(TARGETS _fast_deconv DESTINATION ${SKBUILD_PROJECT_NAME})

  # We install all the libs next to each other in the fast_deconv package.
  # Let them simply link with each other when installed.
  set(CMAKE_INSTALL_RPATH "$ORIGIN")

  # We also copy/bundle every shared libraries to the destination
  install(TARGETS fast_deconv DESTINATION ${SKBUILD_PROJECT_NAME})
  file(GLOB SHARED_LIBS "${CMAKE_BINARY_DIR}/[!_]*.so*")
  install(FILES ${SHARED_LIBS} DESTINATION ${SKBUILD_PROJECT_NAME})
endif()

#-------------------------------------------------------------------#
# Example executable
#-------------------------------------------------------------------#
 add_executable(fast_deconv_app
   src/main.cu
   # src/examples/argmax_example.hpp
 )
 target_link_libraries(fast_deconv_app PRIVATE fast_deconv CUDA::curand)

#-------------------------------------------------------------------#
# Installation
#-------------------------------------------------------------------#
if (NOT SKBUILD)
  # Only install the headers and the library when building the C++ library.
  install(TARGETS fast_deconv
        FILE_SET HEADERS
  )
endif()




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
