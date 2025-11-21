#-------------------------------------------------------------------#
# Project configuration
#-------------------------------------------------------------------#
# Force CUDA compiler
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc" CACHE FILEPATH "NVCC compiler" FORCE)
# Set cuTENSOR root
# set(CUTENSOR_ROOT "/usr" CACHE PATH "cuTENSOR install prefix")

set(CMAKE_CUDA_ARCHITECTURES "native")
cmake_minimum_required(VERSION 3.26)
project(fast_deconv LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(
  CONNAN_ARGS
  "" CACHE STRING
  "Additional conan args"
)

# Only when building the python module
if(SKBUILD)
  # CONAN_INSTALL_ARGS is used by cmake-conan.
  # We want to tell conan to call the install command with the python_module
  # option set to True.
  set(CONAN_INSTALL_ARGS "--build=missing;--options=fast_deconv/*:python_module=True;${CONAN_ARGS}")
endif()

# Set build type to release if nothing was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

#-------------------------------------------------------------------#
# Cuda configuration
#-------------------------------------------------------------------#
# find_package(CUDA 12.8 REQUIRED)
# include_directories("${CUDA_INCLUDE_DIRS}")

# Global CXX flags/options
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
 # set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_EXTENSIONS OFF)

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

#-------------------------------------------------------------------#
# Dependencies
#-------------------------------------------------------------------#
find_package(fmt CONFIG REQUIRED)
find_package(emu CONFIG REQUIRED)

#-------------------------------------------------------------------#
# Core library
#-------------------------------------------------------------------#
# Do not generate response file since lsp can't read it
# To disable for release
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)
set(CMAKE_NINJA_FORCE_RESPONSE_FILE OFF)

add_library(fast_deconv INTERFACE)

target_include_directories(fast_deconv INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(fast_deconv INTERFACE 
  fmt::fmt
  emu::emu
  cublas
  # cutensor
)

# Set the targeting architectures
set_target_properties(fast_deconv PROPERTIES CUDA_ARCHITECTURES native)
# Set the minimum C++ standard required
# target_compile_features(fast_deconv INTERFACE cxx_std_20)

#-------------------------------------------------------------------#
# Python extension
#-------------------------------------------------------------------#

if (SKBUILD)
  # Let pybind11 find the python interpreter by itself.
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_fast_deconv src/python/fast_deconv.cpp)

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
   src/examples/kronecker_example.cu
   src/examples/kronecker_mixed_product_example.cu
   # src/examples/kronecker_tensor_example.cu
   src/examples/wscms_example.cu
 )
 target_link_libraries(fast_deconv_app PRIVATE fast_deconv)

#-------------------------------------------------------------------#
# Installation
#-------------------------------------------------------------------#
if (NOT SKBUILD)
  # Only install the headers and the library when building the C++ library.
  install(TARGETS fast_deconv
        FILE_SET HEADERS
  )
endif()
