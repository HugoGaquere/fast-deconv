#-------------------------------------------------------------------#
# Project configuration
#-------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.26)
project(fast_deconv LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
find_package(CUDA 12.8 REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

#-------------------------------------------------------------------#
# cuTensor configuration
#-------------------------------------------------------------------#
if(NOT DEFINED ENV{CUTENSOR_ROOT} AND NOT DEFINED CUTENSOR_ROOT)
  message(FATAL_ERROR "CUTENSOR_ROOT not set!")
else()
  if(DEFINED ENV{CUTENSOR_ROOT})
    set(CUTENSOR_ROOT "$ENV{CUTENSOR_ROOT}")
  endif()
  message("-- Looking for cuTENSOR in ${CUTENSOR_ROOT}")
  if(NOT EXISTS ${CUTENSOR_ROOT})
    message(FATAL_ERROR "Cannot find CUTENSOR_ROOT")
  endif()
endif()

if(NOT TARGET cutensor)
  add_library(cutensor SHARED IMPORTED)
  set(LIB_DIR "/lib64/libcutensor/${CUDA_VERSION_MAJOR}")
  set(CUTENSOR_LIBRARY_NAME "libcutensor.so")
  set_target_properties(cutensor PROPERTIES
    IMPORTED_LOCATION "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_NAME}"
    IMPORTED_IMPLIB "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_DEF}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_ROOT}/include")
endif()

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
target_link_libraries(fast_deconv INTERFACE cutensor)
set_target_properties(fast_deconv PROPERTIES CUDA_ARCHITECTURES native)


#-------------------------------------------------------------------#
# Nanobind configuration
#-------------------------------------------------------------------#
# find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
#
# # Detect the installed nanobind package and import it into CMake
# execute_process(
#   COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
#   OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)
#
# list(PREPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
#
# find_package(nanobind CONFIG REQUIRED)

#-------------------------------------------------------------------#
# Python extension
#-------------------------------------------------------------------#
# set_source_files_properties(
#     src/python/module.cpp
#     PROPERTIES LANGUAGE CUDA
# )
#
# nanobind_add_module(pyfast_deconv
#     src/python/module.cpp
# )
# target_link_libraries(pyfast_deconv PRIVATE fast_deconv)

#-------------------------------------------------------------------#
# Example executable
#-------------------------------------------------------------------#
add_executable(fast_deconv_app 
  src/main.cu
  src/examples/kronecker_example.cu
  src/examples/kronecker_tensor_example.cu
  src/examples/wscms_example.cu
)
target_link_libraries(fast_deconv_app PRIVATE fast_deconv)

#-------------------------------------------------------------------#
# Installation
#-------------------------------------------------------------------#
# install(TARGETS pyfast_deconv
#     LIBRARY DESTINATION pyfast_deconv
# )
#
# install(FILES python/pyfast_deconv/__init__.py
#     DESTINATION pyfast_deconv
# )
