#-------------------------------------------------------------------#
# Project configuration
#-------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.26)
project(fast_deconv LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
# set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#-------------------------------------------------------------------#
# Cuda configuration
#-------------------------------------------------------------------#
find_package(CUDAToolkit REQUIRED)

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
target_compile_features(fast_deconv INTERFACE cxx_std_20 cuda_std_20)
target_link_libraries(fast_deconv INTERFACE CUDA::cudart)
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
# nanobind_add_module(_fast_deconv_ext
#     src/python/module.cpp
# )
# target_link_libraries(_fast_deconv_ext PRIVATE fast_deconv)
# set_target_properties(_fast_deconv_ext PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/fast_deconv")

# set_target_properties(_fast_deconv_ext  PROPERTIES
    # LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/fast_deconv)

#-------------------------------------------------------------------#
# Example executable
#-------------------------------------------------------------------#
add_executable(fast_deconv_app src/main.cu)
target_link_libraries(fast_deconv_app PRIVATE fast_deconv)

#-------------------------------------------------------------------#
# Installation
#-------------------------------------------------------------------#
# Nothing for now
