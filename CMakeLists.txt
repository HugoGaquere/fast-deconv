cmake_minimum_required(VERSION 3.26)
project(fast_deconv LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#-------------------------------------------------------------------#
# Nanobind
#-------------------------------------------------------------------#
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

# Perform an optimized release build by default unless another build type is specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(kernels
    src/kernels.cpp
    src/utils.cuh
    src/argmax.cuh
    src/argmax.cu
)

#-------------------------------------------------------------------#
# Cuda configuration
#-------------------------------------------------------------------#
find_package(CUDAToolkit REQUIRED)

set_target_properties(kernels PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/fast_deconv
    OUTPUT_NAME "kernels"
    CUDA_ARCHITECTURES native
)

target_link_libraries(kernels PRIVATE CUDA::cudart)

#-------------------------------------------------------------------#
# Finalize
#-------------------------------------------------------------------#

install(TARGETS kernels
    LIBRARY DESTINATION fast_deconv
)

