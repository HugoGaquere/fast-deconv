cmake_minimum_required(VERSION 3.18)
project(fast_deconv LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#-------------------------------------------------------------------#
# Pybind / Cuda configuration
#-------------------------------------------------------------------#
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(_core src/main.cpp src/add.cu)

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/fast_deconv
    OUTPUT_NAME "_core"
    CUDA_ARCHITECTURES native
)

target_link_libraries(_core PRIVATE CUDA::cudart)

install(TARGETS _core
    LIBRARY DESTINATION fast_deconv
)

