cmake_minimum_required(VERSION 3.9)

project(cudaexpr LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CUDA_STANDARD 17)

add_library(cudaexpr SHARED
        cudaexpr.cu
)
target_compile_features(cudaexpr PRIVATE cxx_std_11)
target_link_libraries(cudaexpr PRIVATE ${TORCH_LIBRARIES} Python::Python)

# Use if the default GCC version gives issues.
target_compile_options(cudaexpr PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ccbin g++>)

# Use this for the executable file.
add_executable(cudaexprbin main.cpp cudaexpr.cu)
target_link_libraries(cudaexprbin "${TORCH_LIBRARIES}" Python::Python)
