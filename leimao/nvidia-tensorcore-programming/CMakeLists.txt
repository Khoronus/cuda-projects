cmake_minimum_required(VERSION 3.17)
project(tensorcore_programming CUDA)

enable_language(CXX)  # Ensure C++ is enabled

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 14)

# Set architecture to A6000 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(tensorcore_programming tensorcore_programming.cu)

# Link CUDA libraries
target_link_libraries(tensorcore_programming CUDA::cudart)