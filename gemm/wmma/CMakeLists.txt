cmake_minimum_required(VERSION 3.17)
project(tensorcore_programming LANGUAGES CXX CUDA)

# Ensure C++ is enabled
enable_language(CXX)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set the architecture to A100 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Add the executable
add_executable(wmma wmma.cu)

# Link CUDA libraries
target_link_libraries(wmma CUDA::cudart)

# Ensure that the CUDA runtime library is linked dynamically
set_target_properties(wmma PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
)
