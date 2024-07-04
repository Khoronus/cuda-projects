cmake_minimum_required(VERSION 3.17)
project(matrix_multiplication CUDA CXX)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# C++ executable (naive_mm.cpp)
add_executable(naive_mm naive_mm.cpp)

# Specify CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 80 86)  # Adjust according to your GPU architecture, e.g., 80 for A6000

# CUDA executable (cublas_mm.cu)
add_executable(cublas_mm cublas_mm.cu)

# Link CUDA libraries to cublas_mm
target_link_libraries(cublas_mm CUDA::cudart CUDA::cublas)