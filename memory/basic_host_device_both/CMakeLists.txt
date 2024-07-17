cmake_minimum_required(VERSION 3.10)

# Project name
project(basic_host_device_both)

# Find CUDA
find_package(CUDA REQUIRED)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the source files
set(SOURCES
    memory_alloc.cu
)

# Add the executable
cuda_add_executable(memory_alloc ${SOURCES})

# Set CUDA flags
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xptxas -O0")

# Optionally, set the flag to -Xptxas -O3 if needed
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xptxas -O3")
