cmake_minimum_required(VERSION 3.10)

# Project name
project(CudaVectorizedMemoryAccess)

# Find CUDA
find_package(CUDA REQUIRED)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the source files
set(SOURCES
    memcpy.cu
)

# Add the executable
cuda_add_executable(CudaVectorizedMemoryAccess ${SOURCES})

# Set CUDA flags
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xptxas -O0")

# Optionally, set the flag to -Xptxas -O3 if needed
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xptxas -O3")