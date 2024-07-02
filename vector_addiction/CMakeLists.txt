cmake_minimum_required(VERSION 3.10)

# Project name
project(VectorAddiction)

# Find CUDA
find_package(CUDA REQUIRED)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Add the source files
set(SOURCES
    vector_addiction.cpp
    kernel.cu
)

# Add the executable
cuda_add_executable(VectorAddiction ${SOURCES})

# Link CUDA libraries
target_link_libraries(VectorAddiction ${CUDA_LIBRARIES})
