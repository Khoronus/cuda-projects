cmake_minimum_required(VERSION 3.10)

# Project name
project(MonteCarloSym)

# Find CUDA
find_package(CUDA REQUIRED)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Add the source files
set(SOURCES
    main.cpp
    kernel.cu
)

# Add the executable
cuda_add_executable(MonteCarloSym ${SOURCES})

# Link CUDA and cuRAND libraries
target_link_libraries(MonteCarloSym ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})