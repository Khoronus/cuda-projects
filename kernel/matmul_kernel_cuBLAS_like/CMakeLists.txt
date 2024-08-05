cmake_minimum_required(VERSION 3.10)

# Project name
project(matmul_kernel_cuBLAS_like)

# Find CUDA
find_package(CUDA REQUIRED)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the architecture to A100 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Add the source files
set(SOURCES
    naive_kernel.cu
)

# Add the executable
cuda_add_executable(naive_kernel ${SOURCES})

# Set CUDA flags
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xptxas -O0")

# Optionally, set the flag to -Xptxas -O3 if needed
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xptxas -O3")

# Add tests if enabled
if (BUILD_TESTS)
    # Add the source files
    set(SOURCES
        test_naive_kernel.cu
    )

    # Add the executable
    cuda_add_executable(test_naive_kernel ${SOURCES})

    # Link test executable against Google Test and CUDA libraries
    target_link_libraries(test_naive_kernel gtest gtest_main CUDA::cudart)

    # Register tests
    add_test(NAME test_naive_kernel COMMAND test_naive_kernel)
endif()