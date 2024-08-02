cmake_minimum_required(VERSION 3.17)
project(wmma LANGUAGES CXX CUDA)

# Ensure C++ is enabled
enable_language(CXX)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set the architecture to A100 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Add the executable
add_executable(wmma_executable wmma.cu)

# Link CUDA libraries
target_link_libraries(wmma_executable CUDA::cudart)

# Ensure that the CUDA runtime library is linked dynamically
set_target_properties(wmma_executable PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
)

# Add tests if enabled
if (BUILD_TESTS)
    # Add test executable
    add_executable(wmma_test test_wmma.cu)

    # Link test executable against Google Test and CUDA libraries
    target_link_libraries(wmma_test gtest gtest_main CUDA::cudart)

    # Register tests
    add_test(NAME wmma_test COMMAND wmma_test)
endif()
