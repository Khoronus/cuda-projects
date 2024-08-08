cmake_minimum_required(VERSION 3.17)
project(cuda_projects 
  DESCRIPTION "Collection of simple C++/CUDA projects."
  HOMEPAGE_URL "https://github.com/Khoronus/cuda-projects"
  LANGUAGES CXX CUDA)

  # Set build type to RELEASE by default:
if (NOT CMAKE_BUILD_TYPE)
message(STATUS "No build type provided, default to RELEASE.")
set(CMAKE_BUILD_TYPE "RELEASE")
endif ()

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add option for building tests
option(BUILD_TESTS "Build tests" ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Include FetchContent module for Google Test if tests are enabled
if (BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1  # Use the appropriate version tag or branch
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Enable testing
enable_testing()

# Add subdirectories for each project
add_subdirectory(gemm/wmma)
add_subdirectory(kernel/cuda_kernel_fusion)
add_subdirectory(kernel/matmul_kernel_cuBLAS_like)
#add_subdirectory(memory/flops)


message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
message(STATUS "  Compiler ID          : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA                 : ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA include dirs    : ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "  CUDA library dir     : ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "  Build tests          : ${BUILD_TESTS}")
message(STATUS "")
