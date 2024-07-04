cmake_minimum_required(VERSION 3.10)

# Project name
project(SharedMemoryBank)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the source files for the first executable
set(SOURCES1
    memory_bank.cpp
)

# Add the source files for the second executable
set(SOURCES2
    memory_bank2.cpp
)

# Add the first executable
add_executable(SharedMemoryBank1 ${SOURCES1})

# Add the second executable
add_executable(SharedMemoryBank2 ${SOURCES2})