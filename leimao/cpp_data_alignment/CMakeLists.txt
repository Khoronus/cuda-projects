cmake_minimum_required(VERSION 3.10)

# Project name
project(CPPDataAlignment)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the source files
set(SOURCES
    main.cpp
)

# Add the executable
add_executable(CPPDataAlignment ${SOURCES})