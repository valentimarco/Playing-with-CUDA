cmake_minimum_required(VERSION 3.7)
project(CUDA_Examples CXX CUDA)
enable_language(CUDA)



if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory(common)
add_subdirectory(Introduction)
add_subdirectory(ParallelProgramming)
add_subdirectory(ThreadCooperation)

# add_subdirectory(lib)

