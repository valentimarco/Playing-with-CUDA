cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Introduction VERSION 0.1.0 LANGUAGES CXX CUDA)
enable_language(CUDA)



if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()



include(CTest)
enable_testing()



add_executable(main main.cu)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "native")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
