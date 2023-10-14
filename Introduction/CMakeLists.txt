cmake_minimum_required(VERSION 3.8 FATAL_ERROR)


include(CTest)
enable_testing()



add_executable(main_In main.cu)
set_target_properties(main_In PROPERTIES CUDA_ARCHITECTURES "native")
set_target_properties(main_In PROPERTIES LINKER_LANGUAGE "CUDA")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
