cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

include(CTest)
enable_testing()


add_executable(main_TC main.cu)
set_target_properties(main_TC PROPERTIES CUDA_ARCHITECTURES "native")
target_link_libraries(main_TC PRIVATE Common)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
