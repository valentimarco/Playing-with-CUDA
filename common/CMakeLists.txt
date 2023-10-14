cmake_minimum_required(VERSION 3.7)

add_library(Common INTERFACE)
target_include_directories(Common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Common INTERFACE GLU GL glut)

