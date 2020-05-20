cmake_minimum_required (VERSION 3.2)

project("neural_network_prototype")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

set(CMAKE_CXX_STANDARD 14)

include(dlib)

add_subdirectory(libnnp)
add_subdirectory(example)
