cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mtcl_dfl)


find_package(Torch REQUIRED HINTS $ENV{TORCH_HOME})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory( master-slave )
add_subdirectory( master-slave-detached )
