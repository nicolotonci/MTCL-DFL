# Check prefix paths
# list(APPEND CMAKE_PREFIX_PATH "")

foreach(path ${CMAKE_PREFIX_PATH})
  message("Path: " ${path})
endforeach(path)

find_path(RAPIDJSON_DIR NAMES "rapidjson/rapidjson.h" HINTS $ENV{RAPIDJSON_HOME})

if(RAPIDJSON_DIR STREQUAL "RAPIDJSON_DIR-NOTFOUND")
  message(FATAL_ERROR "Could not find RAPIDJSON library.")  
endif()
include_directories(${RAPIDJSON_DIR})
add_compile_definitions(ENABLE_CONFIGFILE)


find_path(MTCL_DIR NAMES "mtcl.hpp" HINTS $ENV{MTCL_HOME})

if(MTCL_DIR STREQUAL "MTCL_DIR-NOTFOUND")
  message(FATAL_ERROR "Could not find MTCL library.")  
endif()
include_directories(${MTCL_DIR})

find_package(Torch REQUIRED HINTS $ENV{TORCH_HOME})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(master master.cpp)
add_executable(slave slave.cpp)
target_link_libraries(master "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(slave "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT}) 
target_include_directories(master PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../fl/>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_include_directories(slave PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../fl/>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
set_property(TARGET master slave PROPERTY CXX_STANDARD 20)

find_library(LIBRT rt)
if(LIBRT)
target_link_libraries(master ${LIBRT})
target_link_libraries(slave ${LIBRT})
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(master Threads::Threads)
target_link_libraries(slave Threads::Threads)

find_package(MPI)

if ( NOT MPI_FOUND )
  message ("MPI not found")
endif()

if (MPI_FOUND)
  target_link_libraries(master MPI::MPI_CXX)
  target_include_directories(master  PRIVATE ${MPI_INCLUDE_PATH})
  target_link_libraries(slave MPI::MPI_CXX)
  target_include_directories(slave  PRIVATE ${MPI_INCLUDE_PATH})
  add_compile_definitions(ENABLE_MPI)
endif()

# use this instruction to copy config file 
#configure_file(config.json config.json COPYONLY)
