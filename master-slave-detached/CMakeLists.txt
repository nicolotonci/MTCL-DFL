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

#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
#set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")
#set(TORCH_CXX_FLAGS "-DSTRIP_ERROR_MESSAGES")

add_executable(master-d master.cpp)
add_executable(slave-d slave.cpp)
add_executable(broker broker.cpp)
target_link_libraries(master-d "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(slave-d "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT}) 

target_include_directories(master-d PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../fl/>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_include_directories(slave-d PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../fl/>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
set_property(TARGET slave-d PROPERTY CXX_STANDARD 20)
set_property(TARGET master-d PROPERTY CXX_STANDARD 20)
set_property(TARGET broker PROPERTY CXX_STANDARD 20)

find_library(LIBRT rt)
if(LIBRT)
target_link_libraries(master-d ${LIBRT})
target_link_libraries(slave-d ${LIBRT})
target_link_libraries(broker ${LIBRT})
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(master-d Threads::Threads)
target_link_libraries(slave-d Threads::Threads)
target_link_libraries(broker Threads::Threads)
find_package(MPI)

if ( NOT MPI_FOUND )
  message ("DFF excluding MPI")
  add_compile_definitions(DFF_EXCLUDE_MPI)
endif()

if (MPI_FOUND)
  target_link_libraries(master-d MPI::MPI_CXX)
  target_include_directories(master-d  PRIVATE ${MPI_INCLUDE_PATH})
  target_link_libraries(broker MPI::MPI_CXX)
  target_include_directories(broker  PRIVATE ${MPI_INCLUDE_PATH})
  target_link_libraries(slave-d MPI::MPI_CXX)
  target_include_directories(slave-d  PRIVATE ${MPI_INCLUDE_PATH})
  add_compile_definitions(ENABLE_MPI)
endif()

# use this instruction to copy config file 
#configure_file(mnist.json mnist.json COPYONLY)
