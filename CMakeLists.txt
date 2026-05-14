cmake_minimum_required(VERSION 3.16)

project(memory_accessor_ch4_minimal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_path(CBLAS_INCLUDE_DIR
  NAMES cblas.h
  PATH_SUFFIXES openblas
)

find_library(OPENBLAS_LIBRARY
  NAMES openblas
)

if(NOT CBLAS_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find cblas.h. Please install OpenBLAS development headers, e.g. libopenblas-dev.")
endif()

if(NOT OPENBLAS_LIBRARY)
  message(FATAL_ERROR "Could not find libopenblas. Please install OpenBLAS, e.g. libopenblas-dev.")
endif()

add_executable(blocked_trsv_benchmark
  src/main.cpp
)

target_include_directories(blocked_trsv_benchmark PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CBLAS_INCLUDE_DIR}
)

target_link_libraries(blocked_trsv_benchmark PRIVATE ${OPENBLAS_LIBRARY})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(blocked_trsv_benchmark PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    -Wall
    -Wextra
    -Wpedantic
  )
endif()
