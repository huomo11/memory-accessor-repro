cmake_minimum_required(VERSION 3.16)

project(memory_accessor_ch4_minimal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(MKL_ROOT "/opt/conda" CACHE PATH "MKL root directory")

find_package(Threads REQUIRED)

find_path(CBLAS_INCLUDE_DIR
  NAMES cblas.h
  PATHS
    /usr/include
    /usr/include/x86_64-linux-gnu
    "${MKL_ROOT}/include"
)

if(NOT CBLAS_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find cblas.h. Install a CBLAS development header package or pass a valid include path.")
endif()

set(MKL_RT_SO "${MKL_ROOT}/lib/libmkl_rt.so")
set(MKL_RT_SO2 "${MKL_ROOT}/lib/libmkl_rt.so.2")

if(EXISTS "${MKL_RT_SO}")
  set(MKL_RT_LIBRARY "${MKL_RT_SO}")
elseif(EXISTS "${MKL_RT_SO2}")
  set(MKL_RT_LIBRARY "${MKL_RT_SO2}")
else()
  message(FATAL_ERROR "Could not find MKL runtime. Expected ${MKL_RT_SO} or ${MKL_RT_SO2}.")
endif()

message(STATUS "Using MKL runtime: ${MKL_RT_LIBRARY}")

add_executable(blocked_trsv_benchmark
  src/main.cpp
)

target_include_directories(blocked_trsv_benchmark PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CBLAS_INCLUDE_DIR}
)

target_link_libraries(blocked_trsv_benchmark PRIVATE
  ${MKL_RT_LIBRARY}
  Threads::Threads
  m
  dl
)

set_target_properties(blocked_trsv_benchmark PROPERTIES
  BUILD_RPATH "${MKL_ROOT}/lib"
  INSTALL_RPATH "${MKL_ROOT}/lib"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(blocked_trsv_benchmark PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    -Wall
    -Wextra
    -Wpedantic
  )
endif()
