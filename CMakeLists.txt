cmake_minimum_required(VERSION 3.16)

project(memory_accessor_ch4_minimal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(USE_MKL "Link against MKL runtime libmkl_rt" OFF)
set(MKL_ROOT "/opt/conda" CACHE PATH "MKL root directory")

add_executable(blocked_trsv_benchmark
  src/main.cpp
)

target_include_directories(blocked_trsv_benchmark PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  "${MKL_ROOT}/include"
)

if(USE_MKL)
  target_compile_definitions(blocked_trsv_benchmark PRIVATE USE_MKL)

  find_library(MKL_RT_LIBRARY
    NAMES mkl_rt libmkl_rt.so.2
    PATHS "${MKL_ROOT}/lib"
    NO_DEFAULT_PATH
  )

  if(NOT MKL_RT_LIBRARY AND EXISTS "${MKL_ROOT}/lib/libmkl_rt.so.2")
    set(MKL_RT_LIBRARY "${MKL_ROOT}/lib/libmkl_rt.so.2")
  endif()

  if(NOT MKL_RT_LIBRARY)
    message(FATAL_ERROR "Could not find libmkl_rt under ${MKL_ROOT}/lib. Pass -DMKL_ROOT=/path/to/conda_or_mkl.")
  endif()

  target_link_directories(blocked_trsv_benchmark PRIVATE
    "${MKL_ROOT}/lib"
  )

  target_link_libraries(blocked_trsv_benchmark PRIVATE
    ${MKL_RT_LIBRARY}
    pthread
    m
    dl
  )

  set_target_properties(blocked_trsv_benchmark PROPERTIES
    BUILD_RPATH "${MKL_ROOT}/lib"
    INSTALL_RPATH "${MKL_ROOT}/lib"
  )
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(blocked_trsv_benchmark PRIVATE ${BLAS_LIBRARIES})
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(blocked_trsv_benchmark PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    -Wall
    -Wextra
    -Wpedantic
  )
endif()
