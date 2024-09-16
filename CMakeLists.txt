cmake_minimum_required(VERSION 3.20)
project(rrt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
enable_testing()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(rrt ${CMAKE_SOURCE_DIR}/src/rrt.cpp)

add_executable(test_kd_tree ${CMAKE_SOURCE_DIR}/test/test_kd_tree.cpp)

add_executable(bench_kd_tree ${CMAKE_SOURCE_DIR}/test/bench_kd_tree.cpp)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.0
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark's tests" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Disable gtest installation" FORCE)

FetchContent_MakeAvailable(benchmark googletest)
target_link_libraries(bench_kd_tree PRIVATE benchmark::benchmark pthread)

find_path(EIGEN_INCLUDE_DIR NAMES signature_of_eigen3_matrix_library
  HINTS  ENV EIGEN3_INC_DIR
         ENV EIGEN3_DIR
  PATHS  Eigen/Core
         /usr/local/include
         /usr/include
         /opt/local/include
  PATH_SUFFIXES include eigen3 eigen2 eigen
  DOC "Directory containing the Eigen3 header files"
)
include_directories(${EIGEN_INCLUDE_DIR})

find_package(libnabo)
target_link_libraries(bench_kd_tree PUBLIC ${libnabo_LIBRARIES})
target_include_directories(bench_kd_tree PUBLIC "libnabo")

