cmake_minimum_required(VERSION 3.10)
project(GP-ANNS)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wpedantic -Wunused")
set(CMAKE_CXX_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g -mtune=native -march=native -ffast-math")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lm")
#set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}")

add_subdirectory(external/parlaylib)

include_directories("src")

add_executable(BuildKNNGraph build_knn_graph.cpp)
add_executable(RunQueries run_queries.cpp)
add_executable(Convert convert.cpp)

target_link_libraries(BuildKNNGraph PUBLIC parlay)
target_link_libraries(RunQueries PUBLIC parlay)
