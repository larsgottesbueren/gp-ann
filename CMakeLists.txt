cmake_minimum_required(VERSION 3.10)
project(GP-ANNS)

option(KAMINPAR_64BIT_WEIGHTS "Enable 64 bit weights in KaMinPar" ON)
option(KAMINPAR_64BIT_EDGE_IDS "Enable 64 bit edge IDs in KaMinPar" ON)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Wunused")
set(CMAKE_CXX_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mtune=native -march=native -ffast-math")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lm")
#set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}")

add_subdirectory(external/parlaylib)
add_subdirectory(external/KaMinPar)
add_subdirectory(external/hnswlib)

include_directories("src")

add_executable(BuildKNNGraph build_knn_graph.cpp)
add_executable(RunQueries run_queries.cpp)
add_executable(Convert convert.cpp)
add_executable(Partition partition.cpp)
add_executable(QueryAttribution run_query_attribution.cpp)

target_link_libraries(BuildKNNGraph PUBLIC parlay)
target_link_libraries(RunQueries PUBLIC parlay)
target_link_libraries(Convert PUBLIC parlay)
target_link_libraries(Partition PUBLIC parlay kaminpar_shm)
target_link_libraries(QueryAttribution PUBLIC parlay)

OPTION(MIPS_DISTANCE "Use MIPS distance instead of L2" OFF)
if(MIPS_DISTANCE)
	target_compile_definitions(BuildKNNGraph PUBLIC MIPS_DISTANCE)
	target_compile_definitions(Partition PUBLIC MIPS_DISTANCE)
	target_compile_definitions(RunQueries PUBLIC MIPS_DISTANCE)
	target_compile_definitions(QueryAttribution PUBLIC MIPS_DISTANCE)
endif(MIPS_DISTANCE)

find_library(TCMALLOC_LIB tcmalloc)
if (TCMALLOC_LIB)
	target_link_libraries(Partition PUBLIC ${TCMALLOC_LIB})
	message(STATUS "Using tcmalloc: ${TCMALLOC_LIB}")
else ()
	message(STATUS "tcmalloc enabled but unavailable on this system")
endif ()
