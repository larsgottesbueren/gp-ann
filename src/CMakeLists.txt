set(Sources
		defs.cpp
		dist.cpp
		kmeans.cpp
		points_io.cpp
		metis_io.cpp
)

foreach(target IN LISTS TARGETS)
	message(STATUS "Adding sources to target : ${target}")
	message(STATUS "Sources:  ${Sources}")
	target_sources(${target} PRIVATE ${Sources})
endforeach()

target_sources(Partition PRIVATE partitioning.cpp)

target_sources(QueryAttribution PRIVATE routes.cpp shard_searches.cpp route_search_combinations.cpp)
