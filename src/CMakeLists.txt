set(Sources
		dist.cpp
		kmeans.cpp
)

foreach(target IN LISTS TARGETS)
	message(STATUS "Adding sources to target : ${target}")
	message(STATUS "Sources:  ${Sources}")
	target_sources(${target} PRIVATE ${Sources})
endforeach()
