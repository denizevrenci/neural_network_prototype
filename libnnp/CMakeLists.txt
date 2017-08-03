set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(libnnp INTERFACE)

target_link_libraries(libnnp
	INTERFACE dlib
)

target_include_directories(libnnp
	INTERFACE ${INCLUDE_DIR}
)
