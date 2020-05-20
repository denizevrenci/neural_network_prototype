cmake_minimum_required (VERSION 3.2)

if(dlib_included)
    return()
endif (dlib_included)
set(dlib_included TRUE)

include(DownloadProject)

download_project(
	PROJ dlib_proj
	GIT_REPOSITORY https://github.com/davisking/dlib.git
	GIT_TAG d71497d4669c73d09899bb4b7e49864625d77772
	UPDATE_DISCONNECTED 1
	QUIET
)

set(DLIB_USE_BLAS ON CACHE "" INTERNAL)
set(DLIB_USE_LAPACK ON CACHE "" INTERNAL)
set(DLIB_NO_GUI_SUPPORT ON CACHE "" INTERNAL)
add_subdirectory(${dlib_proj_SOURCE_DIR})
