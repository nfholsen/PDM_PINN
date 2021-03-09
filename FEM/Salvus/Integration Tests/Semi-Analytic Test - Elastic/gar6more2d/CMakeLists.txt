cmake_minimum_required (VERSION 2.6)

#change default flags
SET(CMAKE_USER_MAKE_RULES_OVERRIDE
"${CMAKE_SOURCE_DIR}/cmake/gar6more_flags.cmake")

project (Gar6more2d)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
enable_language (Fortran)


include_directories(build/mod) 
include_directories(includefic) 
add_subdirectory(mod)
add_subdirectory(lib/libacousacous)
add_subdirectory(lib/libacouselasto)
add_subdirectory(lib/libelastoelasto)
add_subdirectory(lib/libgeneral)
add_subdirectory(lib/libporoporo)
add_subdirectory(lib/libacousporo)
# make sure that the default is a RELEASE
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are: None Debug Release."
      FORCE)
endif (NOT CMAKE_BUILD_TYPE)

# default installation
get_filename_component (default_prefix ".." ABSOLUTE)
set (CMAKE_INSTALL_PREFIX ${default_prefix} CACHE STRING
      "Choose the installation directory; by default it installs in the Gar6more2D directory.")

add_dependencies(acousacous mod)
add_dependencies(acouselasto mod)
add_dependencies(acousporo mod)
add_dependencies(elastoelasto mod)
add_dependencies(poroporo mod)
add_dependencies(gene mod)

# build executables
set (EXECUTABLES Gar6more2D.out)

add_executable (Gar6more2D.out  lib/bin/Gar6more2D.F90)
target_link_libraries (Gar6more2D.out mod  acousacous   acouselasto acousporo elastoelasto poroporo gene ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}) 

# install executables and scripts
install (TARGETS ${EXECUTABLES} 
         RUNTIME DESTINATION "bin")
