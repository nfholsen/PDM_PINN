set ( src_files "" )

    foreach ( ext IN ITEMS "f90" "F90" )
        file ( GLOB_RECURSE _files "*.${ext}" )
        set ( src_files ${src_files} ${_files} )
    endforeach ()

    message ( STATUS "Found: ${src_files}" )

add_library (mod STATIC  ${src_files}) 
install(TARGETS mod DESTINATION lib)
