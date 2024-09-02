#
# This CMake file is adapted for building Mojo projects.
#

# Set the appropriate Mojo build directory
set(MOJO_BINDINGS_DIR "bindings/mojo_bindings")

message("-- Configuring Mojo bindings in: ${MOJO_BINDINGS_DIR}")

# Add the Mojo bindings directory
add_subdirectory(${MOJO_BINDINGS_DIR})

#
# Install Mojo examples
#
file(GLOB mojoExamples examples/*.mojo)

foreach(mojoExample ${mojoExamples})
	message("-- Copying ${mojoExample}")
	file(COPY ${mojoExample} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	install(FILES "${mojoExample}" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endforeach()

#
# Replicate legacy Mojo examples (these will be deprecated)
#
function(copy_install file_in file_out)
    message("-- Copying ${file_in} -> ${file_out}")
    configure_file(${file_in} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${file_out} COPYONLY)
    install(FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${file_out}" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endfunction()
