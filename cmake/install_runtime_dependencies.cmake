# Called after installing the Python extension on Linux. Scan the build artifact
# so its build RPATH can locate dependencies in Conan's cache and the toolkit.
file(GET_RUNTIME_DEPENDENCIES
  MODULES "${_fd_module}"
  RESOLVED_DEPENDENCIES_VAR _fd_runtime_libraries
  PRE_EXCLUDE_REGEXES
    "^ld-linux.*" "^lib(c|m|dl|rt|pthread|resolv|util)\\.so.*"
    "^libstdc\\+\\+\\.so.*" "^libgcc_s\\.so.*" "^libcuda\\.so.*"
)

foreach(_fd_library IN LISTS _fd_runtime_libraries)
  # The resolved path uses the loader's name, e.g. libcublas.so.13. Copy the
  # actual file under that name without installing its entire symlink chain:
  # wheel archives would otherwise contain multiple copies of the same library.
  get_filename_component(_fd_library_name "${_fd_library}" NAME)
  file(REAL_PATH "${_fd_library}" _fd_library_real)
  file(INSTALL "${_fd_library_real}"
    DESTINATION "${_fd_runtime_destination}"
    TYPE FILE
    RENAME "${_fd_library_name}"
  )
endforeach()
