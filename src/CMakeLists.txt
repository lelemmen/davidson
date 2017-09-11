# src-level CMakeLists

# Based on the given source files, we want to make a library 'davidson'
add_library(davidson
        davidson.cpp)


# We must include our own project's headers
target_include_directories(davidson PRIVATE ../include)

# We must include the eigen headers
target_include_directories(davidson PRIVATE /opt/local/include/eigen3)