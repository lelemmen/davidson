# Top-level CMakeLists

CMAKE_MINIMUM_REQUIRED(VERSION 3.7 FATAL_ERROR)
project(davidson LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)



# Source files for the davidson routine library can be found in src
add_subdirectory(src)

# Set up testing
enable_testing()
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
add_subdirectory(tests)
