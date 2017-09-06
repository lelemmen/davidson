# tests-level CMakeLists

add_executable(davidson_test davidson_test.cpp)


# Include this project's headers
target_include_directories(davidson_test PRIVATE ../include)

# Include the Boost headers (necessary for testing)
target_include_directories(davidson_test PRIVATE ${Boost_INCLUDE_DIRS})

# Include the Eigen headers
target_include_directories(davidson_test PRIVATE /opt/local/include/eigen3)

# Link the test executable with the project library
target_link_libraries(davidson_test PRIVATE davidson)


add_test(NAME davidson_test COMMAND davidson_test)
