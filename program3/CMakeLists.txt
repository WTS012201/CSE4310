project (program3)
cmake_minimum_required(VERSION 2.8)

# configure OpenCV
find_package(OpenCV REQUIRED)
find_package(Threads)

# create create individual projects
add_executable(program3 program3.cpp)
target_link_libraries(program3 ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})