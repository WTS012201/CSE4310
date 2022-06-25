project (program1)
cmake_minimum_required(VERSION 2.8)

# configure OpenCV
find_package(OpenCV REQUIRED)

# create create individual projects
add_executable(program1 program1.cpp)
target_link_libraries(program1 ${OpenCV_LIBS})