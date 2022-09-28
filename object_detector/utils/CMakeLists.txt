project (buffs)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 20)
# configure OpenCV
find_package(OpenCV REQUIRED)
find_package(X11 REQUIRED)

add_executable(sample_video ../sample_video.cpp ../detect.cpp)
target_link_libraries(sample_video ${OpenCV_LIBS} ${X11_LIBRARIES} ${X11_Xtst_LIB})