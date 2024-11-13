PROJECT(edge_detection C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

find_package(OpenCV )
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.4 not found.")
   endif()
   find_package(realsense2)
   MESSAGE("OPENCV VERSION:")
   MESSAGE(${OpenCV_VERSION})

add_executable(edge_demo ${PROJECT_SOURCE_DIR}/edge_demo.cc)
target_link_libraries(edge_demo  ${OpenCV_LIBS} ${DEPENDENCIES} ${realsense2_LIBRARY}) 











































































#//code by tomLee, 2021/11/16
