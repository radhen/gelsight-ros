cmake_minimum_required(VERSION 3.0.2)
project(gelsight-ros)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  rospy 
  roscpp
  image_transport
  cv_bridge
  std_msgs
  geometry_msgs
  message_generation
)

find_package(OpenCV REQUIRED)

catkin_python_setup()

add_message_files(
  FILES
  MarkerFlow.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS roscpp
                 rospy
                 image_transport
                 cv_bridge		 
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(gelsight_stream_node src/gelsight_stream.cpp)

target_link_libraries(gelsight_stream_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
