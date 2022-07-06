cmake_minimum_required(VERSION 3.0.2)
project(gelsight_ros)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  rospy 
  roscpp
  std_msgs
  geometry_msgs
  message_generation
  pybind11_catkin 
)

find_package(OpenCV REQUIRED)

catkin_python_setup()

add_message_files(
  FILES
  GelsightMarkers.msg
  GelsightMarkersStamped.msg
  GelsightFlowStamped.msg
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
                 pybind11_catkin
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include/
)

catkin_install_python(PROGRAMS scripts/call_icp.py scripts/gelsight.py scripts/vis_flow.py scripts/record.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

pybind11_add_module(find_marker MODULE src/tracking_class.cpp)

catkin_install_python(PROGRAMS scripts/gelsight.py scripts/vis_flow.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(TARGETS find_marker
  LIBRARY DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)