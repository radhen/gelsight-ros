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
  DIRECTORY msg
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

catkin_install_python(PROGRAMS scripts/gelsight.py scripts/label_data.py scripts/record.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

pybind_add_module(find_marker MODULE src/tracking_class.cpp)

install(TARGETS find_marker
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})