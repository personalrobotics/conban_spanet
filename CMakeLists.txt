cmake_minimum_required(VERSION 2.8.3)
project(conban_spanet)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
)

add_service_files(
  FILES
  GetAction.srv
  PublishLoss.srv
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package()
