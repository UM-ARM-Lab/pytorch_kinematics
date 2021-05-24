cmake_minimum_required(VERSION 2.8.3)
project(tensorflow_kinematics)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package(CATKIN_DEPENDS rospy)

catkin_python_setup()
