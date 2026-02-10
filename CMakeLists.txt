cmake_minimum_required(VERSION 3.10)
project(comando_planner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(crazyflie_interfaces REQUIRED)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Create executable
add_executable(comando_planner
  src/planner_node.cpp
  src/quadrotor_mpc.cpp
)

# Set include directories
target_include_directories(comando_planner PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  include
  ${CMAKE_SOURCE_DIR}/../ALIPDDP-main/alipddp
  ${CMAKE_SOURCE_DIR}/../ALIPDDP-main/problem_maker
)

# Link ALIPDDP static library
target_link_libraries(comando_planner
  ${CMAKE_SOURCE_DIR}/../ALIPDDP-main/build/libalipddp_library.a
)

# Link ROS2 dependencies
ament_target_dependencies(comando_planner
  rclcpp
  tf2_ros
  geometry_msgs
  nav_msgs
  crazyflie_interfaces
)

# Install executable
install(TARGETS comando_planner
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install Python logger script
install(PROGRAMS launch/logger.py
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()