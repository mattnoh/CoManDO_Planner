cmake_minimum_required(VERSION 3.10)
project(comando_planner)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Core ROS1 dependencies ────────────────────────────────────────────────────
find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf2_ros
  geometry_msgs
  nav_msgs
  trajectory_msgs
  mavros_msgs
)

find_package(Eigen3 REQUIRED)

# ── Include ALIPDDP solver as a sibling ──────────────────────────────────────
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ALIPDDP-main ${CMAKE_CURRENT_BINARY_DIR}/alipddp)

# ── catkin package declaration ────────────────────────────────────────────────
catkin_package(
  INCLUDE_DIRS include config
  CATKIN_DEPENDS roscpp geometry_msgs nav_msgs trajectory_msgs mavros_msgs
)

# ── Interface-only library for ROS-free planner core ─────────────────────────
add_library(planner_core INTERFACE)
target_include_directories(planner_core INTERFACE
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/config
)
target_link_libraries(planner_core INTERFACE
  alipddp_library
)

# ── Helper macro ──────────────────────────────────────────────────────────────
macro(configure_target TARGET_NAME)
  target_include_directories(${TARGET_NAME} PRIVATE
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    include
    config
  )
  target_link_libraries(${TARGET_NAME}
    planner_core
    ${catkin_LIBRARIES}
  )
endmacro()

# ── Executables ───────────────────────────────────────────────────────────────
set(SHARED_SOURCES src/quadrotor_mpc.cpp)

add_executable(comando_planner
  src/planner_node.cpp
  ${SHARED_SOURCES}
)
configure_target(comando_planner)

add_executable(target_publisher src/target_publisher.cpp)
configure_target(target_publisher)

# ── ROS-free test executables ─────────────────────────────────────────────────
add_executable(test_poly_ocp test/test_poly_ocp_standalone.cpp)
target_include_directories(test_poly_ocp PRIVATE ${EIGEN3_INCLUDE_DIR} include config)
target_link_libraries(test_poly_ocp planner_core)

add_executable(test_trajectory_replayer test/test_trajectory_replayer.cpp)
target_include_directories(test_trajectory_replayer PRIVATE ${EIGEN3_INCLUDE_DIR} include)
target_link_libraries(test_trajectory_replayer planner_core)

add_executable(test_target_frame_warm_start test/test_target_frame_warm_start.cpp)
target_include_directories(test_target_frame_warm_start PRIVATE ${EIGEN3_INCLUDE_DIR} include config)
target_link_libraries(test_target_frame_warm_start planner_core)

add_executable(test_planner_core test/test_planner_core.cpp ${SHARED_SOURCES})
target_include_directories(test_planner_core PRIVATE ${EIGEN3_INCLUDE_DIR} include config)
target_link_libraries(test_planner_core planner_core)

add_executable(test_stateswitch_predictor test/test_stateswitch_predictor.cpp)
target_include_directories(test_stateswitch_predictor PRIVATE ${EIGEN3_INCLUDE_DIR} include config)
target_link_libraries(test_stateswitch_predictor planner_core)

add_executable(test_stc_landing test/test_stc_landing.cpp ${SHARED_SOURCES})
target_include_directories(test_stc_landing PRIVATE ${EIGEN3_INCLUDE_DIR} include config)
target_link_libraries(test_stc_landing planner_core)

add_executable(test_stc_landing_noaug test/test_stc_landing_noaug.cpp ${SHARED_SOURCES})
target_include_directories(test_stc_landing_noaug PRIVATE ${EIGEN3_INCLUDE_DIR} include config)
target_link_libraries(test_stc_landing_noaug planner_core)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS
    comando_planner
    target_publisher
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
