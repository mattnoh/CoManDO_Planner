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

# ── Core ROS2 dependencies ────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(crazyflie_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)

# ── Include ALIPDDP solver as a standalone sibling ────────────────────────────
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ALIPDDP-main ${CMAKE_CURRENT_BINARY_DIR}/alipddp)

add_library(planner_core INTERFACE)
target_include_directories(planner_core INTERFACE
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/config
)
target_link_libraries(planner_core INTERFACE
  alipddp_library
)

option(COMANDO_ENABLE_MOCAP "Enable mocap4r2 message support" ON)
find_package(mocap4r2_msgs QUIET)
if(COMANDO_ENABLE_MOCAP AND mocap4r2_msgs_FOUND)
  message(STATUS "mocap4r2_msgs FOUND — mocap target mode enabled")
elseif(COMANDO_ENABLE_MOCAP)
  message(WARNING "mocap4r2_msgs NOT FOUND — building without mocap target mode")
else()
  message(STATUS "mocap target mode disabled by COMANDO_ENABLE_MOCAP=OFF")
endif()
find_package(mavros_msgs QUIET)
if(mavros_msgs_FOUND)
  message(STATUS "mavros_msgs FOUND — MAVROS platform support enabled")
else()
  message(STATUS "mavros_msgs NOT FOUND — building without MAVROS support")
endif()

# ── Shared sources (no main) ──────────────────────────────────────────────────
# quadrotor_mpc.cpp is the only shared translation unit.
set(SHARED_SOURCES
  src/quadrotor_mpc.cpp
)

# ── Helper macro: apply common settings to any target ────────────────────────
macro(configure_target TARGET_NAME)

  target_include_directories(${TARGET_NAME} PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    include
    config
  )

  if(COMANDO_ENABLE_MOCAP AND mocap4r2_msgs_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE HAS_MOCAP4R2_MSGS)
  endif()
  if(mavros_msgs_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE HAS_MAVROS_MSGS)
  endif()

  target_link_libraries(${TARGET_NAME}
    planner_core
  )

  set(_AMENT_DEPS
    rclcpp
    tf2_ros
    geometry_msgs
    nav_msgs
    trajectory_msgs
    visualization_msgs
    crazyflie_interfaces
  )
  if(COMANDO_ENABLE_MOCAP AND mocap4r2_msgs_FOUND)
    list(APPEND _AMENT_DEPS mocap4r2_msgs)
  endif()
  if(mavros_msgs_FOUND)
    list(APPEND _AMENT_DEPS mavros_msgs)
  endif()
  ament_target_dependencies(${TARGET_NAME} ${_AMENT_DEPS})

endmacro()

# ── Executable 1: normal receding-horizon planner ────────────────────────────
add_executable(comando_planner
  src/planner_node.cpp
  ${SHARED_SOURCES}
)
configure_target(comando_planner)

# ── Executable 3: target_publisher
add_executable(target_publisher src/target_publisher.cpp)
configure_target(target_publisher)

# ── Standalone ROS-free validation target ────────────────────────────────────
add_executable(test_poly_ocp
  test/test_poly_ocp_standalone.cpp
)
target_include_directories(test_poly_ocp PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  include
  config
)
target_link_libraries(test_poly_ocp
  planner_core
)

add_executable(test_trajectory_replayer
  test/test_trajectory_replayer.cpp
)
target_include_directories(test_trajectory_replayer PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  include
)
target_link_libraries(test_trajectory_replayer
  planner_core
)

add_executable(test_target_frame_warm_start
  test/test_target_frame_warm_start.cpp
)
target_include_directories(test_target_frame_warm_start PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  include
  config
)
target_link_libraries(test_target_frame_warm_start
  planner_core
)

add_executable(test_planner_core
  test/test_planner_core.cpp
  ${SHARED_SOURCES}
)
target_include_directories(test_planner_core PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  include
  config
)
target_link_libraries(test_planner_core
  planner_core
)

add_executable(test_stateswitch_predictor
  test/test_stateswitch_predictor.cpp
)
target_include_directories(test_stateswitch_predictor PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  include
  config
)
target_link_libraries(test_stateswitch_predictor
  planner_core
)

add_executable(test_stc_landing
  test/test_stc_landing.cpp
  ${SHARED_SOURCES}
)
target_include_directories(test_stc_landing PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  include
  config
)
target_link_libraries(test_stc_landing
  planner_core
)

add_executable(test_stc_landing_noaug
  test/test_stc_landing_noaug.cpp
  ${SHARED_SOURCES}
)
target_include_directories(test_stc_landing_noaug PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  include
  config
)
target_link_libraries(test_stc_landing_noaug
  planner_core
)

if(BUILD_TESTING)
  add_test(NAME test_trajectory_replayer COMMAND test_trajectory_replayer)
  add_test(NAME test_planner_core COMMAND test_planner_core)
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS
    comando_planner
    target_publisher
    test_poly_ocp
    test_trajectory_replayer
    test_target_frame_warm_start
    test_planner_core
    test_stateswitch_predictor
    test_stc_landing
    test_stc_landing_noaug
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

install(PROGRAMS
    scripts/test_bodyrate_cmd.py
    logs/plot_bag_circle.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
