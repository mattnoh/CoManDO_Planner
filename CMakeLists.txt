cmake_minimum_required(VERSION 3.10)
project(comando_planner)

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
find_package(crazyflie_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)

# ── Include ALIPDDP solver as a standalone sibling ────────────────────────────
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ALIPDDP-main ${CMAKE_CURRENT_BINARY_DIR}/alipddp)

# ── Optional: PX4 messages ────────────────────────────────────────────────────
option(COMANDO_ENABLE_QUALISYS "Enable Qualisys/mocap4r2 message support" ON)
find_package(mocap4r2_msgs QUIET)
find_package(px4_msgs QUIET)
if(COMANDO_ENABLE_QUALISYS AND mocap4r2_msgs_FOUND)
  message(STATUS "mocap4r2_msgs FOUND — Qualisys target mode enabled")
elseif(COMANDO_ENABLE_QUALISYS)
  message(WARNING "mocap4r2_msgs NOT FOUND — building without Qualisys target mode")
else()
  message(STATUS "Qualisys target mode disabled by COMANDO_ENABLE_QUALISYS=OFF")
endif()
if(px4_msgs_FOUND)
  message(STATUS "px4_msgs FOUND — PX4 platform support enabled")
else()
  message(STATUS "px4_msgs NOT FOUND — building without PX4 support")
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

  if(px4_msgs_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE HAS_PX4_MSGS)
  endif()
  if(COMANDO_ENABLE_QUALISYS AND mocap4r2_msgs_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE HAS_MOCAP4R2_MSGS)
  endif()

  target_link_libraries(${TARGET_NAME}
    alipddp_library
  )

  set(_AMENT_DEPS
    rclcpp
    tf2_ros
    geometry_msgs
    nav_msgs
    trajectory_msgs
    crazyflie_interfaces
  )
  if(px4_msgs_FOUND)
    list(APPEND _AMENT_DEPS px4_msgs)
  endif()
  if(COMANDO_ENABLE_QUALISYS AND mocap4r2_msgs_FOUND)
    list(APPEND _AMENT_DEPS mocap4r2_msgs)
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
  alipddp_library
)



# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS
    comando_planner
    target_publisher
    test_poly_ocp
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
