cmake_minimum_required(VERSION 3.10)
project(comando_planner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Core ROS2 dependencies ────────────────────────────────────────────────────
find_package(ament_cmake              REQUIRED)
find_package(rclcpp                   REQUIRED)
find_package(tf2_ros                  REQUIRED)
find_package(geometry_msgs            REQUIRED)
find_package(nav_msgs                 REQUIRED)
find_package(std_msgs                 REQUIRED)
find_package(crazyflie_interfaces     REQUIRED)
find_package(Eigen3                   REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ── Optional: PX4 messages ────────────────────────────────────────────────────
find_package(px4_msgs QUIET)
if(px4_msgs_FOUND)
  message(STATUS "px4_msgs FOUND — PX4 bridge enabled")
else()
  message(STATUS "px4_msgs NOT FOUND — building without PX4 bridge")
endif()

# ── Optional: Acados solver ───────────────────────────────────────────────────
set(ACADOS_DIR "$ENV{HOME}/acados")
set(ACADOS_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/acados_generated/c_generated_code")

if(EXISTS "${ACADOS_DIR}/lib/libacados.so" AND
   EXISTS "${ACADOS_GENERATED_DIR}/acados_solver_quadrotor.h")
  set(ACADOS_FOUND TRUE)
  message(STATUS "Acados FOUND — acados solver enabled")
else()
  set(ACADOS_FOUND FALSE)
  message(STATUS "Acados NOT FOUND — building without acados solver")
endif()

# ══════════════════════════════════════════════════════════════════════════════
# Custom ROS2 message
# ══════════════════════════════════════════════════════════════════════════════
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MpcCommand.msg"
  DEPENDENCIES std_msgs
)

# ── Helper: get typesupport target for linking against generated message ──────
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ══════════════════════════════════════════════════════════════════════════════
# Shared sources
# ══════════════════════════════════════════════════════════════════════════════
set(SOLVER_SOURCES
  src/solver/alipddp_solver.cpp
)
if(ACADOS_FOUND)
  list(APPEND SOLVER_SOURCES src/solver/acados_solver.cpp)
endif()

# ══════════════════════════════════════════════════════════════════════════════
# Helper macro — apply common settings to any target
# ══════════════════════════════════════════════════════════════════════════════
macro(configure_target TARGET_NAME)

  target_include_directories(${TARGET_NAME} PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    include                                                 # include/solver/, include/ocp/, etc.
    ${CMAKE_SOURCE_DIR}/../ALIPDDP-main/alipddp
    ${CMAKE_SOURCE_DIR}/../ALIPDDP-main/problem_maker
    ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp       # generated MpcCommand header
  )

  if(px4_msgs_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE HAS_PX4_MSGS)
  endif()

  if(ACADOS_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE HAS_ACADOS)
    target_include_directories(${TARGET_NAME} PRIVATE
      ${ACADOS_DIR}/include
      ${ACADOS_DIR}/include/acados
      ${ACADOS_DIR}/include/blasfeo/include
      ${ACADOS_DIR}/include/hpipm/include
      ${ACADOS_GENERATED_DIR}
    )
  endif()

  # Link ALIPDDP static library
  target_link_libraries(${TARGET_NAME}
    ${CMAKE_SOURCE_DIR}/../ALIPDDP-main/build/libalipddp_library.a
    ${cpp_typesupport_target}
  )

  if(ACADOS_FOUND)
    target_link_libraries(${TARGET_NAME}
      ${ACADOS_GENERATED_DIR}/libacados_ocp_solver_quadrotor.so
      ${ACADOS_DIR}/lib/libacados.so
      ${ACADOS_DIR}/lib/libhpipm.so
      ${ACADOS_DIR}/lib/libblasfeo.so
    )
  endif()

  set(_AMENT_DEPS
    rclcpp
    tf2_ros
    geometry_msgs
    nav_msgs
    std_msgs
    crazyflie_interfaces
  )
  if(px4_msgs_FOUND)
    list(APPEND _AMENT_DEPS px4_msgs)
  endif()
  ament_target_dependencies(${TARGET_NAME} ${_AMENT_DEPS})

  # Ensure the message interface is generated before compiling this target
  add_dependencies(${TARGET_NAME} ${PROJECT_NAME})

endmacro()

# ══════════════════════════════════════════════════════════════════════════════
# Executable 1: CoManDO Planner (solver node — platform-agnostic)
# ══════════════════════════════════════════════════════════════════════════════
add_executable(comando_planner_node
  src/solver/comando_planner_node.cpp
  ${SOLVER_SOURCES}
)
configure_target(comando_planner_node)

# ══════════════════════════════════════════════════════════════════════════════
# Executable 2: Crazyflie bridge
# ══════════════════════════════════════════════════════════════════════════════
add_executable(crazyflie_bridge
  src/bridges/crazyflie_bridge.cpp
)
configure_target(crazyflie_bridge)

# ══════════════════════════════════════════════════════════════════════════════
# Executable 3: PX4 bridge (always compiled; guards itself with #ifdef)
# ══════════════════════════════════════════════════════════════════════════════
add_executable(px4_bridge
  src/bridges/px4_bridge.cpp
)
configure_target(px4_bridge)

# ══════════════════════════════════════════════════════════════════════════════
# Executable 4: Simulation bridge (no Gazebo, no hardware, no lower controller)
#   Integrates Quad6DOF dynamics in-loop with optional Gaussian noise.
#   Modes: none | moderate | large  (ROS param noise_mode)
# ══════════════════════════════════════════════════════════════════════════════
add_executable(sim_bridge
  src/bridges/sim_bridge.cpp
)
configure_target(sim_bridge)

# ══════════════════════════════════════════════════════════════════════════════
# Install
# ══════════════════════════════════════════════════════════════════════════════
install(TARGETS
    comando_planner_node
    crazyflie_bridge
    px4_bridge
    sim_bridge
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(ACADOS_FOUND)
  install(FILES ${ACADOS_GENERATED_DIR}/libacados_ocp_solver_quadrotor.so
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

ament_package()