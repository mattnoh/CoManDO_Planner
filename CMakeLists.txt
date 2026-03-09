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
find_package(crazyflie_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)

# ── Optional: PX4 messages ────────────────────────────────────────────────────
find_package(px4_msgs QUIET)
if(px4_msgs_FOUND)
  message(STATUS "px4_msgs FOUND — PX4 platform support enabled")
else()
  message(STATUS "px4_msgs NOT FOUND — building without PX4 support")
endif()

# ── Optional: Acados solver ───────────────────────────────────────────────────
set(ACADOS_DIR "$ENV{HOME}/acados")
set(ACADOS_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/acados_generated/c_generated_code")

if(EXISTS "${ACADOS_DIR}/lib/libacados.so" AND EXISTS "${ACADOS_GENERATED_DIR}/acados_solver_quadrotor.h")
  set(ACADOS_FOUND TRUE)
  message(STATUS "Acados FOUND — acados solver support enabled")
  message(STATUS "  ACADOS_DIR: ${ACADOS_DIR}")
  message(STATUS "  Generated code: ${ACADOS_GENERATED_DIR}")
else()
  set(ACADOS_FOUND FALSE)
  message(STATUS "Acados NOT FOUND — building without acados solver")
endif()

# ── Shared sources (no main) ──────────────────────────────────────────────────
# quadrotor_mpc.cpp is the only shared translation unit.
set(SHARED_SOURCES
  src/quadrotor_mpc.cpp
)

if(ACADOS_FOUND)
  list(APPEND SHARED_SOURCES src/acados_solver.cpp)
endif()

# ── Helper macro: apply common settings to any target ────────────────────────
macro(configure_target TARGET_NAME)

  target_include_directories(${TARGET_NAME} PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    include
    ${CMAKE_SOURCE_DIR}/../ALIPDDP-main/alipddp
    ${CMAKE_SOURCE_DIR}/../ALIPDDP-main/problem_maker
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

  target_link_libraries(${TARGET_NAME}
    ${CMAKE_SOURCE_DIR}/../ALIPDDP-main/build/libalipddp_library.a
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
    crazyflie_interfaces
  )
  if(px4_msgs_FOUND)
    list(APPEND _AMENT_DEPS px4_msgs)
  endif()
  ament_target_dependencies(${TARGET_NAME} ${_AMENT_DEPS})

endmacro()

# ── Executable 1: normal receding-horizon planner ────────────────────────────
add_executable(comando_planner
  src/planner_node.cpp
  ${SHARED_SOURCES}
)
configure_target(comando_planner)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS
    comando_planner
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# install(PROGRAMS launch/logger.py
#   DESTINATION share/${PROJECT_NAME}/launch
# )

if(ACADOS_FOUND)
  install(FILES ${ACADOS_GENERATED_DIR}/libacados_ocp_solver_quadrotor.so
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

ament_package()