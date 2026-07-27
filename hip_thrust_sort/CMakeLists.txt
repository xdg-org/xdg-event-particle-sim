cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED HIP_PATH)
  if(DEFINED ENV{HIP_PATH})
    set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to the HIP installation")
  else()
    set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to the HIP installation")
  endif()
endif()

list(PREPEND CMAKE_MODULE_PATH "${HIP_PATH}/cmake")

project(event_sim_hip_thrust_sort
  DESCRIPTION "Sort XDG particle event queues on AMD devices with Thrust"
  LANGUAGES CXX HIP
)

add_library(event_sim_thrust_sort SHARED src/particle_event_thrust_sort.hip)
target_include_directories(event_sim_thrust_sort PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

set_target_properties(event_sim_thrust_sort PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
  HIP_STANDARD 17
  HIP_STANDARD_REQUIRED YES
  HIP_EXTENSIONS NO
)
