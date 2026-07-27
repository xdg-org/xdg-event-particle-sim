cmake_minimum_required(VERSION 3.18)

project(event_sim_cuda_thrust_sort
  DESCRIPTION "Sort XDG particle event queues on NVIDIA devices with Thrust"
  LANGUAGES CXX CUDA
)

add_library(event_sim_thrust_sort SHARED src/particle_event_thrust_sort.cu)
target_include_directories(event_sim_thrust_sort PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

set_target_properties(event_sim_thrust_sort PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED YES
  CUDA_EXTENSIONS NO
  CUDA_SEPARABLE_COMPILATION ON
)

if(CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(event_sim_thrust_sort PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  set_target_properties(event_sim_thrust_sort PROPERTIES CUDA_ARCHITECTURES native)
endif()
