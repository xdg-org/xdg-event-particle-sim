cmake_minimum_required(VERSION 3.18)

project(xdg_event_particle_sim VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type" FORCE)
endif()

option(EVENT_SIM_ENABLE_CUDA_THRUST_SORT "Enable CUDA Thrust event-queue sorting" OFF)
option(EVENT_SIM_ENABLE_HIP_THRUST_SORT "Enable HIP Thrust event-queue sorting" OFF)

if(EVENT_SIM_ENABLE_CUDA_THRUST_SORT AND EVENT_SIM_ENABLE_HIP_THRUST_SORT)
  message(FATAL_ERROR "CUDA and HIP Thrust sorting cannot both be enabled")
endif()

find_package(XDG REQUIRED PATH_SUFFIXES lib/cmake)
message(STATUS "Found XDG: ${XDG_DIR} (version ${XDG_VERSION})")

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(fmt CONFIG REQUIRED)
find_package(argparse CONFIG QUIET)

if(NOT argparse_FOUND)
  set(ARGPARSE_INSTALL OFF CACHE BOOL "" FORCE)
  set(ARGPARSE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(ARGPARSE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(vendor/argparse)
endif()

add_executable(xdg-particle-sim-event
  src/particle_sim_event.cpp
)

target_compile_features(xdg-particle-sim-event PRIVATE cxx_std_17)

target_include_directories(xdg-particle-sim-event PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(xdg-particle-sim-event PRIVATE
  xdg::xdg
  OpenMP::OpenMP_CXX
  fmt::fmt
  argparse::argparse
)

if(EVENT_SIM_ENABLE_CUDA_THRUST_SORT)
  message(STATUS "Building xdg-particle-sim-event with CUDA Thrust sorting")
  add_subdirectory(cuda_thrust_sort)
elseif(EVENT_SIM_ENABLE_HIP_THRUST_SORT)
  message(STATUS "Building xdg-particle-sim-event with HIP Thrust sorting")
  add_subdirectory(hip_thrust_sort)
endif()

if(TARGET event_sim_thrust_sort)
  target_compile_definitions(xdg-particle-sim-event PRIVATE EVENT_SIM_THRUST_SORT)
  target_link_libraries(xdg-particle-sim-event PRIVATE event_sim_thrust_sort)
endif()

install(TARGETS xdg-particle-sim-event
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(TARGET event_sim_thrust_sort)
  install(TARGETS event_sim_thrust_sort
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
