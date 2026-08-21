cmake_minimum_required(VERSION 3.18)

project(xdg_event_particle_sim VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type" FORCE)
endif()

option(EVENT_SIM_ENABLE_CUDA_THRUST_SORT "Enable CUDA Thrust event-queue sorting" OFF)
option(EVENT_SIM_ENABLE_HIP_THRUST_SORT "Enable HIP Thrust event-queue sorting" OFF)
option(EVENT_SIM_BUILD_TESTS "Build event simulator tests" ON)

if(EVENT_SIM_ENABLE_CUDA_THRUST_SORT AND EVENT_SIM_ENABLE_HIP_THRUST_SORT)
  message(FATAL_ERROR "CUDA and HIP Thrust sorting cannot both be enabled")
endif()

find_package(XDG REQUIRED PATH_SUFFIXES lib/cmake)
message(STATUS "Found XDG: ${XDG_DIR} (version ${XDG_VERSION})")

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(fmt CONFIG REQUIRED)
find_package(Git QUIET)

set(VENDOR_PATHS)

find_package(argparse CONFIG QUIET)

if(NOT argparse_FOUND)
  list(APPEND VENDOR_PATHS vendor/argparse)
endif()

# attempt to find external Catch2
if(EVENT_SIM_BUILD_TESTS)
  find_package(Catch2 QUIET NO_SYSTEM_ENVIRONMENT_PATH)
  if(NOT Catch2_FOUND)
    list(APPEND VENDOR_PATHS vendor/catch2)
  endif()
  list(APPEND VENDOR_PATHS tests/test_files)
endif()

if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  option(EVENT_SIM_GIT_SUBMODULE "Check submodules during configuration" ON)
  if(EVENT_SIM_GIT_SUBMODULE)
    message(STATUS "Updating required submodules")
    foreach(VENDOR_PATH IN LISTS VENDOR_PATHS)
      message(STATUS "Updating submodule at ${VENDOR_PATH}")
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive "${VENDOR_PATH}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE GIT_SUBMOD_RESULT
      )
      if(NOT GIT_SUBMOD_RESULT EQUAL 0)
        message(FATAL_ERROR
          "git submodule update --init failed with error code ${GIT_SUBMOD_RESULT}, "
          "please check out the submodule at ${VENDOR_PATH} manually and try again."
        )
      endif()
    endforeach()
  endif()
endif()

foreach(VENDOR_PATH IN LISTS VENDOR_PATHS)
  file(GLOB _event_sim_vendor_readmes
    "${CMAKE_CURRENT_SOURCE_DIR}/${VENDOR_PATH}/README.*"
  )
  if(NOT _event_sim_vendor_readmes)
    message(FATAL_ERROR
      "The vendored dependency at ${VENDOR_PATH} is missing. "
      "Expected a README.* file. Please update submodules and try again."
    )
  endif()
endforeach()

if(NOT argparse_FOUND)
  set(ARGPARSE_INSTALL OFF CACHE BOOL "" FORCE)
  set(ARGPARSE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(ARGPARSE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(vendor/argparse)
endif()

# Catch2 testing module
if(EVENT_SIM_BUILD_TESTS)
  if(NOT Catch2_FOUND)
    add_subdirectory("vendor/catch2")
  endif()
  enable_testing()
  include(Catch)
  message(STATUS "Using Catch2 ${Catch2_VERSION} at ${Catch2_DIR}")
endif()

set(EVENT_SIM_LIBRARY_SOURCES
  src/particle_sim_event_transport.cpp
)

add_library(event-sim-core STATIC
  ${EVENT_SIM_LIBRARY_SOURCES}
)

target_compile_features(event-sim-core PUBLIC cxx_std_17)

target_include_directories(event-sim-core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(event-sim-core
  PUBLIC
    xdg::xdg
    OpenMP::OpenMP_CXX
  PRIVATE
    fmt::fmt
)

add_executable(xdg-particle-sim-event
  src/event_sim_output.cpp
  src/particle_sim_event.cpp
)

target_compile_features(xdg-particle-sim-event PRIVATE cxx_std_17)

target_link_libraries(xdg-particle-sim-event PRIVATE
  event-sim-core
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
  target_compile_definitions(event-sim-core PRIVATE EVENT_SIM_THRUST_SORT)
  target_link_libraries(event-sim-core PRIVATE event_sim_thrust_sort)
endif()

if(EVENT_SIM_BUILD_TESTS)
  add_subdirectory("tests")
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
