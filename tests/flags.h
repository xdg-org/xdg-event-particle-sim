#ifndef EVENT_SIM_TEST_FLAGS_H
#define EVENT_SIM_TEST_FLAGS_H

#include <type_traits>

#include "event_simulation_data.h"

namespace event_sim::test {

using NoSort = std::integral_constant<ParticleSortMode, ParticleSortMode::Disabled>;
using VolumeSort = std::integral_constant<ParticleSortMode, ParticleSortMode::Volume>;
using DirectionSort = std::integral_constant<ParticleSortMode, ParticleSortMode::Direction>;
using VolumeDirectionSort = std::integral_constant<ParticleSortMode, ParticleSortMode::VolumeDirection>;
using DirectionVolumeSort = std::integral_constant<ParticleSortMode, ParticleSortMode::DirectionVolume>;

#ifdef EVENT_SIM_THRUST_SORT
#define EVENT_SIM_SORTING_OPTIONS \
  NoSort, \
  VolumeSort, \
  DirectionSort, \
  VolumeDirectionSort, \
  DirectionVolumeSort
#else
#define EVENT_SIM_SORTING_OPTIONS NoSort
#endif

} // namespace event_sim::test

#endif
