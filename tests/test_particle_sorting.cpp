#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "particle_event_queue.h"

namespace {

AdvanceQueueItem item(std::uint32_t index, std::int32_t volume, DirectionOctant direction)
{
  return {index, volume, direction};
}

} // namespace

// This is not a device sort testing the underlying thrust implementation
// Rather this tests the comparators used for sorting on device
TEST_CASE("Sorting comparators")
{
  const std::array input {
    item(0, 2, DirectionOctant::X_NEG_Y_POS_Z_POS),
    item(1, 1, DirectionOctant::X_NEG_Y_NEG_Z_NEG),
    item(2, 1, DirectionOctant::X_POS_Y_NEG_Z_POS),
    item(3, 2, DirectionOctant::X_POS_Y_POS_Z_POS),
    item(4, 1, DirectionOctant::X_NEG_Y_POS_Z_POS)
  };

  SECTION("Volume then direction") {
    auto sorted = input;
    std::sort(sorted.begin(), sorted.end(), VolumeDirectionCompare {});

    const std::array<std::uint32_t, 5> expected_indices {4, 2, 1, 3, 0};
    for (std::size_t i = 0; i < sorted.size(); ++i) {
      REQUIRE(sorted[i].idx == expected_indices[i]);
    }
  }

  SECTION("Direction then volume") {
    auto sorted = input;
    std::sort(sorted.begin(), sorted.end(), DirectionVolumeCompare {});

    const std::array<std::uint32_t, 5> expected_indices {3, 4, 0, 2, 1};
    for (std::size_t i = 0; i < sorted.size(); ++i) {
      REQUIRE(sorted[i].idx == expected_indices[i]);
    }
  }
}
