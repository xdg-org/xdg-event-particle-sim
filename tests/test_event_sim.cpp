// for testing
#include <array>
#include <cstdint>
#include <utility>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "xdg/xdg.h"
#include "particle_sim_event.h"

using namespace xdg;

TEST_CASE("pwr-pincell event sim test")
{
  // Setup XDG
  std::shared_ptr<XDG> xdg = XDG::create(MeshLibrary::MOAB, RTLibrary::CUBQL);
  const auto& mm = xdg->mesh_manager();
  mm->load_file("pwr_pincell.h5m");
  mm->init();
  mm->parse_metadata();
  xdg->prepare_raytracer();

  // Setup EventSimulationData
  EventSimulationData sim_data;
  sim_data.xdg_ = xdg;
  sim_data.n_particles_ = 100000;
  sim_data.max_events_ = 100;
  sim_data.mfp_ = 0.5;
  sim_data.profile_ray_launches_ = true;
  sim_data.profile_volume_occupancy_ = true;

  transport_particle_event_based(sim_data);

  const std::array expected_cell_tracks {
    std::pair{MeshID{1}, 716521.42314976},
    std::pair{MeshID{2}, 240571.85058854704},
    std::pair{MeshID{3}, 1278350.0991759342},
    std::pair{MeshID{4}, 0.0}
  };

  // Basic triage of the results
  REQUIRE(sim_data.profiling.particles_reached_max_events == 100000);
  REQUIRE(sim_data.profiling.particles_dead == 0);
  REQUIRE(sim_data.profiling.particles_lost == 0);
  REQUIRE(sim_data.profiling.ray_launches == 166);
  REQUIRE(sim_data.profiling.rays_traced == 10000000);

  REQUIRE(sim_data.host_ray_launch_records_.size()
          == sim_data.profiling.ray_launches);

  std::uint64_t num_profiled_rays = 0;
  std::uint64_t num_initial_rays = 0;
  std::uint64_t num_collision_rays = 0;
  std::uint64_t num_surface_crossing_rays = 0;
  std::uint64_t num_volume_occupancy_rays = 0;
  for (const auto& launch : sim_data.host_ray_launch_records_) {
    const auto num_event_source_rays =
      launch.num_initial_rays
      + launch.num_collision_rays
      + launch.num_surface_crossing_rays;
    REQUIRE(num_event_source_rays == launch.num_rays);

    num_profiled_rays += static_cast<std::uint64_t>(launch.num_rays);
    num_initial_rays += static_cast<std::uint64_t>(launch.num_initial_rays);
    num_collision_rays += static_cast<std::uint64_t>(launch.num_collision_rays);
    num_surface_crossing_rays +=
      static_cast<std::uint64_t>(launch.num_surface_crossing_rays);

    std::uint64_t launch_occupancy_rays = 0;
    for (const auto& occupancy : launch.volume_occupancies) {
      REQUIRE(occupancy.num_rays > 0);
      launch_occupancy_rays +=
        static_cast<std::uint64_t>(occupancy.num_rays);
    }
    REQUIRE(launch_occupancy_rays == static_cast<std::uint64_t>(launch.num_rays));
    REQUIRE(launch.volume_occupancies.size()
            == static_cast<std::size_t>(launch.num_active_volumes));
    num_volume_occupancy_rays += launch_occupancy_rays;
  }

  REQUIRE(num_profiled_rays == sim_data.profiling.rays_traced);
  REQUIRE(num_initial_rays == sim_data.n_particles_);
  REQUIRE(num_collision_rays > 0);
  REQUIRE(num_surface_crossing_rays > 0);
  REQUIRE(num_volume_occupancy_rays == sim_data.profiling.rays_traced);

  const auto& initial_launch = sim_data.host_ray_launch_records_.front();
  REQUIRE(initial_launch.num_initial_rays == initial_launch.num_rays);
  REQUIRE(initial_launch.num_collision_rays == 0);
  REQUIRE(initial_launch.num_surface_crossing_rays == 0);
  REQUIRE(initial_launch.volume_occupancies.size() == 1);
  REQUIRE(initial_launch.volume_occupancies.front().num_rays
          == initial_launch.num_rays);

  // Check the cell track lengths
  for (const auto& [volume, expected_track] : expected_cell_tracks) {
    INFO("volume_id=" << volume);
    REQUIRE_THAT(sim_data.cell_tracks.at(volume),
                 Catch::Matchers::WithinRel(expected_track, 1.0e-12));
  }

}
