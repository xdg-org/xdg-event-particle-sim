// for testing
#include <array>
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

  // Check the cell track lengths
  for (const auto& [volume, expected_track] : expected_cell_tracks) {
    INFO("volume_id=" << volume);
    REQUIRE_THAT(sim_data.cell_tracks.at(volume),
                 Catch::Matchers::WithinRel(expected_track, 1.0e-12));
  }

}
