#include <array>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "particle_sim_history.h"
#include "xdg/xdg.h"

using namespace xdg;

TEST_CASE("pwr-pincell history sim agrees with event reference")
{
  std::shared_ptr<XDG> xdg = XDG::create(MeshLibrary::MOAB, RTLibrary::EMBREE);
  const auto& mesh_manager = xdg->mesh_manager();
  mesh_manager->load_file("pwr_pincell.h5m");
  mesh_manager->init();
  mesh_manager->parse_metadata();
  xdg->prepare_raytracer();

  HistorySimulationData sim_data;
  sim_data.xdg_ = xdg;
  sim_data.n_particles_ = 100000;
  sim_data.max_events_ = 100;
  sim_data.mfp_ = 0.5;

  transport_particle_history_based(sim_data);

  const std::array expected_cell_tracks {
    std::pair{MeshID{1}, 716521.42314976},
    std::pair{MeshID{2}, 240571.85058854704},
    std::pair{MeshID{3}, 1278350.0991759342},
    std::pair{MeshID{4}, 0.0}
  };

  REQUIRE(sim_data.profiling.particles_reached_max_events == 100000);
  REQUIRE(sim_data.profiling.particles_dead == 0);
  REQUIRE(sim_data.profiling.particles_lost == 0);
  REQUIRE(sim_data.profiling.rays_traced == 10000000);

  for (const auto& [volume, expected_track] : expected_cell_tracks) {
    INFO("volume_id=" << volume);
    REQUIRE_THAT(sim_data.cell_tracks.at(volume),
                 Catch::Matchers::WithinRel(expected_track, 1.0e-12));
  }
}
