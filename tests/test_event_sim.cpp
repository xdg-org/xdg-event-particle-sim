// for testing
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
  mm->load_file(model_filename);
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


}
