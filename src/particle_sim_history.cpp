#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "argparse/argparse.hpp"

#include "event_sim_output.h"
#include "particle_sim_history.h"

#include "xdg/error.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/timer.h"
#include "xdg/xdg.h"

using namespace xdg;

int main(int argc, char** argv)
{
  argparse::ArgumentParser args(
    "XDG History-Based Particle Pseudo-Simulation",
    "1.0",
    argparse::default_arguments::help);

  args.add_argument("filename")
      .help("Path to the input file");

  args.add_argument("-m", "--mfp")
      .default_value(1.0)
      .help("Mean free path of the particles").scan<'g', double>();

  args.add_argument("-n", "--n-particles")
      .default_value(1000000u)
      .help("Number of particles to simulate").scan<'u', std::uint32_t>();

  args.add_argument("-e", "--max-events")
      .default_value(1000u)
      .help("Maximum number of events per particle").scan<'u', std::uint32_t>();

  args.add_argument("-s", "--seed")
      .default_value(42u)
      .help("Base random-number seed").scan<'u', std::uint32_t>();

  args.add_argument("-ml", "--mesh-library")
      .help("Mesh library to use. One of (MOAB, LIBMESH)")
      .default_value("MOAB");

  args.add_argument("-rt", "--rt-library")
      .help("CPU ray tracing library to use. Currently EMBREE")
      .default_value("EMBREE");

  args.add_argument("-f", "--format")
      .default_value("human")
      .choices("human", "csv")
      .help("stdout format. Human readable (default) or csv");

  args.add_argument("--cell-track-output")
      .default_value("cell-tracks.csv")
      .help("Output file for per-volume cell track-length tallies");

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cout << err.what() << '\n';
    std::cout << args;
    return 1;
  }

  const std::string mesh_name = args.get<std::string>("--mesh-library");
  const std::string rt_name = args.get<std::string>("--rt-library");
  const std::string model_filename = args.get<std::string>("filename");
  const std::string model_name =
    std::filesystem::path(model_filename).filename().string();
  const std::string output_format = args.get<std::string>("--format");

  if (rt_name != "EMBREE") {
    fatal_error(
      "The history-based CPU comparison currently requires the EMBREE ray tracing backend");
  }
  const RTLibrary rt_library = RTLibrary::EMBREE;

  MeshLibrary mesh_library;
  if (mesh_name == "MOAB") {
    mesh_library = MeshLibrary::MOAB;
  } else if (mesh_name == "LIBMESH") {
    mesh_library = MeshLibrary::LIBMESH;
  } else {
    fatal_error("Invalid mesh library '{}' specified", mesh_name);
  }

  Timer wall_timer;
  wall_timer.start();

  Timer setup_timer;
  setup_timer.start();
  std::shared_ptr<XDG> xdg = XDG::create(mesh_library, rt_library);
  const auto& mesh_manager = xdg->mesh_manager();
  mesh_manager->load_file(model_filename);
  mesh_manager->init();
  mesh_manager->parse_metadata();
  xdg->prepare_raytracer();
  setup_timer.stop();

  HistorySimulationData sim_data;
  sim_data.xdg_ = xdg;
  sim_data.mfp_ = args.get<double>("--mfp");
  sim_data.seed_ = args.get<std::uint32_t>("--seed");
  sim_data.n_particles_ = args.get<std::uint32_t>("--n-particles");
  sim_data.max_events_ = args.get<std::uint32_t>("--max-events");
  sim_data.profiling.xdg_setup_s = setup_timer.elapsed();

  transport_particle_history_based(sim_data);

  wall_timer.stop();

  std::uint64_t model_num_surface_primitives = 0;
  for (const MeshID surface : mesh_manager->surfaces()) {
    model_num_surface_primitives += static_cast<std::uint64_t>(
      mesh_manager->num_surface_faces(surface));
  }

  const event_sim_output::SummaryMetadata summary_metadata {
    model_name,
    mesh_name,
    rt_name,
    mesh_manager->num_volumes(),
    mesh_manager->num_surfaces(),
    mesh_manager->num_volume_elements(),
    mesh_manager->num_vertices(),
    model_num_surface_primitives,
    wall_timer.elapsed()
  };

  event_sim_output::write_cell_track_csv(
    args.get<std::string>("--cell-track-output"),
    mesh_manager->volumes(),
    sim_data.cell_tracks);
  event_sim_output::write_summary(
    std::cout, output_format, summary_metadata, sim_data);

  return 0;
}
