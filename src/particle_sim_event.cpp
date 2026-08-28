#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "xdg/error.h"
#include "xdg/timer.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/vec3da.h"
#include "xdg/xdg.h"

#include "argparse/argparse.hpp"

#include "event_sim_output.h"
#include "particle_sim_event.h"

using namespace xdg;

int main(int argc, char** argv) {

  // argument parsing
  argparse::ArgumentParser args("XDG Particle Pseudo-Simulation", "1.0", argparse::default_arguments::help);

  args.add_argument("filename")
      .help("Path to the input file");

  args.add_argument("-m", "--mfp")
      .default_value(1.0)
      .help("Mean free path of the particles").scan<'g', double>();

  args.add_argument("-n", "--n-particles")
      .default_value(1000000u)
      .help("Number of particles to simulate").scan<'u', uint32_t>();

  args.add_argument("-e", "--max-events")
      .default_value(1000u)
      .help("Maximum number of events per particle").scan<'u', uint32_t>();

  args.add_argument("-g", "--ipc-graveyard")
      .default_value(false)
      .implicit_value(true)
      .help("Treat the implicit complement as a graveyard (i.e. particles that enter it are killed)");

  args.add_argument("-ml", "--mesh-library")
      .help("Mesh library to use. One of (MOAB, LIBMESH)")
      .default_value("MOAB");

  args.add_argument("-rt", "--rt-library")
      .help("Ray tracing library to use. Event transport currently requires CUBQL")
      .default_value("CUBQL");

  args.add_argument("-f", "--format")
    .default_value("human")
    .choices("human", "csv")
    .help("stdout format. Human readable (default) or csv");

  args.add_argument("-p", "--enable-profiling-ray-launch")
      .default_value(false)
      .implicit_value(true)
      .help("Collect per-launch ray tracing profiling data");

  args.add_argument("-o", "--ray-launch-profile-output")
      .default_value("ray-launches.csv")
      .help("Output file for per-launch ray tracing profiling data");

  args.add_argument("--lost-particle-output")
      .default_value("lost-particles.csv")
      .help("Output file for lost particle state records");

  args.add_argument("--max-lost-particle-records")
      .default_value(1024u)
      .help("Maximum number of lost particle state records to retain")
      .scan<'u', uint32_t>();

  args.add_argument("--cell-track-output")
      .default_value("cell-tracks.csv")
      .help("Output file for per-volume cell track-length tallies");

  args.add_argument("--sort-mode")
      .metavar("{disabled,volume,direction,volume-direction,direction-volume}") // print in help message
      .default_value("disabled")
      .choices("disabled", "volume", "direction", "volume-direction", "direction-volume")
      .help("Advance-queue sorting mode; compound modes list the primary key first. Pick from: "
            "{disabled,volume,direction,volume-direction,direction-volume}");

  args.add_argument("-i", "--nsort", "--minimum-sort-items")
      .default_value(20000)
      .help("Minimum advance queue size required for particle sorting").scan<'i', int>();

  args.add_argument("-d", "--exit-on-bvh-failure")
      .default_value(false)
      .implicit_value(true)
      .help("Log BVH diagnostics and exit the program if the BVH traversal unexpectedly fails");

  try {
    args.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << args;
    return 1;
  }

  const std::string sort_mode_name = args.get<std::string>("--sort-mode");
  const int minimum_sort_items = args.get<int>("--minimum-sort-items");

  ParticleSortMode particle_sort_mode = ParticleSortMode::Disabled;
  if (sort_mode_name == "volume") {
    particle_sort_mode = ParticleSortMode::Volume;
  } else if (sort_mode_name == "direction") {
    particle_sort_mode = ParticleSortMode::Direction;
  } else if (sort_mode_name == "volume-direction") {
    particle_sort_mode = ParticleSortMode::VolumeDirection;
  } else if (sort_mode_name == "direction-volume") {
    particle_sort_mode = ParticleSortMode::DirectionVolume;
  }

#ifndef EVENT_SIM_THRUST_SORT
  if (particle_sorting_enabled(particle_sort_mode)) {
    std::cerr << "Particle sorting requested, but this build does not include a Thrust sorting backend.\n";
    return 1;
  }
#endif

  if (minimum_sort_items < 0) {
    std::cerr << "Minimum sort items must be non-negative.\n";
    return 1;
  }

  Timer wall_timer;
  wall_timer.start();

  EventSimulationData sim_data;

  // create a mesh manager
  std::string mesh_str = args.get<std::string>("--mesh-library");
  std::string rt_str = args.get<std::string>("--rt-library");
  const std::string model_filename = args.get<std::string>("filename");
  const std::string model_name = std::filesystem::path(model_filename).filename().string();
  const std::string output_format = args.get<std::string>("--format");
  const std::string lost_particle_output =
    args.get<std::string>("--lost-particle-output");
  const uint32_t max_lost_particle_records =
    args.get<uint32_t>("--max-lost-particle-records");
  const bool exit_on_bvh_failure =
    args.get<bool>("--exit-on-bvh-failure");

  if (max_lost_particle_records == 0) {
    fatal_error("Maximum number of lost particle records must be greater than 0");
  }

  RTLibrary rt_lib;
  if (rt_str == "EMBREE")
    rt_lib = RTLibrary::EMBREE;
  else if (rt_str == "GPRT")
    rt_lib = RTLibrary::GPRT;
  else if (rt_str == "CUBQL")
    rt_lib = RTLibrary::CUBQL;
  else
    fatal_error("Invalid ray tracing library '{}' specified", rt_str);

  if (exit_on_bvh_failure && rt_lib != RTLibrary::CUBQL) {
    fatal_error("--exit-on-bvh-failure currently requires the CUBQL ray tracing backend");
  }

  MeshLibrary mesh_lib;
  if (mesh_str == "MOAB")
    mesh_lib = MeshLibrary::MOAB;
  else if (mesh_str == "LIBMESH") {
    mesh_lib = MeshLibrary::LIBMESH;
    if (rt_lib == RTLibrary::GPRT)
      fatal_error("LibMesh is not currently supported with GPRT");
  }
  else
    fatal_error("Invalid mesh library '{}' specified", mesh_str);

  // create an XDG instance with the specified mesh and ray tracing library
  Timer xdg_setup_timer;
  xdg_setup_timer.start();
  std::shared_ptr<XDG> xdg = XDG::create(mesh_lib, rt_lib);
  const auto& mm = xdg->mesh_manager();
  mm->load_file(model_filename);
  mm->init();
  mm->parse_metadata();
  xdg->prepare_raytracer();
  xdg_setup_timer.stop();
  sim_data.profiling.xdg_setup_s += xdg_setup_timer.elapsed();

  sim_data.xdg_ = xdg;

  const int model_num_volumes = mm->num_volumes();
  const int model_num_surfaces = mm->num_surfaces();
  const int model_num_volume_elements = mm->num_volume_elements();
  const int model_num_vertices = mm->num_vertices();
  std::uint64_t model_num_surface_primitives = 0;
  for (MeshID surface : mm->surfaces()) {
    model_num_surface_primitives += static_cast<std::uint64_t>(mm->num_surface_faces(surface));
  }

  // update the mean free path
  sim_data.mfp_ = args.get<double>("--mfp");

  sim_data.profile_ray_launches_ = args.get<bool>("--enable-profiling-ray-launch");
  sim_data.particle_sort_mode_ = particle_sort_mode;
  sim_data.minimum_sort_items_ = minimum_sort_items;
  sim_data.implicit_complement_is_graveyard_ = args.get<bool>("--ipc-graveyard");
  sim_data.n_particles_ = args.get<uint32_t>("--n-particles");
  sim_data.max_events_ = args.get<uint32_t>("--max-events");
  sim_data.record_lost_particles_ = true;
  sim_data.max_lost_particle_records_ = max_lost_particle_records;
  sim_data.exit_on_bvh_failure_ = exit_on_bvh_failure;

  transport_particle_event_based(sim_data);

  wall_timer.stop();
  const double wall_time = wall_timer.elapsed();

  event_sim_output::sort_lost_particle_records(sim_data);

  if (sim_data.stopped_on_bvh_failure_) {
    event_sim_output::write_lost_particle_csv(lost_particle_output, sim_data);
    event_sim_output::print_lost_particle_diagnostic(
      std::cout, lost_particle_output, sim_data);

    if (!sim_data.host_lost_particles_.empty()) {
      xdg->bvh_diagnostics(sim_data.host_lost_particles_.front().volume);
    }

    std::cout << "Exiting early after detecting a BVH traversal failure.\n";
    return 0;
  }

  if (sim_data.profile_ray_launches_) {
    const std::string ray_launch_profile_output =
      args.get<std::string>("--ray-launch-profile-output");
    event_sim_output::write_ray_launch_profile_csv(ray_launch_profile_output, sim_data);
  }

  event_sim_output::write_lost_particle_csv(lost_particle_output, sim_data);

  const event_sim_output::SummaryMetadata summary_metadata {
    model_name,
    mesh_str,
    rt_str,
    model_num_volumes,
    model_num_surfaces,
    model_num_volume_elements,
    model_num_vertices,
    model_num_surface_primitives,
    wall_time
  };

  // Print lost particle diagnostics to stderr if the output format is CSV, otherwise print to stdout
  std::ostream& lost_particle_diagnostic_output = output_format == "csv" ? std::cerr : std::cout;
  event_sim_output::print_lost_particle_diagnostic(lost_particle_diagnostic_output, lost_particle_output, sim_data);

  // Write cell track-length tallies to CSV
  const std::string cell_track_output = args.get<std::string>("--cell-track-output");
  event_sim_output::write_cell_track_csv(cell_track_output, mm->volumes(), sim_data);

  // Write summary metadata and simulation statistics to stdout in the specified format
  event_sim_output::write_summary(std::cout, output_format, summary_metadata, sim_data);

  return 0;
}
