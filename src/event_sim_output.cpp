#include "event_sim_output.h"

#include <algorithm>
#include <fstream>
#include <ostream>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "xdg/error.h"

namespace event_sim_output {

void write_summary(std::ostream& output,
                   const std::string& format,
                   const SummaryMetadata& metadata,
                   const EventSimulationData& sim_data)
{
  // Compute total ray throughput (rays traced per second) if the total ray trace time is greater than zero
  const auto& profiling = sim_data.profiling;
  const double total_ray_throughput =
    profiling.total_ray_trace_s > 0.0
      ? static_cast<double>(profiling.rays_traced) / profiling.total_ray_trace_s
      : 0.0;

  // Rank cell tracks by track length in descending order
  const auto& mm = sim_data.xdg_->mesh_manager();
  std::vector<std::pair<MeshID, double>> ranked_cell_tracks;
  ranked_cell_tracks.reserve(mm->volumes().size());
  for (const MeshID volume : mm->volumes()) {
    const auto volume_index = static_cast<std::size_t>(volume);
    ranked_cell_tracks.emplace_back(volume, sim_data.cell_tracks.at(volume_index));
  }

  auto rank_cell_tallies = [](const auto& lhs, const auto& rhs) {
    if (lhs.second != rhs.second) {
      return lhs.second > rhs.second;
    }
    return lhs.first < rhs.first;
  };

  std::sort(ranked_cell_tracks.begin(), ranked_cell_tracks.end(), rank_cell_tallies);

  const auto num_cell_tracks_to_print =
    std::min<std::size_t>(5, ranked_cell_tracks.size());

  if (format == "csv") {
    const std::vector<std::string> columns {
      "model",
      "mesh_library",
      "rt_library",
      "model_num_volumes",
      "model_num_surfaces",
      "model_num_volume_elements",
      "model_num_vertices",
      "model_num_surface_primitives",
      "n_particles",
      "max_events",
      "mean_free_path",
      "particles_reached_max_events",
      "particles_dead",
      "particles_lost",
      "sort_rays_by_volume",
      "minimum_sort_items",
      "wall_time_s",
      "profile_xdg_setup_s",
      "profile_transport_s",
      "profile_advance_total_s",
      "profile_advance_sort_rays_s",
      "profile_advance_pack_rays_s",
      "profile_total_ray_trace_s",
      "profile_advance_update_particles_s",
      "profile_collision_s",
      "profile_surface_crossing_s",
      "profile_ray_launches",
      "profile_collision_calls",
      "profile_surface_crossing_calls",
      "profile_rays_traced",
      "profile_total_ray_throughput_rays_per_s",
      "profile_average_ray_launch_throughput_rays_per_s"
    };

    const std::vector<std::string> values {
      metadata.model_name,
      metadata.mesh_library,
      metadata.ray_tracing_library,
      fmt::format("{}", metadata.model_num_volumes),
      fmt::format("{}", metadata.model_num_surfaces),
      fmt::format("{}", metadata.model_num_volume_elements),
      fmt::format("{}", metadata.model_num_vertices),
      fmt::format("{}", metadata.model_num_surface_primitives),
      fmt::format("{}", sim_data.n_particles_),
      fmt::format("{}", sim_data.max_events_),
      fmt::format("{}", sim_data.mfp_),
      fmt::format("{}", profiling.particles_reached_max_events),
      fmt::format("{}", profiling.particles_dead),
      fmt::format("{}", profiling.particles_lost),
      fmt::format("{}", sim_data.sort_rays_by_volume_),
      fmt::format("{}", sim_data.minimum_sort_items_),
      fmt::format("{}", metadata.wall_time_s),
      fmt::format("{}", profiling.xdg_setup_s),
      fmt::format("{}", profiling.transport_s),
      fmt::format("{}", profiling.advance_total_s),
      fmt::format("{}", profiling.advance_sort_rays_s),
      fmt::format("{}", profiling.advance_pack_rays_s),
      fmt::format("{}", profiling.total_ray_trace_s),
      fmt::format("{}", profiling.advance_update_particles_s),
      fmt::format("{}", profiling.collision_s),
      fmt::format("{}", profiling.surface_crossing_s),
      fmt::format("{}", profiling.ray_launches),
      fmt::format("{}", profiling.collision_calls),
      fmt::format("{}", profiling.surface_crossing_calls),
      fmt::format("{}", profiling.rays_traced),
      fmt::format("{}", total_ray_throughput),
      fmt::format("{}", profiling.average_ray_launch_throughput)
    };

    output << fmt::format("{}\n", fmt::join(columns, ","));
    output << fmt::format("{}\n", fmt::join(values, ","));
    return;
  }

  output << "\nXDG event-based particle pseudo-simulation\n";
  output << "----------------------------------------\n";
  output << "Model                 : " << metadata.model_name << "\n";
  output << "Mesh library          : " << metadata.mesh_library << "\n";
  output << "Ray tracing library   : " << metadata.ray_tracing_library << "\n";
  output << "Volumes               : " << metadata.model_num_volumes << "\n";
  output << "Surfaces              : " << metadata.model_num_surfaces << "\n";
  output << "Volume elements       : " << metadata.model_num_volume_elements << "\n";
  output << "Vertices              : " << metadata.model_num_vertices << "\n";
  output << "Surface primitives    : " << metadata.model_num_surface_primitives << "\n";
  output << "Particles             : " << sim_data.n_particles_ << "\n";
  output << "Max events/particle   : " << sim_data.max_events_ << "\n";
  output << "Mean free path        : " << sim_data.mfp_ << "\n";
  output << "Volume sorting        : "
         << (sim_data.sort_rays_by_volume_ ? "enabled" : "disabled") << "\n";
  output << "Minimum sort items    : " << sim_data.minimum_sort_items_ << "\n";
  output << "----------------------------------------\n";
  output << "Full wall-clock time  : " << metadata.wall_time_s << " s\n";
  output << "XDG setup             : " << profiling.xdg_setup_s << " s\n";
  output << "Transport time        : " << profiling.transport_s << " s\n";
  output << "Advance total         : " << profiling.advance_total_s
         << " s (" << profiling.ray_launches << " ray launches)\n";
  output << "  Sort rays           : " << profiling.advance_sort_rays_s << " s\n";
  output << "  Pack rays           : " << profiling.advance_pack_rays_s << " s\n";
  output << "  Ray trace           : " << profiling.total_ray_trace_s << " s\n";
  output << "  Update particles    : " << profiling.advance_update_particles_s << " s\n";
  output << "  Collision events    : " << profiling.collision_s
         << " s (" << profiling.collision_calls << " calls)\n";
  output << "  Surface crossings   : " << profiling.surface_crossing_s
         << " s (" << profiling.surface_crossing_calls << " calls)\n";
  output << "----------------------------------------\n";
  output << "Reached max events    : " << profiling.particles_reached_max_events << "\n";
  output << "Particles dead        : " << profiling.particles_dead << "\n";
  output << "Particles lost        : " << profiling.particles_lost << "\n";
  output << "Ray launches          : " << profiling.ray_launches << "\n";
  output << "Rays traced           : "
         << fmt::format("{:.6e}", static_cast<double>(profiling.rays_traced)) << "\n";
  output << "Avg launch throughput : "
         << profiling.average_ray_launch_throughput << " rays/s\n";
  output << "Total ray throughput  : " << total_ray_throughput << " rays/s\n";
  output << "----------------------------------------\n";
  output << "Cell Track Lengths\n";
  output << "----------------------------------------\n";
  for (std::size_t i = 0; i < num_cell_tracks_to_print; ++i) {
    const auto& [volume, track_length] = ranked_cell_tracks[i];
    output << fmt::format("Cell {}: {}\n", volume, track_length);
  }
  output << "----------------------------------------\n";
}

void write_ray_launch_profile_csv(const std::string& filename,
                                  const EventSimulationData& sim_data)
{
  std::ofstream output(filename);
  if (!output) {
    fatal_error("Failed to open ray launch profiling output '{}'.", filename);
  }

  output << "launch_index,num_rays,num_active_volumes,volume_sort_s,ray_trace_s,"
            "ray_throughput_rays_per_s\n";
  for (const auto& launch : sim_data.host_ray_launch_records_) {
    output << fmt::format("{},{},{},{:.17g},{:.17g},{:.17g}\n",
                          launch.launch_index,
                          launch.num_rays,
                          launch.num_active_volumes,
                          launch.volume_sort_s,
                          launch.ray_trace_s,
                          launch.ray_throughput);
  }
}

void sort_lost_particle_records(EventSimulationData& sim_data)
{
  std::sort(sim_data.host_lost_particles_.begin(),
            sim_data.host_lost_particles_.end(),
            [](const LostParticleState& lhs, const LostParticleState& rhs) {
              return lhs.particle_id < rhs.particle_id;
            });
}

void write_lost_particle_csv(const std::string& filename,
                             const EventSimulationData& sim_data)
{
  std::ofstream output(filename);
  if (!output) {
    fatal_error("Failed to open lost particle output '{}'.", filename);
  }

  output << "particle_id,n_events,rng_state,volume,last_surface_hit,"
            "position_x,position_y,position_z,"
            "direction_x,direction_y,direction_z\n";

  for (const auto& lost : sim_data.host_lost_particles_) {
    output << fmt::format(
      "{},{},{},{},{},"
      "{:.17g},{:.17g},{:.17g},"
      "{:.17g},{:.17g},{:.17g}\n",
      lost.particle_id,
      lost.n_events,
      lost.rng_state,
      lost.volume,
      lost.last_surface_hit,
      lost.position.x,
      lost.position.y,
      lost.position.z,
      lost.direction.x,
      lost.direction.y,
      lost.direction.z);
  }
}

void print_lost_particle_diagnostic(std::ostream& output,
                                    const std::string& filename,
                                    const EventSimulationData& sim_data)
{
  if (sim_data.host_lost_particles_.empty()) {
    return;
  }

  const auto& lost = sim_data.host_lost_particles_.front();

  output << "\nRepresentative lost particle (lowest particle ID)\n";
  output << "----------------------------------------\n";
  output << "Particle ID           : " << lost.particle_id << "\n";
  output << "Events                : " << lost.n_events << "\n";
  output << "RNG state             : " << lost.rng_state << "\n";
  output << "Volume                : " << lost.volume << "\n";
  output << "Last surface hit      : " << lost.last_surface_hit << "\n";
  output << fmt::format("Position              : {:.17g}, {:.17g}, {:.17g}\n",
                        lost.position.x, lost.position.y, lost.position.z);
  output << fmt::format("Direction             : {:.17g}, {:.17g}, {:.17g}\n",
                        lost.direction.x, lost.direction.y, lost.direction.z);
  output << "Lost particle records : " << filename << "\n";

  if (sim_data.profiling.particles_lost > sim_data.host_lost_particles_.size()) {
    output << "Warning               : " << sim_data.profiling.particles_lost
           << " particles were lost, but only "
           << sim_data.host_lost_particles_.size() << " records were captured.\n";
  }
  output << "----------------------------------------\n";
}

void write_cell_track_csv(const std::string& filename,
                          const std::vector<MeshID>& volumes,
                          const EventSimulationData& sim_data)
{
  std::ofstream cell_track_ofstream(filename);
  if (!cell_track_ofstream) {
    fatal_error("Failed to open cell track-length output '{}'.", filename);
  }
  cell_track_ofstream << "volume_id,track_length\n";
  for (const MeshID volume : volumes) {
    cell_track_ofstream << fmt::format(
      "{},{:.17g}\n", volume, sim_data.cell_tracks.at(volume));
  }
}

} // namespace event_sim_output
