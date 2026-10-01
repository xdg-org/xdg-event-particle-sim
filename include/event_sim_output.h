#ifndef EVENT_SIM_EVENT_SIM_OUTPUT_H
#define EVENT_SIM_EVENT_SIM_OUTPUT_H

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "event_simulation_data.h"
#include "history_simulation_data.h"

namespace event_sim_output {

struct SummaryMetadata {
  std::string model_name;
  std::string mesh_library;
  std::string ray_tracing_library;
  int model_num_volumes {0};
  int model_num_surfaces {0};
  int model_num_volume_elements {0};
  int model_num_vertices {0};
  std::uint64_t model_num_surface_primitives {0};
  double wall_time_s {0.0};
};

void write_summary(std::ostream& output,
                   const std::string& format,
                   const SummaryMetadata& metadata,
                   const EventSimulationData& sim_data);

void write_summary(std::ostream& output,
                   const std::string& format,
                   const SummaryMetadata& metadata,
                   const HistorySimulationData& sim_data);

void write_ray_launch_profile_csv(const std::string& filename,
                                  const EventSimulationData& sim_data);

void sort_lost_particle_records(EventSimulationData& sim_data);

void write_lost_particle_csv(const std::string& filename,
                             const EventSimulationData& sim_data);

void print_lost_particle_diagnostic(std::ostream& output,
                                    const std::string& filename,
                                    const EventSimulationData& sim_data);


void write_cell_track_csv(const std::string& filename,
                          const std::vector<MeshID>& volumes,
                          const std::vector<double>& cell_tracks);
} // namespace event_sim_output

#endif
