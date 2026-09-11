#ifndef EVENT_SIM_HISTORY_SIMULATION_DATA_H
#define EVENT_SIM_HISTORY_SIMULATION_DATA_H

#include <cstdint>
#include <memory>
#include <vector>

#include "xdg/xdg.h"

using namespace xdg;

struct HistorySimulationData {
  struct Profiling {
    double xdg_setup_s {0.0};
    double transport_s {0.0};
    std::uint64_t rays_traced {0};
    std::uint64_t particles_reached_max_events {0};
    std::uint64_t particles_dead {0};
    std::uint64_t particles_lost {0};
  };

  std::shared_ptr<XDG> xdg_;
  double mfp_ {1.0};
  std::uint32_t seed_ {42};
  std::uint32_t n_particles_ {1000000};
  std::uint32_t max_events_ {1000};
  std::vector<double> cell_tracks;
  int num_threads_ {1};
  Profiling profiling;
};

#endif // EVENT_SIM_HISTORY_SIMULATION_DATA_H
