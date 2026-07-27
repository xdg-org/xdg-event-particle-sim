#ifndef EVENT_SIM_EVENT_SIMULATION_DATA_H
#define EVENT_SIM_EVENT_SIMULATION_DATA_H

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <omp.h>

#include "xdg/xdg.h"

#include "device_append_queue.h"
#include "event_particle.h"
#include "particle_event_queue.h"

using namespace xdg;

using ParticleEventQueue = DeviceAppendQueue<EventQueueItem>;

struct EventSimulationData {
  struct Profiling {
    double xdg_setup_s {0.0};
    double transport_s {0.0};
    double advance_total_s {0.0};
    double advance_sort_rays_s {0.0};
    double advance_pack_rays_s {0.0};
    double total_ray_trace_s {0.0};
    double average_batch_ray_throughput {0.0};
    double advance_update_particles_s {0.0};
    double collision_s {0.0};
    double surface_crossing_s {0.0};
    std::uint64_t advance_calls {0};
    std::uint64_t collision_calls {0};
    std::uint64_t surface_crossing_calls {0};
    std::uint64_t rays_traced {0};
  };

  struct RayBatchProfilingRecord {
    std::uint64_t batch_index {0};
    std::int32_t num_rays {0};
    std::int32_t num_unique_volumes {0};
    double volume_sort_s {0.0};
    double ray_trace_s {0.0};
    double ray_throughput {0.0};
  };

  std::shared_ptr<XDG> xdg_;
  double mfp_ {1.0};
  std::uint32_t seed_ {42};
  uint32_t n_particles_ {1000000};
  uint32_t max_events_ {1000};
  bool profile_rays_ {false};
  bool sort_rays_by_volume_ {false};
  int minimum_sort_items_ {20000};
  bool implicit_complement_is_graveyard_ {false};
  std::unordered_map<MeshID, double> cell_tracks;

  uint32_t max_particles_in_flight_ {1000000};
  EventParticle* device_particles {nullptr};
  int gpu_id {0};
  int host_id {omp_get_initial_device()};
  XDGRayHitBuffer ray_hits;
  ParticleEventQueue advance_particle_queue;
  ParticleEventQueue surface_crossing_queue;
  ParticleEventQueue collision_queue;
  Profiling profiling;

  std::vector<RayBatchProfilingRecord> host_ray_batch_records_;

  // Last batch index that queried each volume MeshID.
  std::uint64_t* device_last_queried_batch_by_volume {nullptr};
};

#endif
