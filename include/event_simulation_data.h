#ifndef EVENT_SIM_EVENT_SIMULATION_DATA_H
#define EVENT_SIM_EVENT_SIMULATION_DATA_H

#include <cstdint>
#include <memory>
#include <vector>

#include <omp.h>

#include "xdg/xdg.h"

#include "device_append_queue.h"
#include "event_particle.h"
#include "particle_event_queue.h"

using namespace xdg;

struct LostParticleState {
  std::uint32_t particle_id {0};
  Position position {0.0, 0.0, 0.0};
  Direction direction {0.0, 0.0, 0.0};
  MeshID volume {ID_NONE};
  std::uint32_t rng_state {0};
  MeshID last_surface_hit {ID_NONE};
  std::int32_t n_events {0};
};

using ParticleEventQueue = DeviceAppendQueue<EventQueueItem>;
using LostParticleBank = DeviceAppendQueue<LostParticleState>;

struct EventSimulationData {
  struct Profiling {
    double xdg_setup_s {0.0};
    double transport_s {0.0};
    double advance_total_s {0.0};
    double advance_sort_rays_s {0.0};
    double advance_pack_rays_s {0.0};
    double total_ray_trace_s {0.0};
    double average_ray_launch_throughput {0.0};
    double advance_update_particles_s {0.0};
    double collision_s {0.0};
    double surface_crossing_s {0.0};
    std::uint64_t ray_launches {0};
    std::uint64_t collision_calls {0};
    std::uint64_t surface_crossing_calls {0};
    std::uint64_t rays_traced {0};
    std::uint64_t particles_reached_max_events {0};
    std::uint64_t particles_dead {0};
    std::uint64_t particles_lost {0};
  };

  struct RayLaunchProfilingRecord {
    std::uint64_t launch_index {0};
    std::int32_t num_rays {0};
    std::int32_t num_active_volumes {0};
    double volume_sort_s {0.0};
    double ray_trace_s {0.0};
    double ray_throughput {0.0};
  };

  std::shared_ptr<XDG> xdg_;
  double mfp_ {1.0};
  std::uint32_t seed_ {42};
  uint32_t n_particles_ {1000000};
  uint32_t max_events_ {1000};
  bool profile_ray_launches_ {false};
  bool sort_rays_by_volume_ {false};
  int minimum_sort_items_ {20000};
  bool implicit_complement_is_graveyard_ {false};
  std::vector<double> cell_tracks;
  double* device_cell_tracks {nullptr};

  uint32_t max_particles_in_flight_ {10000000};
  EventParticle* device_particles {nullptr};
  int gpu_id {0};
  int host_id {omp_get_initial_device()};
  XDGRayHitBuffer ray_hits;
  ParticleEventQueue advance_particle_queue;
  ParticleEventQueue surface_crossing_queue;
  ParticleEventQueue collision_queue;
  Profiling profiling;

  std::vector<RayLaunchProfilingRecord> host_ray_launch_records_;

  // Last ray launch index that queried each volume MeshID.
  std::uint64_t* device_last_queried_launch_by_volume {nullptr};

  // Lost particle state information
  bool record_lost_particles_ {false};
  bool exit_on_bvh_failure_ {false};
  bool stopped_on_bvh_failure_ {false};
  LostParticleBank lost_particle_bank;
  uint32_t max_lost_particle_records_ {1024};
  std::vector<LostParticleState> host_lost_particles_;
};


#endif
