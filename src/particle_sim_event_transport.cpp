#include "particle_sim_event.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <omp.h>

#include "xdg/error.h"
#include "xdg/timer.h"

using namespace xdg;

namespace {

std::int32_t count_active_queued_volumes(const AdvanceParticleQueue::DD& advance_queue,
                                         int num_rays,
                                         std::uint64_t* device_last_queried_launch_by_volume,
                                         std::uint64_t launch_index,
                                         int gpu_id)
{
  std::int32_t num_active_volumes = 0;

  #pragma omp target teams distribute parallel for device(gpu_id) \
    is_device_ptr(device_last_queried_launch_by_volume) \
    map(tofrom: num_active_volumes) \
    firstprivate(advance_queue, launch_index)
  for (int i = 0; i < num_rays; ++i) {
    const MeshID volume = advance_queue.data[i].volume;
    std::uint64_t previous_launch_index;

    #pragma omp atomic capture
    {
      previous_launch_index = device_last_queried_launch_by_volume[volume];
      device_last_queried_launch_by_volume[volume] = launch_index;
    }

    if (previous_launch_index != launch_index) {
      #pragma omp atomic update
      num_active_volumes++;
    }
  }

  return num_active_volumes;
}

void count_particle_states(
  const EventParticle* device_particles,
  int num_particles,
  std::uint64_t& particles_reached_max_events,
  std::uint64_t& particles_dead,
  std::uint64_t& particles_lost,
  int gpu_id)
{
  std::uint64_t reached_max_events = 0;
  std::uint64_t dead = 0;
  std::uint64_t lost = 0;

  #pragma omp target teams distribute parallel for device(gpu_id) \
    is_device_ptr(device_particles) \
    reduction(+: reached_max_events, dead, lost)
  for (int i = 0; i < num_particles; ++i) {
    if (device_particles[i].alive_) {
      reached_max_events++;
    } else if (device_particles[i].lost_) {
      lost++;
    } else {
      dead++;
    }
  }

  particles_reached_max_events = reached_max_events;
  particles_dead = dead;
  particles_lost = lost;
}

} // namespace

void transport_particle_event_based(EventSimulationData& sim_data)
{
  const double xdg_setup_s = sim_data.profiling.xdg_setup_s;
  sim_data.profiling = {};
  sim_data.profiling.xdg_setup_s = xdg_setup_s;
  sim_data.host_ray_launch_records_.clear();
  sim_data.host_lost_particles_.clear();
  sim_data.stopped_on_bvh_failure_ = false;

  if (sim_data.exit_on_bvh_failure_) {
    sim_data.record_lost_particles_ = true;
  }

  Timer transport_timer;
  transport_timer.start();

  if (sim_data.n_particles_ == 0) {
    fatal_error("Number of event particles must be greater than 0");
  }

  if (sim_data.n_particles_ > sim_data.max_particles_in_flight_) {
    fatal_error("Event particle refill is not implemented; n_particles must be <= max_particles_in_flight");
  }

  sim_data.ray_hits = sim_data.xdg_->allocate_ray_hits(sim_data.n_particles_);
  sim_data.gpu_id = sim_data.ray_hits.device_id;
  sim_data.host_id = omp_get_initial_device();

  // Setup device side storage for cell tracklengths
  const auto num_volumes = sim_data.xdg_->mesh_manager()->next_volume_id();
  sim_data.cell_tracks.assign(num_volumes, 0.0);
  const auto cell_track_bytes = static_cast<std::size_t>(num_volumes) * sizeof(double);

  sim_data.device_cell_tracks = static_cast<double*>(
    omp_target_alloc(cell_track_bytes, sim_data.gpu_id));
  if (!sim_data.device_cell_tracks) {
    fatal_error("Failed to allocate cell tracks on OpenMP target device.");
  }

  if (omp_target_memcpy(sim_data.device_cell_tracks,
                        sim_data.cell_tracks.data(),
                        cell_track_bytes,
                        0,
                        0,
                        sim_data.gpu_id,
                        sim_data.host_id) != 0) {
    fatal_error("Failed to initialize device cell tracks.");
  }

  // Allocate device side storage for event particles and event queues
  sim_data.device_particles = static_cast<EventParticle*>(
    omp_target_alloc(sim_data.n_particles_ * sizeof(EventParticle), sim_data.gpu_id));
  if (!sim_data.device_particles) {
    fatal_error("Failed to allocate event particles on OpenMP target device.");
  }

  sim_data.advance_particle_queue.allocate(sim_data.n_particles_, sim_data.gpu_id);
  sim_data.surface_crossing_queue.allocate(sim_data.n_particles_, sim_data.gpu_id);
  sim_data.collision_queue.allocate(sim_data.n_particles_, sim_data.gpu_id);

  // Allocate storage for lost particle bank
  if (sim_data.record_lost_particles_) {
    if (sim_data.max_lost_particle_records_ == 0) {
      fatal_error("Maximum number of lost particle records must be greater than 0");
    }
    sim_data.lost_particle_bank.allocate(sim_data.max_lost_particle_records_, sim_data.gpu_id);
  }

  if (sim_data.profile_ray_launches_) {
    const std::size_t table_size = static_cast<std::size_t>(sim_data.xdg_->mesh_manager()->next_volume_id());
    const std::size_t table_bytes = table_size * sizeof(std::uint64_t);
    const std::vector<std::uint64_t> initial_values(table_size, std::numeric_limits<std::uint64_t>::max());

    sim_data.device_last_queried_launch_by_volume =
      static_cast<std::uint64_t*>(omp_target_alloc(table_bytes, sim_data.gpu_id));
    omp_target_memcpy(sim_data.device_last_queried_launch_by_volume,
                      initial_values.data(),
                      table_bytes,
                      0,
                      0,
                      sim_data.gpu_id,
                      sim_data.host_id);
  }

  process_init_events(sim_data);

  while (true) {
    const std::int64_t max = std::max({
      sim_data.advance_particle_queue.size(),
      sim_data.surface_crossing_queue.size(),
      sim_data.collision_queue.size()});

    if (max == 0) {
      break;
    } else if (max == sim_data.advance_particle_queue.size()) {
      process_advance_particle_events(sim_data);

      if (sim_data.exit_on_bvh_failure_) {
        sim_data.lost_particle_bank.sync_size_device_to_host();
        if (sim_data.lost_particle_bank.size() > 0) {
          sim_data.stopped_on_bvh_failure_ = true;
          break;
        }
      }
    } else if (max == sim_data.surface_crossing_queue.size()) {
      process_surface_crossing_events(sim_data);
    } else if (max == sim_data.collision_queue.size()) {
      process_collision_events(sim_data);
    }
  }

  count_particle_states(sim_data.device_particles,
                        static_cast<int>(sim_data.n_particles_),
                        sim_data.profiling.particles_reached_max_events,
                        sim_data.profiling.particles_dead,
                        sim_data.profiling.particles_lost,
                        sim_data.gpu_id);

  sim_data.advance_particle_queue.release();
  sim_data.surface_crossing_queue.release();
  sim_data.collision_queue.release();

  if (sim_data.device_last_queried_launch_by_volume) {
    omp_target_free(sim_data.device_last_queried_launch_by_volume, sim_data.gpu_id);
    sim_data.device_last_queried_launch_by_volume = nullptr;
  }

  if (sim_data.record_lost_particles_) {
    sim_data.lost_particle_bank.sync_size_device_to_host();

    const int n_lost_records = sim_data.lost_particle_bank.size();
    sim_data.host_lost_particles_.resize(n_lost_records);

    if (n_lost_records > 0) {
      const int copy_status = omp_target_memcpy(
        sim_data.host_lost_particles_.data(),
        sim_data.lost_particle_bank.d_data,
        static_cast<std::size_t>(n_lost_records) * sizeof(LostParticleState),
        0,
        0,
        sim_data.host_id,
        sim_data.gpu_id);
      if (copy_status != 0) {
        fatal_error("Failed to copy lost particle records to the host");
      }
    }

    sim_data.lost_particle_bank.release();
  }

  // Copy cell tracklengths back to host
  if (omp_target_memcpy(sim_data.cell_tracks.data(),
                        sim_data.device_cell_tracks,
                        cell_track_bytes,
                        0,
                        0,
                        sim_data.host_id,
                        sim_data.gpu_id) != 0) {
    fatal_error("Failed to copy cell tracks from the OpenMP target device.");
  }

  omp_target_free(sim_data.device_cell_tracks, sim_data.gpu_id);
  sim_data.device_cell_tracks = nullptr;

  omp_target_free(sim_data.device_particles, sim_data.gpu_id);
  sim_data.device_particles = nullptr;

  sim_data.xdg_->free_ray_hits(sim_data.ray_hits);

  transport_timer.stop();
  sim_data.profiling.transport_s += transport_timer.elapsed();
}

void process_init_events(EventSimulationData& sim_data)
{
  Position r {0.0, 0.0, 0.0};
  Direction u {1.0, 0.0, 0.0};
  MeshID volume = sim_data.xdg_->find_volume(r, u);

  EventParticle* device_particles = sim_data.device_particles;
  const int n_particles = static_cast<int>(sim_data.n_particles_);
  const std::uint32_t seed = sim_data.seed_;
  const int gpu_id = sim_data.gpu_id;
  auto advance_queue = sim_data.advance_particle_queue.get_device_data();

  if (!device_particles) {
    fatal_error("Error allocating event particle device storage.");
  }

  sim_data.advance_particle_queue.reset();
  sim_data.surface_crossing_queue.reset();
  sim_data.collision_queue.reset();

  #pragma omp target teams distribute parallel for device(gpu_id) \
    is_device_ptr(device_particles) \
    firstprivate(advance_queue, seed, r, volume)
  for (int i = 0; i < n_particles; ++i) {
    std::uint32_t rng_state = seed ^ static_cast<std::uint32_t>(i);
    double direction[3];
    event_sim::random::random_unit_dir_lcg(rng_state, direction);

    Direction u;
    u.x = direction[0];
    u.y = direction[1];
    u.z = direction[2];

    device_particles[i].initialize(static_cast<uint32_t>(i), rng_state, r, u, volume);
    advance_queue.thread_safe_append({static_cast<uint32_t>(i), volume, get_direction_octant(u.x, u.y, u.z)});
  }

  sim_data.advance_particle_queue.sync_size_device_to_host();
}

void process_advance_particle_events(EventSimulationData& sim_data)
{
  const int n_advance = sim_data.advance_particle_queue.size();

  if (n_advance == 0) {
    warning("Advance_particle_events launched with a queue size of 0. Early return called...");
    return;
  }

  Timer total_timer;
  total_timer.start();
  sim_data.profiling.ray_launches++;
  sim_data.profiling.rays_traced += static_cast<std::uint64_t>(n_advance);

  EventParticle* device_particles = sim_data.device_particles;
  XDGRayHit* ray_hits = sim_data.ray_hits.data;
  auto advance_queue = sim_data.advance_particle_queue.get_device_data();
  auto surface_crossing_queue = sim_data.surface_crossing_queue.get_device_data();
  auto collision_queue = sim_data.collision_queue.get_device_data();
  const double mfp = sim_data.mfp_;
  const int gpu_id = sim_data.gpu_id;
  double launch_ray_sort_s = 0.0;

  Timer timer;

  const bool sorting_requested = particle_sorting_enabled(sim_data.particle_sort_mode_);

  if (sorting_requested) {
#ifdef EVENT_SIM_THRUST_SORT
    if (n_advance >= sim_data.minimum_sort_items_) {
      timer.start();
      // Sort the advance queue based on the requested sorting mode
      thrust_sort_event_queue(advance_queue.data,
                              advance_queue.data + n_advance,
                              sim_data.particle_sort_mode_,
                              gpu_id);
      timer.stop();
      launch_ray_sort_s = timer.elapsed();
      sim_data.profiling.advance_sort_rays_s += launch_ray_sort_s;
      timer.reset();
    }
#else
    fatal_error("Particle sorting requested, but this build does not include a Thrust sorting backend.");
#endif
  }

  timer.start();

  #pragma omp target teams distribute parallel for device(gpu_id) \
    is_device_ptr(device_particles, ray_hits) \
    firstprivate(advance_queue)
  for (int i = 0; i < n_advance; ++i) {
    const uint32_t particle_idx = advance_queue.data[i].idx;
    EventParticle& p = device_particles[particle_idx];

    ray_hits[i].origin[0] = p.r_.x;
    ray_hits[i].origin[1] = p.r_.y;
    ray_hits[i].origin[2] = p.r_.z;
    ray_hits[i].direction[0] = p.u_.x;
    ray_hits[i].direction[1] = p.u_.y;
    ray_hits[i].direction[2] = p.u_.z;
    ray_hits[i].t_min = 0.0;
    ray_hits[i].t_max = INFTY;
    ray_hits[i].volume = p.volume_;
    ray_hits[i].last_hit_primitive = ID_NONE;
    ray_hits[i].distance = INFTY;
    ray_hits[i].surface = ID_NONE;
    ray_hits[i].primitive = ID_NONE;
    ray_hits[i].point_in_volume = OUTSIDE;
    ray_hits[i].next_volume = ID_NONE;
    ray_hits[i].boundary_condition = static_cast<std::int32_t>(UNSET);
    ray_hits[i].normal[0] = 0.0;
    ray_hits[i].normal[1] = 0.0;
    ray_hits[i].normal[2] = 0.0;
  }
  timer.stop();
  sim_data.profiling.advance_pack_rays_s += timer.elapsed();

  XDGRayHitBuffer active_hits {
    sim_data.ray_hits.data,
    static_cast<std::size_t>(n_advance),
    sim_data.ray_hits.device_id
  };

  timer.reset();
  timer.start();
  sim_data.xdg_->ray_fire_batch(active_hits);
  timer.stop();
  const double launch_ray_trace_s = timer.elapsed();
  sim_data.profiling.total_ray_trace_s += launch_ray_trace_s;

  double launch_ray_throughput = 0.0;
  if (launch_ray_trace_s > 0.0) {
    launch_ray_throughput = static_cast<double>(n_advance) / launch_ray_trace_s;
    const double launch_count = static_cast<double>(sim_data.profiling.ray_launches);
    sim_data.profiling.average_ray_launch_throughput +=
      (launch_ray_throughput - sim_data.profiling.average_ray_launch_throughput) / launch_count;
  }

  timer.reset();

  if (sim_data.profile_ray_launches_) {
    const std::uint64_t launch_index = sim_data.profiling.ray_launches - 1;
    const std::int32_t num_active_volumes = count_active_queued_volumes(advance_queue,
                                                                        n_advance,
                                                                        sim_data.device_last_queried_launch_by_volume,
                                                                        launch_index,
                                                                        gpu_id);

    sim_data.host_ray_launch_records_.push_back({
      launch_index,
      n_advance,
      num_active_volumes,
      launch_ray_sort_s,
      launch_ray_trace_s,
      launch_ray_throughput
    });
  }

  const bool record_lost_particles = sim_data.record_lost_particles_;
  auto lost_particle_bank = sim_data.lost_particle_bank.get_device_data();

  // Get the device pointer to the cell tracks map for atomic updates
  double* device_cell_tracks = sim_data.device_cell_tracks;

  timer.start();
  #pragma omp target teams distribute parallel for device(gpu_id) \
    is_device_ptr(device_particles, ray_hits, device_cell_tracks) \
    firstprivate(advance_queue, surface_crossing_queue, collision_queue, \
                 record_lost_particles, lost_particle_bank, mfp)
  for (int i = 0; i < n_advance; i++) {
    const uint32_t particle_idx = advance_queue.data[i].idx;
    EventParticle& p = device_particles[particle_idx];
    const XDGRayHit& hit = ray_hits[i];

    if (hit.distance == 0.0) {
      p.alive_ = false;
      p.stuck_ = true;
      continue;
    }

    if (hit.surface == ID_NONE) {
      p.alive_ = false;
      p.lost_ = true;
      if (record_lost_particles) {
        lost_particle_bank.thread_safe_append({
          p.id_,
          p.r_,
          p.u_,
          p.volume_,
          p.rng_state_,
          p.last_surface_hit_,
          p.n_events_
        });
      }
      continue;
    }

    p.store_surface_crossing(hit.surface,
                             hit.distance,
                             hit.next_volume,
                             static_cast<SurfaceBoundaryCondition>(hit.boundary_condition),
                             hit.normal);

    p.sample_collision_distance(mfp);

    // Get the volume to score and the tracklength to add
    const MeshID track_volume = p.volume_;

    const double track_length = p.advance();

    // Atomically increment the tracklength tally for the volume
    #pragma omp atomic update
    device_cell_tracks[track_volume] += track_length;

    if (p.collision_distance_ < p.surface_hit_distance_) {
      collision_queue.thread_safe_append({particle_idx});
    } else {
      surface_crossing_queue.thread_safe_append({particle_idx});
    }
  }
  timer.stop();
  sim_data.profiling.advance_update_particles_s += timer.elapsed();

  sim_data.surface_crossing_queue.sync_size_device_to_host();
  sim_data.collision_queue.sync_size_device_to_host();
  sim_data.advance_particle_queue.reset();

  total_timer.stop();
  sim_data.profiling.advance_total_s += total_timer.elapsed();
}

void process_collision_events(EventSimulationData& sim_data)
{
  EventParticle* device_particles = sim_data.device_particles;
  const int n_collisions = sim_data.collision_queue.size();

  if (n_collisions == 0) {
    return;
  }

  Timer timer;
  timer.start();
  sim_data.profiling.collision_calls++;

  auto advance_queue = sim_data.advance_particle_queue.get_device_data();
  auto collision_queue = sim_data.collision_queue.get_device_data();
  const int gpu_id = sim_data.gpu_id;
  const int max_events = sim_data.max_events_;

  #pragma omp target teams distribute parallel for device(gpu_id) \
    is_device_ptr(device_particles) \
    firstprivate(advance_queue, collision_queue, max_events)
  for (int i = 0; i < n_collisions; i++) {
    const uint32_t particle_idx = collision_queue.data[i].idx;
    EventParticle& p = device_particles[particle_idx];

    p.collide();

    if (p.n_events_ >= max_events) {
      continue;
    }

    advance_queue.thread_safe_append({particle_idx, p.volume_, get_direction_octant(p.u_.x, p.u_.y, p.u_.z)});
  }

  sim_data.advance_particle_queue.sync_size_device_to_host();
  sim_data.collision_queue.reset();
  timer.stop();
  sim_data.profiling.collision_s += timer.elapsed();
}

void process_surface_crossing_events(EventSimulationData& sim_data)
{
  EventParticle* device_particles = sim_data.device_particles;
  const int n_surface_crossings = sim_data.surface_crossing_queue.size();

  if (n_surface_crossings == 0) {
    return;
  }

  Timer timer;
  timer.start();
  sim_data.profiling.surface_crossing_calls++;

  auto advance_queue = sim_data.advance_particle_queue.get_device_data();
  auto surface_crossing_queue = sim_data.surface_crossing_queue.get_device_data();
  const int gpu_id = sim_data.gpu_id;
  const int max_events = sim_data.max_events_;

  #pragma omp target teams distribute parallel for device(gpu_id) \
    is_device_ptr(device_particles) \
    firstprivate(advance_queue, surface_crossing_queue, max_events)
  for (int i = 0; i < n_surface_crossings; i++) {
    const uint32_t particle_idx = surface_crossing_queue.data[i].idx;
    EventParticle& p = device_particles[particle_idx];
    p.surface_cross();

    if (!p.alive_ || p.n_events_ >= max_events) {
      continue;
    }

    advance_queue.thread_safe_append({particle_idx, p.volume_, get_direction_octant(p.u_.x, p.u_.y, p.u_.z)});
  }

  sim_data.advance_particle_queue.sync_size_device_to_host();
  sim_data.surface_crossing_queue.reset();
  timer.stop();
  sim_data.profiling.surface_crossing_s += timer.elapsed();
}
