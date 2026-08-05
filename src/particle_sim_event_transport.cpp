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

std::int32_t count_active_queued_volumes(const ParticleEventQueue::DD& advance_queue,
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
  int gpu_id)
{
  std::uint64_t reached_max_events = 0;
  std::uint64_t dead = 0;

  #pragma omp target teams distribute parallel for device(gpu_id) \
    is_device_ptr(device_particles) \
    reduction(+: reached_max_events, dead)
  for (int i = 0; i < num_particles; ++i) {
    if (device_particles[i].alive_) {
      reached_max_events++;
    } else {
      dead++;
    }
  }

  particles_reached_max_events = reached_max_events;
  particles_dead = dead;
}

} // namespace

void transport_particle_event_based(EventSimulationData& sim_data)
{
  const double xdg_setup_s = sim_data.profiling.xdg_setup_s;
  sim_data.profiling = {};
  sim_data.profiling.xdg_setup_s = xdg_setup_s;
  sim_data.host_ray_launch_records_.clear();

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

  sim_data.device_particles = static_cast<EventParticle*>(
    omp_target_alloc(sim_data.n_particles_ * sizeof(EventParticle), sim_data.gpu_id));
  if (!sim_data.device_particles) {
    fatal_error("Failed to allocate event particles on OpenMP target device.");
  }

  sim_data.advance_particle_queue.allocate(sim_data.n_particles_, sim_data.gpu_id);
  sim_data.surface_crossing_queue.allocate(sim_data.n_particles_, sim_data.gpu_id);
  sim_data.collision_queue.allocate(sim_data.n_particles_, sim_data.gpu_id);

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
                        sim_data.gpu_id);

  sim_data.advance_particle_queue.release();
  sim_data.surface_crossing_queue.release();
  sim_data.collision_queue.release();

  if (sim_data.device_last_queried_launch_by_volume) {
    omp_target_free(sim_data.device_last_queried_launch_by_volume, sim_data.gpu_id);
    sim_data.device_last_queried_launch_by_volume = nullptr;
  }

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
    advance_queue.thread_safe_append({static_cast<uint32_t>(i), volume});
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
  double launch_volume_sort_s = 0.0;

  Timer timer;

  if (sim_data.sort_rays_by_volume_) {
#ifdef EVENT_SIM_THRUST_SORT
    if (n_advance > sim_data.minimum_sort_items_) {
      timer.start();
      thrust_sort_by_volume(advance_queue.data, advance_queue.data + n_advance, gpu_id);
      timer.stop();
      launch_volume_sort_s = timer.elapsed();
      sim_data.profiling.advance_sort_rays_s += launch_volume_sort_s;
      timer.reset();
    }
#else
    fatal_error("Volume sorting requested, but this build does not include a Thrust sorting backend.");
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
    ray_hits[i].t_min = TINY_BIT;
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
      launch_volume_sort_s,
      launch_ray_trace_s,
      launch_ray_throughput
    });
  }

  timer.start();
  #pragma omp target teams distribute parallel for device(gpu_id) \
    is_device_ptr(device_particles, ray_hits) \
    firstprivate(advance_queue, surface_crossing_queue, collision_queue, mfp)
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
      continue;
    }

    p.store_surface_crossing(hit.surface,
                             hit.distance,
                             hit.next_volume,
                             static_cast<SurfaceBoundaryCondition>(hit.boundary_condition),
                             hit.normal);
    p.sample_collision_distance(mfp);
    p.advance();

    if (p.collision_distance_ < p.surface_hit_distance_) {
      collision_queue.thread_safe_append({particle_idx, p.volume_});
    } else {
      surface_crossing_queue.thread_safe_append({particle_idx, p.volume_});
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

    advance_queue.thread_safe_append({particle_idx, p.volume_});
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

    advance_queue.thread_safe_append({particle_idx, p.volume_});
  }

  sim_data.advance_particle_queue.sync_size_device_to_host();
  sim_data.surface_crossing_queue.reset();
  timer.stop();
  sim_data.profiling.surface_crossing_s += timer.elapsed();
}
