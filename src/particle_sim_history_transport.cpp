#include "particle_sim_history.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <omp.h>

#include "event_particle.h"
#include "random_lcg.h"

#include "xdg/error.h"
#include "xdg/mesh_manager_interface.h"
#include "xdg/timer.h"

using namespace xdg;

namespace {

SurfaceBoundaryCondition parse_boundary_condition(const std::string& value,
                                                   MeshID surface)
{
  if (value == "transmission") {
    return SurfaceBoundaryCondition::TRANSMISSION;
  }
  if (value == "vacuum") {
    return SurfaceBoundaryCondition::VACUUM;
  }
  if (value == "reflecting" || value == "reflective") {
    return SurfaceBoundaryCondition::REFLECTIVE;
  }

  fatal_error("Unsupported boundary condition '{}' on surface {}", value, surface);
  return SurfaceBoundaryCondition::UNSET;
}

} // namespace

void transport_particle_history_based(HistorySimulationData& sim_data)
{
  if (!sim_data.xdg_) {
    fatal_error("Cannot transport particles without an initialized XDG instance");
  }
  if (sim_data.n_particles_ == 0) {
    fatal_error("Number of history particles must be greater than 0");
  }
  if (sim_data.max_events_ == 0) {
    fatal_error("Maximum events per particle must be greater than 0");
  }
  if (sim_data.mfp_ <= 0.0) {
    fatal_error("Mean free path must be greater than 0");
  }

  const double xdg_setup_s = sim_data.profiling.xdg_setup_s;
  sim_data.profiling = {};
  sim_data.profiling.xdg_setup_s = xdg_setup_s;

  Timer transport_timer;
  transport_timer.start();

  const Position source_position {0.0, 0.0, 0.0};
  const Direction volume_lookup_direction {1.0, 0.0, 0.0};
  const MeshID source_volume =
    sim_data.xdg_->find_volume(source_position, volume_lookup_direction);
  if (source_volume == ID_NONE) {
    fatal_error("Particle source at the origin is not inside a model volume");
  }

  const auto& mesh_manager = sim_data.xdg_->mesh_manager();
  const auto tally_size = static_cast<std::size_t>(mesh_manager->next_volume_id());
  sim_data.cell_tracks.assign(tally_size, 0.0);

  // Boundary metadata does not change during transport. Convert it once on the
  // host rather than performing map/string lookups in every particle history.
  std::vector<SurfaceBoundaryCondition> boundary_conditions(
    static_cast<std::size_t>(mesh_manager->next_surface_id()),
    SurfaceBoundaryCondition::UNSET);
  for (const MeshID surface : mesh_manager->surfaces()) {
    const auto property = mesh_manager->get_surface_property(
      surface, PropertyType::BOUNDARY_CONDITION);
    boundary_conditions.at(static_cast<std::size_t>(surface)) =
      parse_boundary_condition(property.value, surface);
  }

  const int requested_threads = omp_get_max_threads();
  sim_data.num_threads_ = requested_threads;

  std::vector<std::vector<double>> thread_cell_tracks(
    static_cast<std::size_t>(requested_threads),
    std::vector<double>(tally_size, 0.0));

  std::uint64_t rays_traced = 0;
  std::uint64_t particles_reached_max_events = 0;
  std::uint64_t particles_dead = 0;
  std::uint64_t particles_lost = 0;

  #pragma omp parallel num_threads(requested_threads) \
    reduction(+: rays_traced, particles_reached_max_events, particles_dead, \
                 particles_lost)
  {
    const int thread_id = omp_get_thread_num();
    auto& local_cell_tracks =
      thread_cell_tracks[static_cast<std::size_t>(thread_id)];
    std::vector<MeshID> hit_primitives;
    hit_primitives.reserve(1);

    #pragma omp single
    sim_data.num_threads_ = omp_get_num_threads();

    #pragma omp for schedule(static)
    for (std::uint32_t i = 0; i < sim_data.n_particles_; ++i) {
      std::uint32_t rng_state = sim_data.seed_ ^ i;
      double initial_direction[3];
      event_sim::random::random_unit_dir_lcg(rng_state, initial_direction);
      const Direction direction {
        initial_direction[0], initial_direction[1], initial_direction[2]
      };

      EventParticle particle;
      particle.initialize(i, rng_state, source_position, direction, source_volume);

      while (particle.alive_ &&
             particle.n_events_ < static_cast<std::int32_t>(sim_data.max_events_)) {
        // Match the GPU event path: do not exclude the primitive hit by the
        // previous ray, and trace to the nearest surface without a t_max limit.
        hit_primitives.clear();
        const auto surface_intersection = sim_data.xdg_->ray_fire(
          particle.volume_,
          particle.r_,
          particle.u_,
          INFTY,
          HitOrientation::EXITING,
          &hit_primitives);
        rays_traced++;

        const double surface_distance = surface_intersection.first;
        const MeshID surface = surface_intersection.second;

        if (surface_distance == 0.0) {
          particle.alive_ = false;
          particle.stuck_ = true;
          break;
        }
        if (surface == ID_NONE) {
          particle.alive_ = false;
          particle.lost_ = true;
          break;
        }

        const auto boundary_condition =
          boundary_conditions[static_cast<std::size_t>(surface)];
        MeshID next_volume = ID_NONE;
        if (boundary_condition == SurfaceBoundaryCondition::TRANSMISSION) {
          next_volume = mesh_manager->next_volume(particle.volume_, surface);
        }

        double normal[3] {0.0, 0.0, 0.0};
        if (boundary_condition == SurfaceBoundaryCondition::REFLECTIVE) {
          const Position hit_position {
            particle.r_.x + surface_distance * particle.u_.x,
            particle.r_.y + surface_distance * particle.u_.y,
            particle.r_.z + surface_distance * particle.u_.z
          };
          const Direction surface_normal = sim_data.xdg_->surface_normal(
            surface, hit_position, &hit_primitives);
          normal[0] = surface_normal.x;
          normal[1] = surface_normal.y;
          normal[2] = surface_normal.z;
        }

        particle.store_surface_crossing(
          surface, surface_distance, next_volume, boundary_condition, normal);

        // This ordering is intentional and matches the unbounded event-based
        // algorithm: the surface query occurs before collision sampling.
        particle.sample_collision_distance(sim_data.mfp_);

        const MeshID track_volume = particle.volume_;
        const double track_length = particle.advance();
        local_cell_tracks[static_cast<std::size_t>(track_volume)] += track_length;

        if (particle.collision_distance_ < particle.surface_hit_distance_) {
          particle.collide();
        } else {
          particle.surface_cross();
        }
      }

      if (particle.alive_ &&
          particle.n_events_ >= static_cast<std::int32_t>(sim_data.max_events_)) {
        particles_reached_max_events++;
      } else if (particle.lost_) {
        particles_lost++;
      } else {
        particles_dead++;
      }
    }
  }

  // A deterministic host reduction avoids atomics in the transport loop and
  // gives stable results for a fixed OpenMP thread count.
  for (const auto& local_cell_tracks : thread_cell_tracks) {
    for (std::size_t volume = 0; volume < tally_size; ++volume) {
      sim_data.cell_tracks[volume] += local_cell_tracks[volume];
    }
  }

  sim_data.profiling.rays_traced = rays_traced;
  sim_data.profiling.particles_reached_max_events = particles_reached_max_events;
  sim_data.profiling.particles_dead = particles_dead;
  sim_data.profiling.particles_lost = particles_lost;

  transport_timer.stop();
  sim_data.profiling.transport_s = transport_timer.elapsed();
}
