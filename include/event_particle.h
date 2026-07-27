#ifndef EVENT_SIM_EVENT_PARTICLE_H
#define EVENT_SIM_EVENT_PARTICLE_H

#include <cmath>
#include <cstdint>

#include "xdg/constants.h"
#include "xdg/vec3da.h"

#include "random_lcg.h"

using namespace xdg;

#ifdef _OPENMP
#pragma omp declare target
#endif

struct EventParticle {
  void initialize(uint32_t id,
                  std::uint32_t rng_state,
                  Position r,
                  Direction u,
                  MeshID volume)
  {
    id_ = id;
    r_.x = r.x;
    r_.y = r.y;
    r_.z = r.z;
    u_.x = u.x;
    u_.y = u.y;
    u_.z = u.z;
    volume_ = volume;
    surface_hit_ = ID_NONE;
    surface_hit_distance_ = INFTY;
    collision_distance_ = INFTY;
    last_surface_hit_ = ID_NONE;
    next_volume_ = ID_NONE;
    boundary_condition_ = UNSET;
    surface_normal_.x = 0.0;
    surface_normal_.y = 0.0;
    surface_normal_.z = 0.0;
    rng_state_ = rng_state;
    n_events_ = 0;
    alive_ = true;
    stuck_ = false;
  }

  void sample_collision_distance(double mfp)
  {
    collision_distance_ = -std::log(1.0 - event_sim::random::rand01(rng_state_)) * mfp;
  }

  double advance()
  {
    double distance;

    if (collision_distance_ < surface_hit_distance_) {
      distance = collision_distance_;
    } else {
      distance = surface_hit_distance_;
    }

    // Explicit scalar update avoids device compilation issues with vec3da operators.
    r_.x += distance * u_.x;
    r_.y += distance * u_.y;
    r_.z += distance * u_.z;

    return distance;
  }

  void collide()
  {
    n_events_++;

    double direction[3];
    event_sim::random::random_unit_dir_lcg(rng_state_, direction);
    u_.x = direction[0];
    u_.y = direction[1];
    u_.z = direction[2];

    surface_hit_ = ID_NONE;
    surface_hit_distance_ = INFTY;
    last_surface_hit_ = ID_NONE;
    next_volume_ = ID_NONE;
    boundary_condition_ = UNSET;
    surface_normal_.x = 0.0;
    surface_normal_.y = 0.0;
    surface_normal_.z = 0.0;
  }

  void store_surface_crossing(MeshID surface,
                              double distance,
                              MeshID next_volume,
                              SurfaceBoundaryCondition boundary_condition,
                              const double normal[3])
  {
    surface_hit_ = surface;
    surface_hit_distance_ = distance;
    last_surface_hit_ = surface;
    next_volume_ = next_volume;
    boundary_condition_ = boundary_condition;
    surface_normal_.x = normal[0];
    surface_normal_.y = normal[1];
    surface_normal_.z = normal[2];
  }

  void surface_cross()
  {
    n_events_++;

    switch (boundary_condition_) {
    case SurfaceBoundaryCondition::TRANSMISSION:
      volume_ = next_volume_;

      // TODO: Restore optional implicit-complement graveyard handling.
      if (volume_ == ID_NONE) {
        alive_ = false;
      }
      break;

    case SurfaceBoundaryCondition::VACUUM:
      alive_ = false;
      break;

    case SurfaceBoundaryCondition::REFLECTIVE: {
      const double nx = surface_normal_.x;
      const double ny = surface_normal_.y;
      const double nz = surface_normal_.z;

      // cuBQL returns the raw triangle normal. Dividing by n dot n applies
      // the reflection formula without first normalising meaning we skip a square root.
      const double normal_squared = nx * nx + ny * ny + nz * nz;
      const double projection = u_.x * nx + u_.y * ny + u_.z * nz;
      const double scale = 2.0 * projection / normal_squared;

      u_.x -= scale * nx;
      u_.y -= scale * ny;
      u_.z -= scale * nz;

      const double direction_squared = u_.x * u_.x + u_.y * u_.y + u_.z * u_.z;
      const double inverse_direction_length = 1.0 / std::sqrt(direction_squared);
      u_.x *= inverse_direction_length;
      u_.y *= inverse_direction_length;
      u_.z *= inverse_direction_length;
      break;
    }

    case SurfaceBoundaryCondition::UNSET:
    default:
      alive_ = false;
      break;
    }
  }

  uint32_t id_ {0};
  Position r_ {};
  Direction u_ {};
  MeshID volume_ {ID_NONE};

  MeshID surface_hit_ {ID_NONE};
  double surface_hit_distance_ {INFTY};
  double collision_distance_ {INFTY};
  MeshID last_surface_hit_ {ID_NONE};
  MeshID next_volume_ {ID_NONE};
  SurfaceBoundaryCondition boundary_condition_ {UNSET};
  Direction surface_normal_ {0.0};

  std::uint32_t rng_state_ {0};
  int32_t n_events_ {0};
  bool alive_ {true};
  bool stuck_ {false};
};

#ifdef _OPENMP
#pragma omp end declare target
#endif

#endif
