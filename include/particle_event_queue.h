#ifndef EVENT_SIM_PARTICLE_EVENT_QUEUE_H
#define EVENT_SIM_PARTICLE_EVENT_QUEUE_H

#include <cstdint>

// Macro to ensure method is compilable by NVCC and HIPCC.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define EVENT_SIM_HOST_DEVICE __host__ __device__
#else
#define EVENT_SIM_HOST_DEVICE
#endif

enum class DirectionOctant : std::uint8_t {
  X_POS_Y_POS_Z_POS = 0,
  X_NEG_Y_POS_Z_POS = 1,
  X_POS_Y_NEG_Z_POS = 2,
  X_NEG_Y_NEG_Z_POS = 3,
  X_POS_Y_POS_Z_NEG = 4,
  X_NEG_Y_POS_Z_NEG = 5,
  X_POS_Y_NEG_Z_NEG = 6,
  X_NEG_Y_NEG_Z_NEG = 7
};

enum class ParticleSortMode : std::uint8_t {
  Disabled,
  Volume,
  Direction,
  VolumeDirection,
  DirectionVolume
};

constexpr const char* particle_sort_mode_name(ParticleSortMode mode)
{
  switch (mode) {
    case ParticleSortMode::Disabled:
      return "disabled";
    case ParticleSortMode::Volume:
      return "volume";
    case ParticleSortMode::Direction:
      return "direction";
    case ParticleSortMode::VolumeDirection:
      return "volume-direction";
    case ParticleSortMode::DirectionVolume:
      return "direction-volume";
  }

  return "unknown";
}

constexpr bool particle_sorting_enabled(ParticleSortMode mode)
{
  return mode != ParticleSortMode::Disabled;
}

#ifdef _OPENMP
#pragma omp declare target
#endif

EVENT_SIM_HOST_DEVICE
inline DirectionOctant get_direction_octant(double x, double y, double z)
{
  std::uint8_t octant = 0;

  if (x < 0.0) {
    octant |= 1u;
  }
  if (y < 0.0) {
    octant |= 2u;
  }
  if (z < 0.0) {
    octant |= 4u;
  }

  return static_cast<DirectionOctant>(octant);
}

#ifdef _OPENMP
#pragma omp end declare target
#endif

struct EventQueueItem {
  std::uint32_t idx;
  std::int32_t volume;
  DirectionOctant direction_octant;
};

struct VolumeCompare {
  EVENT_SIM_HOST_DEVICE
  bool operator()(const EventQueueItem& lhs, const EventQueueItem& rhs) const
  {
    return lhs.volume < rhs.volume;
  }
};

struct DirectionCompare {
  EVENT_SIM_HOST_DEVICE
  bool operator()(const EventQueueItem& lhs, const EventQueueItem& rhs) const
  {
    return lhs.direction_octant < rhs.direction_octant;
  }
};

struct VolumeDirectionCompare {
  EVENT_SIM_HOST_DEVICE
  bool operator()(const EventQueueItem& lhs, const EventQueueItem& rhs) const
  {
    if (lhs.volume != rhs.volume) {
      return lhs.volume < rhs.volume;
    }
    return lhs.direction_octant < rhs.direction_octant;
  }
};

struct DirectionVolumeCompare {
  EVENT_SIM_HOST_DEVICE
  bool operator()(const EventQueueItem& lhs, const EventQueueItem& rhs) const
  {
    if (lhs.direction_octant != rhs.direction_octant) {
      return lhs.direction_octant < rhs.direction_octant;
    }
    return lhs.volume < rhs.volume;
  }
};

void thrust_sort_event_queue(EventQueueItem* begin,
                             EventQueueItem* end,
                             ParticleSortMode mode,
                             int device_id);

#undef EVENT_SIM_HOST_DEVICE

#endif
