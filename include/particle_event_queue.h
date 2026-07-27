#ifndef EVENT_SIM_PARTICLE_EVENT_QUEUE_H
#define EVENT_SIM_PARTICLE_EVENT_QUEUE_H

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define EVENT_SIM_HOST_DEVICE __host__ __device__
#else
#define EVENT_SIM_HOST_DEVICE
#endif

struct EventQueueItem {
  std::uint32_t idx;
  std::int32_t volume;
};

struct VolumeCompare {
  EVENT_SIM_HOST_DEVICE
  bool operator()(const EventQueueItem& lhs, const EventQueueItem& rhs) const
  {
    return lhs.volume < rhs.volume;
  }
};

void thrust_sort_by_volume(EventQueueItem* begin, EventQueueItem* end, int device_id);

#undef EVENT_SIM_HOST_DEVICE

#endif
