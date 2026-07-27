#include <stdexcept>

#include <hip/hip_runtime_api.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "particle_event_queue.h"

void thrust_sort_by_volume(EventQueueItem* begin, EventQueueItem* end, int device_id)
{
  hipError_t error = hipSetDevice(device_id);
  if (error != hipSuccess) {
    throw std::runtime_error(hipGetErrorString(error));
  }

  thrust::sort(thrust::device, begin, end, VolumeCompare {});

  error = hipDeviceSynchronize();
  if (error != hipSuccess) {
    throw std::runtime_error(hipGetErrorString(error));
  }
}
