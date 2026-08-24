#include <stdexcept>

#include <cuda_runtime_api.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "particle_event_queue.h"

void thrust_sort_event_queue(EventQueueItem* begin,
                             EventQueueItem* end,
                             ParticleSortMode mode,
                             int device_id)
{
  cudaError_t error = cudaSetDevice(device_id);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  switch (mode) {
    case ParticleSortMode::Disabled:
      return;
    case ParticleSortMode::Volume:
      thrust::sort(thrust::device, begin, end, VolumeCompare {});
      break;
    case ParticleSortMode::Direction:
      thrust::sort(thrust::device, begin, end, DirectionCompare {});
      break;
    case ParticleSortMode::VolumeDirection:
      thrust::sort(thrust::device, begin, end, VolumeDirectionCompare {});
      break;
    case ParticleSortMode::DirectionVolume:
      thrust::sort(thrust::device, begin, end, DirectionVolumeCompare {});
      break;
  }

  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}
