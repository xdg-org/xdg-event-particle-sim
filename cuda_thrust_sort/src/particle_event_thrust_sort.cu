#include <stdexcept>

#include <cuda_runtime_api.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "particle_event_queue.h"

void thrust_sort_by_volume(EventQueueItem* begin, EventQueueItem* end, int device_id)
{
  cudaError_t error = cudaSetDevice(device_id);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  thrust::sort(thrust::device, begin, end, VolumeCompare {});

  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}
