#ifndef EVENT_SIM_DEVICE_APPEND_QUEUE_H
#define EVENT_SIM_DEVICE_APPEND_QUEUE_H

#include <omp.h>

#include "xdg/error.h"

// Lightweight append queue used by event queues.
// Uses the same internal device data view pattern borrowed from cuBQL/DPRT.
template<typename T>
struct DeviceAppendQueue {
  struct DD {
    T* data {nullptr};
    int* size {nullptr}; // Number of current valid entries
    int capacity {0}; // Current allocated limit on number of entries before container overflow

    int thread_safe_append(const T& value)
    {
      int idx; 
      #pragma omp atomic capture 
      idx = (*size)++;

      if (idx >= capacity) {
        #pragma omp atomic write
        *size = capacity;
        return -1;
      }

      data[idx] = value;
      return idx;
    };
  };

  T* d_data {nullptr};
  int* d_size {nullptr};
  int h_size {0};
  int capacity {0};
  int gpu_id {0};
  int host_id {omp_get_initial_device()};

  void allocate(int capacity_, int gpu_id_)
  {
    release();

    capacity = capacity_;
    gpu_id = gpu_id_;
    host_id = omp_get_initial_device();
    h_size = 0;

    if (capacity == 0) return;

    d_data = static_cast<T*>(omp_target_alloc(capacity * sizeof(T), gpu_id));
    d_size = static_cast<int*>(omp_target_alloc(sizeof(int), gpu_id));

    if (!d_data || !d_size) {
      release();
      fatal_error("Failed to allocate event queue on OpenMP target device.");
    }

    resize(0);
  }

  void release()
  {
    if (d_data) {
      omp_target_free(d_data, gpu_id);
      d_data = nullptr;
    }
    if (d_size) {
      omp_target_free(d_size, gpu_id);
      d_size = nullptr;
    }

    h_size = 0;
    capacity = 0;
  }

  void resize(int size)
  {
    h_size = size;
    if (!d_size) return;

    omp_target_memcpy(d_size,
                      &h_size,
                      sizeof(int),
                      0,
                      0,
                      gpu_id,
                      host_id);
  }

  // Small wrapper to call resize with size 0
  void reset()
  {
    resize(0);
  }

  void sync_size_device_to_host()
  {
    if (!d_size) return;

    omp_target_memcpy(&h_size,
                      d_size,
                      sizeof(int),
                      0,
                      0,
                      host_id,
                      gpu_id);
  }

  int size() const { return h_size; }

  DD get_device_data() const
  {
    return {d_data, d_size, capacity};
  }
};

#endif
