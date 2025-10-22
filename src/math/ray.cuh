#pragma once

#include "util.cuh"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h> 
#include <device_launch_parameters.h>

extern __device__ int ray_count = 0;




// returns the current value of ray_count
// - needs to be in a wrapper to work
int get_ray_count() {
    int host_count = 0;
    // Copy the value from the GPU variable (ray_count) to a CPU variable (host_count)
    cudaMemcpyFromSymbol(&host_count, ray_count, sizeof(int));
    return host_count;
}

class ray {
public:
    vec3 origin, direction, inv_direction;

    __device__ ray() {}
    __device__ ray(const vec3& o, const vec3& dir) : origin(o), direction(dir) {
        atomicAdd(&ray_count, 1); 
        inv_direction = 1.0f / dir;
    }
    __device__ vec3 at(float t) const { return origin + t * direction; }
};