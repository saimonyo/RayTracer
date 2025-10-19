#pragma once

#include "vec3.cuh"
#include <curand_kernel.h>


#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

// generation via rejection sampling
__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1.0f, 1.0f, 0.0f);
    } while (length_squared(p) >= 1.0f);
    return p;
}

// generation via rejection sampling
__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1.0f);
    } while (length_squared(p) >= 1.0f);
    return p;
}

// cosine weighted direction
__device__ vec3 random_cosine_weighted_direction(curandState* local_rand_state) {
    float cos_theta_sq = curand_uniform(local_rand_state);
    float sin_theta = sqrtf(1.0f - cos_theta_sq);
    float cos_theta = sqrtf(cos_theta_sq);

    float phi = 2 * F_PI * curand_uniform(local_rand_state);

    return vec3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);
}


// orientate local vector to the normal
__host__ __device__ vec3 local_to_global(const vec3& local, const vec3& normal) {
    // create an arbritrary basis vector
    vec3 arbritrary_vec;
    // ensure that it is not in the same direction as normal
    if ((fabsf(normal.x) < fabsf(normal.y) && fabsf(normal.x) < fabsf(normal.z) || fabsf(normal.y) > 0.98f)) {
        arbritrary_vec = vec3(1.0f, 0.0f, 0.0f);
    }
    else {
        arbritrary_vec = vec3(0.0f, 1.0f, 0.0f);
    }

    // create tangent and bitangent direction vectors
    vec3 u = normalise(cross(arbritrary_vec, normal));
    vec3 v = normalise(cross(normal, u));

    // return the orientated vector
    return local.x * u + local.y * v + local.z * normal;
}

// orientate global vector to the normal ( usually (0, 1, 0) )
__host__ __device__ vec3 global_to_local(const vec3& global, const vec3& normal) {
    // create an arbritrary basis vector
    vec3 arbritrary_vec;
    // ensure that it is not in the same direction as normal
    if (fabsf(normal.x) < fabsf(normal.y) && fabsf(normal.x) < fabsf(normal.z) || fabsf(normal.y) > 0.98f) {
        arbritrary_vec = vec3(1.0f, 0.0f, 0.0f);
    }
    else {
        arbritrary_vec = vec3(0.0f, 1.0f, 0.0f);
    }

    // create tangent and bitangent direction vectors
    vec3 u = normalise(cross(arbritrary_vec, normal));
    vec3 v = normalise(cross(normal, u));

    // orientate
    return vec3(dot(u, global), dot(v, global), dot(normal, global));
}


__host__ __device__ inline float safe_sqrt(float a) {
    return a < 0 ? 0 : sqrtf(a);
}

__host__ __device__ inline bool is_any_nan(const vec3& v) {
    // check will only fail if nans or infs - any dodgy values
    return (v.x != v.x || v.y != v.y || v.z != v.z);
}