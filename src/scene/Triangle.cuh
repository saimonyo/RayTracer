#pragma once

#include "Material.cuh"

#define EPSILON 0.000001f

class Triangle{
public:
    vec3 v1, v2, v3;
    vec3 e1, e2;
    vec3 centroid;
    vec3 n;
    Material mat;

    __host__ __device__ Triangle() {}
    __host__ __device__ Triangle(vec3 v_1, vec3 v_2, vec3 v_3, Material m) : v1(v_1), v2(v_2), v3(v_3), mat(m) {
        e1 = v2 - v1;
        e2 = v3 - v1;
        n = normalise(cross(e1, e2));
        centroid = (v1 + v2 + v3) / 3.0f;
    };
    __device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ vec3 normal(vec3 point) const;
    __device__ float area() const;
    __device__ vec3 sample_random_point(curandState* local_rand_state) const;
};


// Möller–Trumbore intersection algorithm
// - based on https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ bool Triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 ray_cross_e2 = cross(r.direction, e2);
    float det = dot(e1, ray_cross_e2);

    //
    // possibly needs a check for det close to eps
    //

    float inv_det = 1.0f / det;
    vec3 s = r.origin - v1;
    float u = inv_det * dot(s, ray_cross_e2);

    // barycentric coord out of range
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    vec3 s_cross_e1 = cross(s, e1);
    float v = inv_det * dot(r.direction, s_cross_e1);

    // barycentric coord out of range, or combination
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    float t = inv_det * dot(e2, s_cross_e1);

    // timestep of intersection is out of valid range
    if (t < t_min || t > t_max) {
        return false;
    }

    rec.t = t;
    rec.p = r.at(rec.t);
    rec.normal = n;
    return true;
}

__device__ vec3 Triangle::normal(vec3 point) const {
    return n;
}

__device__ float Triangle::area() const {
    return 0.5f * length(cross(e1, e2));
}

__device__ vec3 Triangle::sample_random_point(curandState* local_rand_state) const {
    float u = curand_uniform(local_rand_state);
    float v = curand_uniform(local_rand_state);

    if (u + v >= 1) {
        u = 1 - u;
        v = 1 - v;
    }

    return v1 + u * e1 + v * e2;
}