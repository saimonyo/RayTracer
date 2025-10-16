#pragma once

#include "Triangle.cuh"

#define EPSILON 0.000001f

class Quad : public Primitive {
public:
    Triangle a, b;
    vec3 e1, e2;
    vec3 v1;
    vec3 n;

    __device__ Quad() {}
    __device__ Quad(vec3 v_1, vec3 v_2, vec3 v_3, vec3 v_4, Material* m) :
        a(v_1, v_2, v_3, nullptr), b(v_1, v_3, v_4, nullptr), v1(v_1), Primitive(m) {
        e1 = v_2 - v_1;
        e2 = v_4 - v_1;
        n = normalise(cross(e1, e2));
    };
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual vec3 normal(vec3 point) const;
    __device__ virtual float area() const;
    __device__ virtual vec3 sample_random_point(curandState* local_rand_state) const;
};

__device__ bool Quad::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    return a.hit(r, t_min, t_max, rec) || b.hit(r, t_min, t_max, rec);
}

__device__ vec3 Quad::normal(vec3 point) const {
    return n;
}

__device__ float Quad::area() const {
    return length(cross(e1, e2));
}

__device__ vec3 Quad::sample_random_point(curandState* local_rand_state) const {
    float u = curand_uniform(local_rand_state);
    float v = curand_uniform(local_rand_state);

    return v1 + u * e1 + v * e2;
}