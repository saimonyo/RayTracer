#pragma once

#include "../math/ray.cuh"

class Material;
class Primitive;

struct hit_record {
    float t;
    vec3 p;
    vec3 normal;
    Material* mat_ptr = nullptr;
};

class Primitive {
public:
    Material* mat_ptr = nullptr;

    __device__ __host__ Primitive() {}
    __device__ __host__ Primitive(Material* ptr) : mat_ptr(ptr) {}
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    __device__ virtual vec3 normal(vec3 point) const = 0;
    __device__ virtual float area() const = 0;
    __device__ virtual vec3 sample_random_point(curandState* local_rand_state) const = 0;
};