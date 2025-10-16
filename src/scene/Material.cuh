#pragma once

struct hit_record;

#include "../math/ray.cuh"
#include "Primitive.cuh"


__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted);
__device__ vec3 reflect(const vec3& v, const vec3& n);



class Material {
public:
    vec3 albedo;
    float emission_strength;
    vec3 emission_colour;

    __device__ Material(const vec3& a, float em_s, const vec3& em_c) : albedo(a), emission_strength(em_s), emission_colour(em_c) {}
    __device__ virtual bool sample(const hit_record& rec, const ray& r_in, vec3& throughput, ray& scattered, float& pdf, curandState* local_rand_state) const = 0;
    __device__ virtual bool eval(const hit_record& rec, const vec3& w_i, const vec3& w_o, float& pdf, vec3& brdf, curandState* local_rand_state) const = 0;
};


class lambertian : public Material {
public:
    __device__ lambertian(const vec3& a) : Material(a, 0.0f, vec3(0.0f)) {}
    __device__ lambertian(const vec3& a, float em_s, const vec3& em_c) : Material(a, em_s, em_c) {}
    __device__ virtual bool sample(const hit_record& rec, const ray& r_in, vec3& throughput, ray& scattered, float& pdf, curandState* local_rand_state) const {
        vec3 direction = random_cosine_weighted_direction(local_rand_state);
        scattered = ray(rec.p, local_to_global(direction, rec.normal));
        pdf = direction.z * ONE_OVER_PI;
        throughput *= albedo;
        return true;
    }

    __device__ virtual bool eval(const hit_record& rec, const vec3& w_i, const vec3& w_o, float& pdf, vec3& brdf, curandState* local_rand_state) const {
        float cos_i = dot(w_i, rec.normal);
        if (cos_i < 0.0f) {
            return false;
        }
        brdf = albedo * cos_i * ONE_OVER_PI;
        pdf = cos_i * ONE_OVER_PI;
        return true;
    }
};

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = normalise(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}