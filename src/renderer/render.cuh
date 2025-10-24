#pragma once

#include "../scene/setup.cuh"

namespace Sample {

    __device__ vec3 naive_monte_carlo(const ray& r, BVH* world, curandState* local_rand_state) {
        ray cur_ray = r;
        vec3 radiance = 0.0f;
        vec3 throughput = 1.0f;


        // limit maximum amount of bounces
        for (int i = 0; i <= 50; i++) {
            hit_record rec;
            // check for any intersections
            if (world->hit(cur_ray, EPSILON, FLT_MAX, rec)) {
                // hit so get material of object
                Material* mat = rec.mat_ptr;
                // if its emissive get the value of the emission
                // - add this to the current radiance
                vec3 emitted_light = mat->emission_colour * mat->emission_strength;
                radiance += emitted_light * throughput;

                // the resultant scattered ray
                ray scattered;
                float pdf;

                if (rec.mat_ptr->sample(rec, cur_ray, throughput, scattered, pdf, local_rand_state)) {
                    cur_ray = scattered;
                }

                // russian roulette
                float one_minus_p = max_component(throughput);

                if (curand_uniform(local_rand_state) > one_minus_p) {
                    break;
                }

                // need to reweight due to potential termination
                throughput = throughput / one_minus_p;
            }
            else {
                break;
            }
        }
        return radiance;
    }

    __device__ vec3 NEE_monte_carlo(const ray& r, BVH* world, curandState* local_rand_state) {
        ray cur_ray = r;
        vec3 radiance = 0.0f;
        vec3 throughput = 1.0f;


        // limit maximum amount of bounces
        for (int i = 0; i <= 50; i++) {
            hit_record rec;
            // check for any intersections
            if (!world->hit(cur_ray, EPSILON, FLT_MAX, rec)) {
                break;
            }

            // hit so get material of object
            Material* mat = rec.mat_ptr;
                
            // hit an emissive object, double counting unless first hit (i==0)
            if (mat->emission_strength > 0.0f) {
                if (i == 0) {
                    radiance = throughput * mat->emission_colour * mat->emission_strength;
                }
                return radiance;
            }

            // sample a random emitter to sample
            Triangle* random_emitter = world->get_random_emitter(local_rand_state);
            vec3 random_point_on_emitter = random_emitter->sample_random_point(local_rand_state);

            float emitter_area = random_emitter->area();

            vec3 to_light = random_point_on_emitter - rec.p;
            float distance_to_light_squared = length_squared(to_light);
            float distance_to_light = sqrtf(distance_to_light_squared);

            // normalise
            to_light = to_light / distance_to_light;

            vec3 light_normal = random_emitter->normal(random_point_on_emitter);

            float cos_o = -dot(to_light, light_normal);
            float cos_i = dot(to_light, rec.normal);


            // the emitter can possibly contribute light
            if (cos_o > 0.0f && cos_i > 0.0f) {
                ray shadow_ray = ray(rec.p, to_light);
                hit_record shadow_rec;

                // check if we didnt hit anything on the way to the light
                if (!world->hit(shadow_ray, EPSILON, distance_to_light - 0.001f, shadow_rec)) {
                    float solid_angle = (cos_o * emitter_area) / distance_to_light_squared;

                    float pdf;
                    vec3 brdf;
                    
                    // calculate how much light is received and what colour
                    if (mat->eval(rec, to_light, -cur_ray.direction, pdf, brdf, local_rand_state)) {
                        Material light_mat = random_emitter->mat;
                        vec3 emitted_colour = light_mat.emission_colour * light_mat.emission_strength;

                        radiance += throughput * brdf * emitted_colour * solid_angle;
                    }
                }
            }

            // the resultant scattered ray
            ray scattered;
            float pdf;

            if (rec.mat_ptr->sample(rec, cur_ray, throughput, scattered, pdf, local_rand_state)) {
                cur_ray = scattered;
            }
            else {
                break;
            }

            // russian roulette only after a certain amount of bounces
            if (i >= 4) {
                float one_minus_p = max_component(throughput);

                if (curand_uniform(local_rand_state) > one_minus_p) {
                    break;
                }

                // need to reweight due to potential termination
                throughput = throughput / one_minus_p;
            }
        }
        return radiance;
    }
}