#pragma once

#include "../scene/setup.cuh"

namespace Sample {

    __device__ vec3 naive_monte_carlo(const ray& r, PrimitiveList** world, curandState* local_rand_state) {
        ray cur_ray = r;
        vec3 radiance = 0.0f;
        vec3 throughput = 1.0f;


        // limit maximum amount of bounces
        for (int i = 0; i <= 50; i++) {
            hit_record rec;
            // check for any intersections
            if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
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
                float p = max_component(throughput);

                if (curand_uniform(local_rand_state) >= p) {
                    break;
                }

                // need to reweight due to potential termination
                throughput = throughput / p;
            }
            else {
                break;
            }
        }
        return radiance;
    }

}