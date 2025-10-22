#pragma once

#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "../math/vec3.cuh"
#include "../math/ray.cuh"
#include "Triangle.cuh"
#include "acceleration_structures/TriangleList.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "Scene.cuh"

#define white vec3(.82f)
#define red vec3(.73f,0.05f,0.05f)
#define green vec3(0.12f,.64f,0.15f)

#define RND (curand_uniform(&local_rand_state))




__global__ void create_model_scene(Camera* d_camera, Scene* world, int nx, int ny, curandState* rand_state, BVH* d_bvh) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {

        Material* shiny_mat = new lambertian(vec3(1.0f), 15.0f, vec3(1.0f));

        for (int i = 0; i < d_bvh->triangle_count; i++) {
            d_bvh->triangles[i].mat_ptr = shiny_mat;
        }

        vec3 lookfrom(0.0f, 0.0f, 0.0f);
        vec3 lookat(0.0f, 0.0f, 1.0f);

        float dist_to_focus = 1;

        float aperture = 0.000025f;

        new (d_camera) Camera(lookfrom,
            lookat,
            vec3(0.0f, 1.0f, 0.0f),
            40.0f,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);

        new (world) Scene(d_bvh, d_camera);
    }
}

__global__ void free_scene_data_kernel(Triangle* d_list, int num_hitables) {
    // This kernel frees the memory for objects allocated with 'new' in create_cornell
    for (int i = 0; i < num_hitables; i++) {
        delete d_list[i].mat_ptr;
    }
}