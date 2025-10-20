#pragma once

#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "../math/vec3.cuh"
#include "../math/ray.cuh"
#include "Triangle.cuh"
#include "TriangleList.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "Scene.cuh"

#define white vec3(.82f)
#define red vec3(.73f,0.05f,0.05f)
#define green vec3(0.12f,.64f,0.15f)

#define RND (curand_uniform(&local_rand_state))



// hard coded function to generate the cornell box scene
// - based on data from https://www.graphics.cornell.edu/online/box/data.html
__global__ void create_cornell(Triangle* d_list, TriangleList* hit_list, Camera* d_camera, Scene* world, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int i = 0;

        d_list[i++] = Triangle(vec3(0.5528f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.5592f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.5528f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.5592f), vec3(0.5496f, 0.0f, 0.5592f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.343f, 0.54875f, 0.227f), vec3(0.343f, 0.54875f, 0.332f), vec3(0.213f, 0.54875f, 0.332f), new lambertian(vec3(1.0f), 15.0f, vec3(1.0f)));
        d_list[i++] = Triangle(vec3(0.343f, 0.54875f, 0.227f), vec3(0.213f, 0.54875f, 0.332f), vec3(0.213f, 0.54875f, 0.227f), new lambertian(vec3(1.0f), 15.0f, vec3(1.0f)));
        d_list[i++] = Triangle(vec3(0.556f, 0.5488f, 0.0f), vec3(0.556f, 0.5488f, 0.5592f), vec3(0.0f, 0.5488f, 0.5592f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.556f, 0.5488f, 0.0f), vec3(0.0f, 0.5488f, 0.5592f), vec3(0.0f, 0.5488f, 0.0f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.5496f, 0.0f, 0.5592f), vec3(0.0f, 0.0f, 0.5592f), vec3(0.0f, 0.5488f, 0.5592f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.5496f, 0.0f, 0.5592f), vec3(0.0f, 0.5488f, 0.5592f), vec3(0.556f, 0.5488f, 0.5592f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.0f, 0.0f, 0.5592f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.5488f, 0.0f), new lambertian(green));
        d_list[i++] = Triangle(vec3(0.0f, 0.0f, 0.5592f), vec3(0.0f, 0.5488f, 0.0f), vec3(0.0f, 0.5488f, 0.5592f), new lambertian(green));
        d_list[i++] = Triangle(vec3(0.5528f, 0.0f, 0.0f), vec3(0.5496f, 0.0f, 0.5592f), vec3(0.556f, 0.5488f, 0.5592f), new lambertian(red));
        d_list[i++] = Triangle(vec3(0.5528f, 0.0f, 0.0f), vec3(0.556f, 0.5488f, 0.5592f), vec3(0.556f, 0.5488f, 0.0f), new lambertian(red));
        d_list[i++] = Triangle(vec3(0.130f, 0.165f, 0.065f), vec3(0.082f, 0.165f, 0.225f), vec3(0.240f, 0.165f, 0.272f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.130f, 0.165f, 0.065f), vec3(0.240f, 0.165f, 0.272f), vec3(0.290f, 0.165f, 0.114f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.290f, 0.0f, 0.114f), vec3(0.290f, 0.165f, 0.114f), vec3(0.240f, 0.165f, 0.272f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.290f, 0.0f, 0.114f), vec3(0.240f, 0.165f, 0.272f), vec3(0.240f, 0.0f, 0.272f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.130f, 0.0f, 0.065f), vec3(0.130f, 0.165f, 0.065f), vec3(0.290f, 0.165f, 0.114f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.130f, 0.0f, 0.065f), vec3(0.290f, 0.165f, 0.114f), vec3(0.290f, 0.0f, 0.114f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.082f, 0.0f, 0.225f), vec3(0.082f, 0.165f, 0.225f), vec3(0.130f, 0.165f, 0.065f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.082f, 0.0f, 0.225f), vec3(0.130f, 0.165f, 0.065f), vec3(0.130f, 0.0f, 0.065f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.240f, 0.0f, 0.272f), vec3(0.240f, 0.165f, 0.272f), vec3(0.082f, 0.165f, 0.225f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.240f, 0.0f, 0.272f), vec3(0.082f, 0.165f, 0.225f), vec3(0.082f, 0.0f, 0.225f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.423f, 0.330f, 0.247f), vec3(0.265f, 0.330f, 0.296f), vec3(0.314f, 0.330f, 0.456f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.423f, 0.330f, 0.247f), vec3(0.314f, 0.330f, 0.456f), vec3(0.472f, 0.330f, 0.406f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.423f, 0.0f, 0.247f), vec3(0.423f, 0.330f, 0.247f), vec3(0.472f, 0.330f, 0.406f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.423f, 0.0f, 0.247f), vec3(0.472f, 0.330f, 0.406f), vec3(0.472f, 0.0f, 0.406f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.472f, 0.0f, 0.406f), vec3(0.472f, 0.330f, 0.406f), vec3(0.314f, 0.330f, 0.456f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.472f, 0.0f, 0.406f), vec3(0.314f, 0.330f, 0.456f), vec3(0.314f, 0.0f, 0.456f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.314f, 0.0f, 0.456f), vec3(0.314f, 0.330f, 0.456f), vec3(0.265f, 0.330f, 0.296f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.314f, 0.0f, 0.456f), vec3(0.265f, 0.330f, 0.296f), vec3(0.265f, 0.0f, 0.296f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.265f, 0.0f, 0.296f), vec3(0.265f, 0.330f, 0.296f), vec3(0.423f, 0.330f, 0.247f), new lambertian(white));
        d_list[i++] = Triangle(vec3(0.265f, 0.0f, 0.296f), vec3(0.423f, 0.330f, 0.247f), vec3(0.423f, 0.0f, 0.247f), new lambertian(white));

        new (hit_list) TriangleList(d_list, 32);

        vec3 lookfrom(0.278f, 0.273f, -0.800f);
        vec3 lookat(0.278f, 0.273f, -0.799f);

        float dist_to_focus = length(lookfrom - vec3(0.278f, 0.273f, 0.280f));

        float aperture = 0.000025f;

        *d_camera = new Camera(lookfrom,
            lookat,
            vec3(0.0f, 1.0f, 0.0f),
            40.0f,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);

        new (world) Scene(hit_list, d_camera);
    }
}

__global__ void free_scene_data_kernel(Triangle* d_list, int num_hitables) {
    // This kernel frees the memory for objects allocated with 'new' in create_cornell
    for (int i = 0; i < num_hitables; i++) {
        delete d_list[i].mat_ptr;
    }
}