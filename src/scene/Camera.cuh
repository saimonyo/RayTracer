#pragma once

#include <curand_kernel.h>
#include "../math/ray.cuh"


class Camera {
public:
    vec3 location;
    // corresponding to the local orientation of the camera
    vec3 u, v, w;
    // properties of the focal plane
    vec3 lower_left_corner;
    //  - the vectors defining the edges
    vec3 horizontal;
    vec3 vertical;
    float lens_radius;

    __host__ __device__ Camera() {}
    __host__ __device__ Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist);
    __host__ __device__ Camera(Camera* cam);
    __device__ ray get_ray(float s, float t, curandState* local_rand_state);
};

__device__ Camera::Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) {
    lens_radius = aperture / 2.0f;

    // convert fov to radians
    float theta = vfov * ((float)F_PI) / 180.0f;
    float half_height = tan(theta / 2.0f);
    float half_width = aspect * half_height;

    location = lookfrom;

    // generate the local axis
    w = normalise(lookfrom - lookat);
    u = normalise(cross(vup, w));
    v = normalise(cross(w, u));

    // calculate the bottom left of the focal plane
    lower_left_corner = location - half_width * focus_dist * u
        - half_height * focus_dist * v - focus_dist * w;

    horizontal = 2.0f * half_width * focus_dist * u;
    vertical = 2.0f * half_height * focus_dist * v;
}

__host__ __device__ Camera::Camera(Camera* cam) {
    location = cam->location;

    u = cam->u;
    v = cam->v;
    w = cam->w;

    lower_left_corner = cam->lower_left_corner;
    horizontal = cam->horizontal;
    vertical = cam->vertical;

    lens_radius = cam->lens_radius;
}

__device__ ray Camera::get_ray(float s, float t, curandState* local_rand_state) {
    // create tiny offset
    vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
    vec3 offset = u * rd.x + v * rd.y;

    vec3 target = lower_left_corner + s * horizontal + t * vertical;

    return ray(location + offset, target - (location + offset));
}