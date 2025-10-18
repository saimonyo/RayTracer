#pragma once

#include <curand_kernel.h>
#include "../math/ray.cuh"


class Camera {
public:
    vec3 location;
    // corresponding to the local orientation of the camera
    vec3 local_up, local_right, local_forward;
    // properties of the focal plane
    vec3 lower_left_corner;
    //  - the vectors defining the edges
    vec3 horizontal;
    vec3 vertical;
    float lens_radius;

    float theta;
    float focal_length;
    float half_width;
    float half_height;

    float total_pitch = 0.0;
    vec3 world_up;

    __host__ __device__ Camera() {}
    __host__ __device__ Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focal_length);
    __host__ __device__ Camera(Camera* cam);
    __host__ __device__ void update_plane();
    __device__ ray get_ray(float s, float t, curandState* local_rand_state);
};

__device__ Camera::Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focal_length) {
    lens_radius = aperture / 2.0f;
    this->focal_length = focal_length;

    // convert fov to radians
    this->theta = vfov * ((float)F_PI) / 180.0f;
    this->half_height = tan(theta / 2.0f);
    this->half_width = aspect * half_height;

    location = lookfrom;

    // generate the local axis
    local_forward = normalise(lookfrom - lookat);
    local_right = normalise(cross(vup, local_forward));
    local_up = vup;

    world_up = vup;

    update_plane();
}

__host__ __device__ void Camera::update_plane() {
    // calculate the bottom left of the focal plane using current u, v, w
    lower_left_corner = location - half_width * focal_length * local_right
        - half_height * focal_length * local_up - focal_length * local_forward;

    horizontal = 2.0f * half_width * focal_length * local_right;
    vertical = 2.0f * half_height * focal_length * local_up;
}

__host__ __device__ Camera::Camera(Camera* cam) {
    location = cam->location;

    local_up = cam->local_up;
    local_right = cam->local_right;
    local_forward = cam->local_forward;

    lower_left_corner = cam->lower_left_corner;
    horizontal = cam->horizontal;
    vertical = cam->vertical;

    lens_radius = cam->lens_radius;
    theta = cam->theta;
    focal_length = cam->focal_length;
    half_width = cam->half_width;
    half_height = cam->half_height;
    total_pitch = cam->total_pitch;

    world_up = cam->world_up;
}

__device__ ray Camera::get_ray(float s, float t, curandState* local_rand_state) {
    // create tiny offset
    vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
    vec3 offset = local_right * rd.x + local_up * rd.y;

    vec3 target = lower_left_corner + s * horizontal + t * vertical;

    return ray(location + offset, target - (location + offset));
}