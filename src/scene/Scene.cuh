#pragma once

#include "TriangleList.cuh"
#include "Camera.cuh"


class Scene {
public:
    __device__ __host__ Scene() {}
    __device__ __host__ Scene(TriangleList* prims, Camera* cam) : primitives(prims), camera(cam) {}

    TriangleList* primitives;
    Camera* camera;
};


