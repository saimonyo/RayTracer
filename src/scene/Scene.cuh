#pragma once

#include "acceleration_structures/TriangleList.cuh"
#include "acceleration_structures/BVH.cuh"
#include "Camera.cuh"


class Scene {
public:
    __device__ __host__ Scene() {}
    __device__ __host__ Scene(BVH* prims, Camera* cam) : primitives(prims), camera(cam) {}

    BVH* primitives;
    Camera* camera;
};


