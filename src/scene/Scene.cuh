#pragma once

#include "PrimitiveList.cuh"
#include "Camera.cuh"


class Scene {
public:
    __device__ __host__ Scene() {}
    __device__ __host__ Scene(PrimitiveList* prims, Camera* cam) : primitives(prims), camera(cam) {}

    PrimitiveList* primitives;
    Camera* camera;
};


