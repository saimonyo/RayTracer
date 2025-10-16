#pragma once

#include "Primitive.cuh"
#include "Material.cuh"


class PrimitiveList {
public:
    __device__ PrimitiveList() {}
    __device__ PrimitiveList(Primitive** l, int n);
    __device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __host__ __device__ inline Primitive* operator[](int i) const { return list[i]; }
    __device__ ~PrimitiveList() { delete[] list; }

    Primitive** list;
    int list_size;
    int emitters_size;
};

__device__ PrimitiveList::PrimitiveList(Primitive** l, int n) {
    list = l;
    list_size = n;
    Primitive** temp = new Primitive * [n];
    emitters_size = 0;
}

__device__ bool PrimitiveList::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
            rec.mat_ptr = list[i]->mat_ptr;
        }
    }
    return hit_anything;
}