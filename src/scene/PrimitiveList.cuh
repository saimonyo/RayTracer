#pragma once

#include "Primitive.cuh"
#include "Material.cuh"


class PrimitiveList {
public:
    __device__ PrimitiveList() {}
    __device__ PrimitiveList(Primitive** l, int n);
    __device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __host__ __device__ inline Primitive* operator[](int i) const { return list[i]; }
    __device__ Primitive* get_random_emitter(curandState* local_rand_state);
    __device__ ~PrimitiveList() { delete[] emitters; delete[] list; }

    Primitive** list;
    int list_size;
    Primitive** emitters;
    int emitters_size;
};

__device__ PrimitiveList::PrimitiveList(Primitive** l, int n) {
    list = l;
    list_size = n;
    Primitive** temp = new Primitive * [n];
    emitters_size = 0;

    for (int i = 0; i < n; i++) {
        Material* mat = list[i]->mat_ptr;
        if (mat != nullptr && mat->emission_strength > 0.0f) {
            temp[emitters_size++] = list[i];
        }
    }

    emitters = new Primitive* [emitters_size];
    for (int i = 0; i < emitters_size; i++) {
        emitters[i] = temp[i];
    }
    delete[] temp;
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

__device__ Primitive* PrimitiveList::get_random_emitter(curandState* local_rand_state) {
    assert(emitters_size > 0);
    int rand_idx = curand(local_rand_state) % emitters_size;
    return emitters[rand_idx];
}