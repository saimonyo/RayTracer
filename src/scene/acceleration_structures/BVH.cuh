#pragma once

#include "../Triangle.cuh"

// Bounding Volume Hierachy

__device__ inline float AABBIntersection(const vec3 v1, const vec3 v2, const ray& r) {
    // ray AABB intersection using ray slab intersection algorithm
    vec3 f = (v1 - r.origin) * r.inv_direction;
    vec3 n = (v2 - r.origin) * r.inv_direction;
    vec3 tmin = min(f, n);
    vec3 tmax = max(f, n);

    float t1 = min_component(tmax);
    float t0 = max_component(tmin);

    return (t1 >= t0) ? (t0 > 0.f ? t0 : t1) : FLT_MAX;
}

struct node {
    vec3 v1 = vec3(-FLT_MAX);
    vec3 v2 = vec3(FLT_MAX);
    int left_child = -1;
    int first_index = -1;
    uint32_t tri_count = 0;

    __host__ __device__ inline bool is_leaf() { return tri_count > 0; }
};


class BVH {
public:
    __device__ __host__ BVH() {}
    __host__ BVH(Triangle* tris, int n);
    __host__ void build();
    __host__ void divide(uint32_t idx);
    __host__ __device__ void refit_node(uint32_t idx);
    __device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ Triangle* get_random_emitter(curandState* local_rand_state) { return &triangles[0]; }

    Triangle* triangles;
    int triangle_count;

    uint32_t* indices;

    node* nodes;
    int node_count;

    uint32_t root_index = 0;
    uint32_t nodes_in_use = 1;
};



__host__ BVH::BVH(Triangle* tris, int n) {
    triangles = tris;
    triangle_count = n;
    node_count = 2 * n - 1;
    // building on CPU then moving to GPU
    // so this heap call is okay
    nodes = new node[node_count];
    indices = new uint32_t[n];

    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    build();
}

__host__ void BVH::build() {
    node& root = nodes[root_index];
    root.first_index = 0;
    root.tri_count = triangle_count;

    refit_node(root_index);
    divide(root_index);
}

__host__ __device__ void BVH::refit_node(uint32_t idx) {
    node& n = nodes[idx];
    n.v1 = vec3(FLT_MAX);
    n.v2 = vec3(-FLT_MAX);

    uint32_t first = n.first_index;
    for (int i = 0; i < n.tri_count; i++) {
        Triangle& tri = triangles[indices[first + i]];
        n.v1 = min(n.v1, tri.v1);
        n.v1 = min(n.v1, tri.v2);
        n.v1 = min(n.v1, tri.v3);

        n.v2 = max(n.v2, tri.v1);
        n.v2 = max(n.v2, tri.v2);
        n.v2 = max(n.v2, tri.v3);
    }
}

__host__ void BVH::divide(uint32_t idx) {
    node& n = nodes[idx];
    if (n.tri_count <= 2) {
        return;
    }

    vec3 dimensions = n.v2 - n.v1;

    int axis = 0;

    // split using the largest axis
    if (dimensions.y > dimensions.x) {
        axis = 1;
    }
    if (dimensions.z > dimensions[axis]) {
        axis = 2;
    }

    float split = n.v1[axis] + dimensions[axis] * 0.5f;

    uint32_t i = n.first_index;
    uint32_t j = i + n.tri_count - 1;

    while (i <= j) {
        if (triangles[indices[i]].centroid[axis] < split) {
            i++;
        }
        else {
            std::swap(indices[i], indices[j]);
            j--;
        }
    }

    int left_count = i - n.first_index;
    if (left_count == 0 || left_count == n.tri_count) {
        return;
    }

    uint32_t left_idx = nodes_in_use++;
    uint32_t right_idx = nodes_in_use++;

    nodes[left_idx].first_index = n.first_index;
    nodes[left_idx].tri_count = left_count;
    nodes[right_idx].first_index = i;
    nodes[right_idx].tri_count = n.tri_count - left_count;
    n.left_child = left_idx;
    n.tri_count = 0;
    refit_node(left_idx);
    refit_node(right_idx);

    divide(left_idx);
    divide(right_idx);
}

__device__ bool BVH::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
    float t = AABBIntersection(nodes[root_index].v1, nodes[root_index].v2, r);

    if (t == FLT_MAX) {
        return false;
    }

    float closest_so_far = tmax;
    hit_record temp_rec;

    node* n = &nodes[root_index];
    node* stack[200];

    uint32_t stack_ptr = 0;

    bool intersected = false;
    
    while (true) {
        if (n->is_leaf()) {
            uint32_t first = n->first_index;

            for (uint32_t i = 0; i < n->tri_count; i++) {
                Triangle& tri = triangles[indices[first + i]];
                if (tri.hit(r, tmin, closest_so_far, temp_rec)) {
                    intersected = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                    rec.mat_ptr = &tri.mat;
                }
            }

            if (stack_ptr == 0) {
                break;
            }
            else {
                n = stack[--stack_ptr];
            }

            continue;
        }

        node* left = &nodes[n->left_child];
        node* right = &nodes[n->left_child + 1];

        float t1 = AABBIntersection(left->v1, left->v2, r);
        float t2 = AABBIntersection(right->v1, right->v2, r);

        if (t1 <= t2) {
            if (t1 == FLT_MAX || t1 > closest_so_far) {
                if (stack_ptr == 0) {
                    break;
                }
                else {
                    n = stack[--stack_ptr];
                }
            }
            else {
                n = left;
                if (t2 != FLT_MAX && t2 < closest_so_far) {
                    stack[stack_ptr++] = right;
                }
            }
        }
        else {
            if (t2 == FLT_MAX || t2 > closest_so_far) {
                if (stack_ptr == 0) {
                    break;
                }
                else {
                    n = stack[--stack_ptr];
                }
            }
            else {
                n = right;
                if (t1 != FLT_MAX && t1 < closest_so_far) {
                    stack[stack_ptr++] = left;
                }
            }
        }
    }
    return intersected;
}