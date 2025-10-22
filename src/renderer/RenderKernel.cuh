#pragma once

#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h> 
#include <device_launch_parameters.h>

#include <cstdio>
#include <cmath>
#include <ctime>
#include <iostream>


// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

struct vec3;

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);
void update_camera_location(const vec3& movement_offset, int width, int height);
void update_camera_rotation(float yaw, float pitch, int width, int height);


__host__ void render_frame(cudaGraphicsResource_t pbo_resource, int render_width, int render_height, int frame_number);
__host__ void init_cornell_scene(int render_width, int render_height);
__host__ void init_model_scene(int render_width, int render_height, vec3* vertices, unsigned int* indices, size_t indices_count, size_t vertices_count, size_t triangle_count);