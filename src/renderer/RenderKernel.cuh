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

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

__host__ void render_frame(cudaGraphicsResource_t pbo_resource, int render_width, int render_height, int frame_number);
__host__ void init_scene(int render_width, int render_height);