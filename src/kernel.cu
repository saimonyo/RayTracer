#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h> 
#include <device_launch_parameters.h>
#include "cuda_helper.cuh"

#include <cstdio>
#include <cmath>
#include <ctime>


__global__ void render(unsigned char* ptr, int width, int height, int frame_number) {
    // placeholder/test code for checking viewport renders correctly
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) {
        return;
    }
    int pbo_index = (j * width + i) * 4;

    float x;
    if (i < width / 2) {
        x = 1.0f * ((i + frame_number) % width) / width;
    }
    else {
        x = 1.0f - (1.0f * ((i + frame_number) % width) / width);
    }


    float y = (1.0f * ((j) % width) / height);


    ptr[pbo_index + 0] = (unsigned char)(x * 255);
    ptr[pbo_index + 1] = (unsigned char)(y * 255);
    ptr[pbo_index + 2] = (unsigned char)(255);
    ptr[pbo_index + 3] = (unsigned char)(255);
}


__host__ void render_frame(cudaGraphicsResource_t pbo_resource, int width, int height, int frame_number) {
    unsigned char* devPtr = 0;
    size_t size = width * height * 4 * sizeof(char);

    // map the pbo for writing from CUDA
    checkCudaErrors(cudaGraphicsMapResources(1, &pbo_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, pbo_resource));

    // render
    dim3 blocks(width / 16 + 1, height / 16 + 1);
    dim3 threads(16, 16);
    render<<<blocks, threads>>>(devPtr, width, height, frame_number);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // unmap the PBO
    checkCudaErrors(cudaGraphicsUnmapResources(1, &pbo_resource, 0));
}