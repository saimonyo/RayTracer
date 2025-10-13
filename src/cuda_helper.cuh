#pragma once

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#include "cuda_helper.cuh"

inline void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result != cudaSuccess) {
		std::cerr << "CUDA error:" << static_cast<unsigned int>(result)
			<< " at " << file << ":" << line
			<< " - for call of '" << func << "'" << std::endl;

		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}