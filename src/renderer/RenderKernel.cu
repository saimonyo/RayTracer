#include "RenderKernel.cuh"
#include "render.cuh"
#include "../scene/Scene.cuh"
#include "../math/Quaternion.cuh"

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

extern Scene* d_world;
extern vec3* d_accumulation_buffer;

void update_camera_location(const vec3& movement_offset, int width, int height) {
    Scene h_scene;
    checkCudaErrors(cudaMemcpy(&h_scene, d_world, sizeof(Scene), cudaMemcpyDeviceToHost));

    Camera* d_camera = h_scene.camera;
    Camera h_camera;
    checkCudaErrors(cudaMemcpy(&h_camera, d_camera, sizeof(Camera), cudaMemcpyDeviceToHost));


    vec3 world_space_offset = h_camera.local_right * movement_offset.x +
        h_camera.local_up * movement_offset.y +
        h_camera.local_forward * movement_offset.z;

    h_camera.location += world_space_offset;
    h_camera.lower_left_corner += world_space_offset;

    checkCudaErrors(cudaMemcpy(d_camera, &h_camera, sizeof(Camera), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_accumulation_buffer, 0, width * height * sizeof(vec3)));
}

void update_camera_rotation(float yaw, float pitch, int width, int height) {
    Scene h_scene;
    checkCudaErrors(cudaMemcpy(&h_scene, d_world, sizeof(Scene), cudaMemcpyDeviceToHost));

    Camera* d_camera = h_scene.camera;
    Camera h_camera;
    checkCudaErrors(cudaMemcpy(&h_camera, d_camera, sizeof(Camera), cudaMemcpyDeviceToHost));

    if (h_camera.total_pitch + pitch > F_PI / 2 || h_camera.total_pitch + pitch < -F_PI / 2) {
        pitch = 0;
    }

    Quaternion pitch_quaternion = quaternion_axis_angle(h_camera.local_right, pitch);
    Quaternion yaw_quaternion = quaternion_axis_angle(h_camera.world_up, yaw);

    h_camera.local_forward = yaw_quaternion * h_camera.local_forward;
    h_camera.local_right = yaw_quaternion * h_camera.local_right;
    h_camera.local_up = yaw_quaternion * h_camera.local_up;

    
    h_camera.local_up = pitch_quaternion * h_camera.local_up;
    h_camera.local_forward = pitch_quaternion * h_camera.local_forward;

    h_camera.total_pitch += pitch;

   

    h_camera.update_plane();


    checkCudaErrors(cudaMemcpy(d_camera, &h_camera, sizeof(Camera), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_accumulation_buffer, 0, width * height * sizeof(vec3)));
}

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

#define pink vec3(1.0f, 0.01f, 0.9f)

__global__ void render_init(int render_width, int render_height, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= render_width) || (j >= render_height)) {
        return;
    }

    int pixel_index = j * render_width + i;
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

constexpr float one_over_gamma = 1.0f / 2.2f;

__global__ void render(unsigned char* ptr, int render_width, int render_height, Scene* world, curandState* rand_state, int frame_number, vec3* accum_buffer) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= render_width) || (j >= render_height)) {
        return;
    }

    int pixel_index = j * render_width + i;

    curandState local_rand_state = rand_state[pixel_index];

    float u = float(i + curand_uniform(&local_rand_state)) / float(render_width);
    float v = float(j + curand_uniform(&local_rand_state)) / float(render_height);
    Camera* cam = world->camera;
    TriangleList* prims = world->primitives;

    ray r = cam->get_ray(u, v, &local_rand_state);
    vec3 new_sample = Sample::NEE_monte_carlo(r, prims, &local_rand_state);

    // get old value in the buffer
    vec3 old_render = accum_buffer[pixel_index];
    float weight = 1.0f / (frame_number + 1);

    // add the new sample on
    vec3 new_render = old_render * (1.0f - weight) + new_sample * weight;
    accum_buffer[pixel_index] = new_render;

    // Update random state
    rand_state[pixel_index] = local_rand_state;

    vec3 final_colour;
    if (is_any_nan(new_render)) {
        final_colour = pink;
    }
    else {
        // set the gamma corrected colour in the pbo
        final_colour = pow(new_render, one_over_gamma);
        final_colour = clamp(final_colour, 0.0f, 1.0f);
    }

    int pbo_index = pixel_index * 4;
    ptr[pbo_index + 0] = (unsigned char)(255.99f * final_colour.x);
    ptr[pbo_index + 1] = (unsigned char)(255.99f * final_colour.y);
    ptr[pbo_index + 2] = (unsigned char)(255.99f * final_colour.z);
    ptr[pbo_index + 3] = 255;
}

curandState* d_rand_state;
TriangleList* hit_list;
Triangle** d_list;
Camera* d_camera;
vec3* d_accumulation_buffer;
Scene* d_world;
int num_hitables = 32;

__host__ void init_scene(int render_width, int render_height) {
    int num_pixels = render_width * render_height;

    // Allocate random states for pixels
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    // Allocate random state for world generation
    curandState* d_rand_state_world;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state_world, sizeof(curandState)));
    rand_init << <1, 1 >> > (d_rand_state_world);

    checkCudaErrors(cudaMalloc((void**)&d_accumulation_buffer, num_pixels * sizeof(vec3)));
    checkCudaErrors(cudaMemset(d_accumulation_buffer, 0, num_pixels * sizeof(vec3)));


    // Make our world of hitables & the camera
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(Triangle*)));
    checkCudaErrors(cudaMalloc((void**)&hit_list, sizeof(TriangleList)));
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Scene)));

    create_cornell << <1, 1 >> > (d_list, hit_list, d_camera, d_world, render_width, render_height, d_rand_state_world);

    // Initialize per-pixel random states
    dim3 blocks(render_width / 16 + 1, render_height / 16 + 1);
    dim3 threads(16, 16);
    render_init << <blocks, threads >> > (render_width, render_height, d_rand_state);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_rand_state_world));
}


__host__ void render_frame(cudaGraphicsResource_t pbo_resource, int render_width, int render_height, int frame_number) {
    unsigned char* devPtr = 0;
    size_t size = render_width * render_height * 4 * sizeof(char);

    // Map the PBO for writing from CUDA
    checkCudaErrors(cudaGraphicsMapResources(1, &pbo_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, pbo_resource));

    // Render the scene
    dim3 blocks(render_width / 16 + 1, render_height / 16 + 1);
    dim3 threads(16, 16);
    render << <blocks, threads >> > (devPtr, render_width, render_height, d_world, d_rand_state, frame_number, d_accumulation_buffer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Unmap the PBO
    checkCudaErrors(cudaGraphicsUnmapResources(1, &pbo_resource, 0));
}


__host__ void cleanup_scene() {
    checkCudaErrors(cudaDeviceSynchronize());

    free_scene_data_kernel << <1, 1 >> > (d_list, num_hitables);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Free the top-level pointers.
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_accumulation_buffer));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(hit_list));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
}