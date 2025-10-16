#pragma once

#define GLEW_STATIC

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"


// OpenGL headers
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// CUDA interop with OpenGL 
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <cstdio>
#include <functional>

#include "renderer/RenderKernel.cuh"



class Window {
private:
	const int VIEWPORT_WIDTH;
	const int VIEWPORT_HEIGHT;
	const int MAINWINDOW_WIDTH;
	const int MAINWINDOW_HEIGHT;
	// pixel buffer object
	GLuint pbo;
	GLuint texture;
	cudaGraphicsResource_t cuda_pbo;
	GLFWwindow* window;

	//imgui
	ImGuiIO io;
	float display_fps = 0.0f;
	float display_ms = 0.0f;
	double last_update_time = -1.0;
	int prev_ray_count = 0;
	float mrays = 0;

	int frame_number = 0;

	bool init_CUDA();
	bool init_GL();
	bool init_imgui();
	bool setup_buffers();
	bool initialise();

	void Window::each_frame_pre_kernel();
	void Window::each_frame_post_kernel();

	void clean_up();


public:
	Window(const int vp_width, const int vp_height, const int mw_width, const int mw_height);
	~Window();
	void main_loop(std::function<void(cudaGraphicsResource_t, int, int, int)> kernel);
};