#include "Window.h"
#include <device_launch_parameters.h>

static auto stationary_window_flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

int get_ray_count();

bool Window::init_CUDA() {
    int device_id = -1;
    unsigned int device_count = 0;
    int devices[1];

    checkCudaErrors(cudaGLGetDevices(&device_count, devices, 1, cudaGLDeviceListAll));

    if (device_count == 0) {
        std::cerr << "Failed to find CUDA compatible GPU" << std::endl;
        return false;
    }

    device_id = devices[0];

    checkCudaErrors(cudaGLSetGLDevice(device_id));

    init_scene(VIEWPORT_WIDTH, VIEWPORT_HEIGHT);

    return true;
}

bool Window::init_GL() {
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialise GLEW" << std::endl;
        return false;
    }

    glViewport(0, 0, MAINWINDOW_WIDTH, MAINWINDOW_HEIGHT);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glDisable(GL_DEPTH_TEST);

    return true;
}

bool Window::init_imgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    return true;
}

bool Window::setup_buffers() {
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // allocate texture memory
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    // set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // create the pixel buffer object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    // allocate memory to the pixel buffer
    glBufferData(GL_PIXEL_UNPACK_BUFFER, VIEWPORT_WIDTH * VIEWPORT_HEIGHT * sizeof(char) * 4, NULL, GL_DYNAMIC_DRAW);

    // unbind buffer to registering with CUDA
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // register pbo with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard));

    return true;
}


bool Window::initialise() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }


    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(MAINWINDOW_WIDTH, MAINWINDOW_HEIGHT, "Ray Tracing", NULL, NULL);

    if (!window) {
        std::cerr << "Failed to instantiate a GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialise other parts required by Window
    if (!init_GL()) {
        return false;
    }
    if (!init_CUDA()) {
        return false;
    }
    if (!setup_buffers()) {
        return false;
    }
    if (!init_imgui()) {
        return false;
    }

    return true;
}

Window::Window(const int vp_width, const int vp_height, const int mw_width, const int mw_height)
    : VIEWPORT_WIDTH(vp_width), VIEWPORT_HEIGHT(vp_height), MAINWINDOW_WIDTH(mw_width), MAINWINDOW_HEIGHT(mw_height) {
    if (!initialise()) {
        exit(-1);
    }
}


void Window::each_frame_pre_kernel() {
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
}

void Window::each_frame_post_kernel() {
    // copy data from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // imgui
    {
        //
        // Render Viewport
        // 
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

        ImGui::SetNextWindowSize(ImVec2(VIEWPORT_WIDTH, VIEWPORT_HEIGHT));
        ImGui::SetNextWindowPos(ImVec2(0, 0));

        bool isopen = true;
        ImGui::Begin("Viewport", &isopen, stationary_window_flags | ImGuiWindowFlags_NoTitleBar);

        // Your ImGui::Image call remains the same
        ImVec2 viewportSize = ImGui::GetContentRegionAvail();
        ImGui::Image(
            (ImTextureID)(intptr_t)texture,
            viewportSize,
            ImVec2(0, 1),
            ImVec2(1, 0)
        );


        ImGui::End();
        ImGui::PopStyleVar();


        //
        // Statistics Table
        //

        // calculate statistics every 0.1s
        //  - less flashy, easier to read
        double curr_time = glfwGetTime();

        double delta = curr_time - last_update_time;

        if (delta >= 0.1) {
            display_fps = ImGui::GetIO().Framerate;
            display_ms = ImGui::GetIO().DeltaTime * 1000.0f;
            last_update_time = curr_time;
            int curr_ray_count = get_ray_count();
            mrays = float((curr_ray_count - prev_ray_count)) / delta / 1'000'000.0f;
            prev_ray_count = curr_ray_count;
        }

        ImGui::SetNextWindowSize(ImVec2(MAINWINDOW_WIDTH - VIEWPORT_WIDTH, 200));
        ImGui::SetNextWindowPos(ImVec2(VIEWPORT_WIDTH, 0));

        ImVec4 bg_color = ImVec4(0.1f, 0.1f, 0.15f, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, bg_color);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));


        bool* isopen_stats = NULL;
        ImGui::Begin("Render Statistics", isopen_stats, stationary_window_flags);

        ImGuiTableFlags table_flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;

        if (ImGui::BeginTable("stats_table", 2, table_flags)) {
            // column titles
            ImGui::TableSetupColumn("Metric");
            ImGui::TableSetupColumn("Value");
            ImGui::TableHeadersRow();

            // statistics
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Frames Per Second");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.1f FPS", display_fps);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Frame Time");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.2f ms", display_ms);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Accumulated Frames");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%d", ImGui::GetFrameCount());


            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Rays Per Second");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.2f MRays/s", mrays);

            ImGui::EndTable();
        }

        ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
    }
}


void Window::main_loop(std::function<void(cudaGraphicsResource_t, int, int, int)> kernel) {
    while (!glfwWindowShouldClose(window)) {
        each_frame_pre_kernel();

        kernel(cuda_pbo, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, frame_number);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        each_frame_post_kernel();


        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
        frame_number++;
    }

    // destructor handles clean up
}



void Window::clean_up() {
    if (cuda_pbo) {
        cudaGraphicsUnregisterResource(cuda_pbo);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
    glfwTerminate();
}

Window::~Window() {
    clean_up();
}