#include "Window.h"
#include "scene/object_loader.cuh"

// screen paramaters
const int vp_width = 800;
const int vp_height = 600;
const int main_width = 1200;
const int main_height = 600;

int main(void) {
    tinygltf::Model model;
    load_model(model, "data/Objects/bunny/scene.gltf");
    Window window = Window(vp_width, vp_height, main_width, main_height);

    window.main_loop(render_frame);
}