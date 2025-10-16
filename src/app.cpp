#include "Window.h"

// screen paramaters
const int vp_width = 800;
const int vp_height = 600;
const int main_width = 1200;
const int main_height = 600;

int main(void) {
    Window window = Window(vp_width, vp_height, main_width, main_height);

    window.main_loop(render_frame);
}