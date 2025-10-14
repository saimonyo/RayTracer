#pragma once

#include "vec3.cuh"

class ray {
    vec3 o;
    vec3 d;

    ray(vec3 origin, vec3 direction) {
        o = origin;
        d = direction;
    }

    vec3 at_time(float t) {
        return o + d * t;
    }
};