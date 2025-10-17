#pragma once

#include "assert.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>

#ifndef F_PI
#define F_PI 3.14159265358979323846f
#endif

constexpr float ONE_OVER_PI = 1.0f / F_PI;


struct vec3 : public float3 {
    //
    // initialisation
    //

    __host__ __device__ vec3() {
        x = 0.0f; y = 0.0f; z = 0.0f;
    }

    __host__ __device__ vec3(float _x, float _y, float _z) {
        x = _x; y = _y; z = _z;
    }

    __host__ __device__ inline vec3(float v) {
        x = v; y = v; z = v;
    }
    __host__ __device__ inline vec3(const float arr[3]) {
        x = arr[0]; y = arr[1]; z = arr[2];
    }

    //
    // inplace operations
    //

    __host__ __device__ inline vec3 operator+=(const vec3& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    __host__ __device__ inline vec3 operator-=(const vec3& v) {
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }
    __host__ __device__ inline vec3 operator*=(const vec3& v) {
        x *= v.x; y *= v.y; z *= v.z;
        return *this;
    }
    __host__ __device__ inline vec3 operator/=(const vec3& v) {
        x /= v.x; y /= v.y; z /= v.z;
        return *this;
    }

    __host__ __device__ inline vec3 operator*=(float scalar) {
        x *= scalar; y *= scalar; z *= scalar;
        return *this;
    }
    __host__ __device__ inline vec3 operator/=(float scalar) {
        float inv_scalar = 1.0f / scalar;
        x *= inv_scalar;
        y *= inv_scalar;
        z *= inv_scalar;
        return *this;
    }

    //
    // getters based on index
    //

    __host__ __device__ inline float& operator[](int index) {
        assert(index >= 0 && index < 3);
        if (index == 0) {
            return x;
        }
        if (index == 1) {
            return y;
        }
        return z;
    }

    __host__ __device__ inline const float& operator[](int index) const {
        assert(index >= 0 && index < 3);
        if (index == 0) {
            return x;
        }
        if (index == 1) {
            return y;
        }
        return z;
    }

};

//
// vector vector basic operations
//

__host__ __device__ inline vec3 operator-(const vec3& v) { 
    return vec3(-v.x, -v.y, -v.z); 
}
__host__ __device__ inline vec3 operator+(const vec3& left, const vec3& right) {
    return vec3(left.x + right.x, left.y + right.y, left.z + right.z);
}
__host__ __device__ inline vec3 operator-(const vec3& left, const vec3& right) { 
    return vec3(left.x - right.x, left.y - right.y, left.z - right.z);
}
__host__ __device__ inline vec3 operator*(const vec3& left, const vec3& right) {
    return vec3(left.x * right.x, left.y * right.y, left.z * right.z);
}
__host__ __device__ inline vec3 operator/(const vec3& left, const vec3& right) {
    return vec3(left.x / right.x, left.y / right.y, left.z / right.z);
}

//
// vector scalar basic operations
//

__host__ __device__ inline vec3 operator+(const vec3& v, float scalar) { 
    return vec3(v.x + scalar, v.y + scalar, v.z + scalar);
}
__host__ __device__ inline vec3 operator-(const vec3& v, float scalar) { 
    return vec3(v.x - scalar, v.y - scalar, v.z - scalar);
}
__host__ __device__ inline vec3 operator*(const vec3& v, float scalar) {
    return vec3(v.x * scalar, v.y * scalar, v.z * scalar);
}
__host__ __device__ inline vec3 operator/(const vec3& v, float scalar) {
    float inv_scalar = 1.0f / scalar; 
    return vec3(v.x * inv_scalar, v.y * inv_scalar, v.z * inv_scalar);
}

//
// scalar vector basic operations
//

__host__ __device__ inline vec3 operator+(float scalar, const vec3& v) {
    return vec3(scalar + v.x, scalar + v.y, scalar + v.z);
}
__host__ __device__ inline vec3 operator-(float scalar, const vec3& v) {
    return vec3(scalar - v.x, scalar - v.y, scalar - v.z);
}
__host__ __device__ inline vec3 operator*(float scalar, const vec3& v) {
    return vec3(scalar * v.x, scalar * v.y, scalar * v.z);
}
__host__ __device__ inline vec3 operator/(float scalar, const vec3& v) {
    return vec3(scalar / v.x, scalar / v.y, scalar / v.z);
}



//
// vector vector operations
//

__host__ __device__ inline float dot(const vec3& left, const vec3& right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

__host__ __device__ inline vec3 cross(const vec3& left, const vec3& right) {
    return vec3(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x
    );
}

__host__ __device__ inline vec3 min(const vec3& left, const vec3& right) {
    return vec3(
        left.x < right.x ? left.x : right.x,
        left.y < right.y ? left.y : right.y,
        left.z < right.z ? left.z : right.z
    );
}

__host__ __device__ inline vec3 max(const vec3& left, const vec3& right) {
    return vec3(
        left.x > right.x ? left.x : right.x,
        left.y > right.y ? left.y : right.y,
        left.z > right.z ? left.z : right.z
    );
}


//
// single vector operations
//

__host__ __device__ inline float length_squared(const vec3& v) {
    return dot(v, v);
}

__host__ __device__ inline float length(const vec3& v) {
    return sqrtf(length_squared(v));
}

__host__ __device__ inline vec3 normalise(const vec3& v) {
    return v / length(v);
}

__host__ __device__ inline vec3 pow(const vec3& v, float e) {
    return vec3(powf(v.x, e), powf(v.y, e), powf(v.z, e));
}

__host__ __device__ inline vec3 sqrt(const vec3& v) {
    return vec3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

__host__ __device__ inline vec3 clamp(const vec3& v, float low, float high) {
    return max(vec3(low), min(vec3(high), v));
}

__host__ __device__ inline float min_component(const vec3& v) {
    return fminf(v.x, fminf(v.y, v.z));
}

__host__ __device__ inline float max_component(const vec3& v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}