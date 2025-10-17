#pragma once

#include "vec3.cuh"

struct Quaternion: public float4 {
	inline Quaternion() { x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f; }
	inline Quaternion(float _x, float _y, float _z, float _w) { x = _x, y = _y, z = _z, w = _w; }
};

inline vec3 operator*(const Quaternion& quaternion, const vec3& vector) {
	vec3 q = vec3(quaternion.x, quaternion.y, quaternion.z);

	return 2.0f * dot(q, vector) * q +
		(quaternion.w * quaternion.w - dot(q, q)) * vector +
		2.0f * quaternion.w * cross(q, vector);
}

// Based on: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
inline static Quaternion quaternion_from_euler(float yaw, float pitch, float roll) {
	float cos_yaw = cosf(yaw * 0.5f);
	float sin_yaw = sinf(yaw * 0.5f);
	float cos_pitch = cosf(pitch * 0.5f);
	float sin_pitch = sinf(pitch * 0.5f);
	float cos_roll = cosf(roll * 0.5f);
	float sin_roll = sinf(roll * 0.5f);

	return {
		sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
		cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
		cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw,
		cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
	};
}