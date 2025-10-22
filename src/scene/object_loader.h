#pragma once

#include <iostream>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#define JSON_NOEXCEPTION

#include "../tiny_gltf/tiny_gltf.h"
#include "../math/vec3.cuh"

#ifndef white
#define white vec3(.82f)
#endif

#define BUFFER_OFFSET(i) ((char *)NULL + (i))


bool load_model(const char* filename, std::vector<vec3>& all_vertices, std::vector<unsigned int>& all_indices) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool res = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cout << "ERR: " << err << std::endl;
    }

    if (!res) {
        std::cout << "Failed to load glTF: " << filename << std::endl;
        return false;
    }
    else {
        std::cout << "Loaded glTF: " << filename << std::endl;
    }


    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                continue;
            }

            const auto& pos_accessor = model.accessors[primitive.attributes.at("POSITION")];
            const auto& pos_buffer_view = model.bufferViews[pos_accessor.bufferView];
            const auto& pos_buffer = model.buffers[pos_buffer_view.buffer];

            const float* positions = reinterpret_cast<const float*>(
                &pos_buffer.data[pos_buffer_view.byteOffset + pos_accessor.byteOffset]
                );

            size_t vertex_offset = all_vertices.size();

            for (size_t i = 0; i < pos_accessor.count; ++i) {
                all_vertices.push_back(100.0f * vec3(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]));
            }

            if (primitive.indices > -1) {
                const auto& index_accessor = model.accessors[primitive.indices];
                const auto& index_buffer_view = model.bufferViews[index_accessor.bufferView];
                const auto& index_buffer = model.buffers[index_buffer_view.buffer];
                const void* data_ptr = &index_buffer.data[index_buffer_view.byteOffset + index_accessor.byteOffset];

                // The indices can be different types (unsigned short, int, etc.)
                switch (index_accessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                    const uint8_t* indices = static_cast<const uint8_t*>(data_ptr);
                    for (size_t i = 0; i < index_accessor.count; ++i) {
                        all_indices.push_back(indices[i] + vertex_offset);
                    }
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                    const uint16_t* indices = static_cast<const uint16_t*>(data_ptr);
                    for (size_t i = 0; i < index_accessor.count; ++i) {
                        all_indices.push_back(indices[i] + vertex_offset);
                    }
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                    const uint32_t* indices = static_cast<const uint32_t*>(data_ptr);
                    for (size_t i = 0; i < index_accessor.count; ++i) {
                        all_indices.push_back(indices[i] + vertex_offset);
                    }
                    break;
                }
                default:
                    break;
                }
            }
            else {
                size_t num_vertices_in_primitive = pos_accessor.count;
                for (size_t i = 0; i < num_vertices_in_primitive; ++i) {
                    all_indices.push_back(i + vertex_offset);
                }
            }
        }
    }
    return res;
}