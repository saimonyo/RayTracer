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

bool load_model(const char* filename, std::vector<vec3>& all_vertices, std::vector<unsigned int>& all_indices, std::vector<HostMaterial>& all_materials) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool res = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    if (!warn.empty()) { std::cout << "WARN: " << warn << std::endl; }
    if (!err.empty()) { std::cout << "ERR: " << err << std::endl; }
    if (!res) {
        std::cout << "Failed to load glTF: " << filename << std::endl;
        return false;
    }

    
    HostMaterial default_material = {white, vec3(0.0f), 0.0f};


    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                continue;
            }

            HostMaterial current_material = default_material;
            // get current material, default to the default matieral
            if (primitive.material >= 0) {
                const tinygltf::Material& mat = model.materials[primitive.material];
                if (mat.emissiveFactor.size() == 3) {
                    vec3 emission = vec3(
                        static_cast<float>(mat.emissiveFactor[0]),
                        static_cast<float>(mat.emissiveFactor[1]),
                        static_cast<float>(mat.emissiveFactor[2])
                    );

                    float strength = max_component(emission);
                    // prevent div by 0 error
                    if (strength > 0.0f) {
                        current_material.emission_colour = emission / strength;
                        current_material.emission_strength = strength;
                    }
                    else {
                        current_material.emission_colour = vec3(0.0f, 0.0f, 0.0f);
                        current_material.emission_strength = 0.0f;
                    }
                }

                // PBR metallic-roughness properties
                const auto& pbr = mat.pbrMetallicRoughness;
                if (pbr.baseColorFactor.size() == 4) {
                    current_material.albedo = vec3(
                        static_cast<float>(pbr.baseColorFactor[0]),
                        static_cast<float>(pbr.baseColorFactor[1]),
                        static_cast<float>(pbr.baseColorFactor[2])
                    );
                }
            }

            const auto& pos_accessor = model.accessors[primitive.attributes.at("POSITION")];
            const auto& pos_buffer_view = model.bufferViews[pos_accessor.bufferView];
            const auto& pos_buffer = model.buffers[pos_buffer_view.buffer];

            const float* positions = reinterpret_cast<const float*>(
                &pos_buffer.data[pos_buffer_view.byteOffset + pos_accessor.byteOffset]
                );

            size_t vertex_offset = all_vertices.size();
            for (size_t i = 0; i < pos_accessor.count; ++i) {
                all_vertices.push_back(vec3(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]));
            }

            if (primitive.indices > -1) {
                const auto& index_accessor = model.accessors[primitive.indices];
                const auto& index_buffer_view = model.bufferViews[index_accessor.bufferView];
                const auto& index_buffer = model.buffers[index_accessor.bufferView];
                const void* data_ptr = &index_buffer.data[index_buffer_view.byteOffset + index_accessor.byteOffset];


                // add material for each triangle
                switch (index_accessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                        const auto* indices = static_cast<const uint8_t*>(data_ptr);
                        for (size_t i = 0; i < index_accessor.count; i += 3) {
                            all_indices.push_back(indices[i + 0] + vertex_offset);
                            all_indices.push_back(indices[i + 1] + vertex_offset);
                            all_indices.push_back(indices[i + 2] + vertex_offset);
                            all_materials.push_back(current_material);
                        }
                        break;
                    }
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                        const auto* indices = static_cast<const uint16_t*>(data_ptr);
                        for (size_t i = 0; i < index_accessor.count; i += 3) {
                            all_indices.push_back(indices[i + 0] + vertex_offset);
                            all_indices.push_back(indices[i + 1] + vertex_offset);
                            all_indices.push_back(indices[i + 2] + vertex_offset);
                            all_materials.push_back(current_material);
                        }
                        break;
                    }
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                        const auto* indices = static_cast<const uint32_t*>(data_ptr);
                        for (size_t i = 0; i < index_accessor.count; i += 3) {
                            all_indices.push_back(indices[i + 0] + vertex_offset);
                            all_indices.push_back(indices[i + 1] + vertex_offset);
                            all_indices.push_back(indices[i + 2] + vertex_offset);
                            all_materials.push_back(current_material);
                        }
                        break;
                    }
                    default: 
                        break;
                }
            }
            else {
                size_t num_vertices = pos_accessor.count;
                for (size_t i = 0; i < num_vertices; i += 3) {
                    all_indices.push_back(i + 0 + vertex_offset);
                    all_indices.push_back(i + 1 + vertex_offset);
                    all_indices.push_back(i + 2 + vertex_offset);
                    all_materials.push_back(current_material);
                }
            }
        }
    }
    return res;
}