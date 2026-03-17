// xatlas.h — Placeholder / Minimal API stub
//
// ⚠️  This is a PLACEHOLDER implementation providing the same API surface
//     as the real xatlas library. For production use, replace this file
//     and xatlas.cpp with the actual source from:
//     https://github.com/jpcy/xatlas  (source/xatlas.h, source/xatlas.cpp)
//
// The real xatlas library by Jonathan Young is licensed under the MIT License.

#ifndef XATLAS_H
#define XATLAS_H

#include <stdint.h>

namespace xatlas {

/// Status codes returned by xatlas operations.
enum class AddMeshError {
    Success,
    Error,
    IndexOutOfRange,
    InvalidIndexCount
};

/// Vertex data for input mesh.
struct MeshDecl {
    const void *vertexPositionData = nullptr;
    uint32_t vertexPositionStride = 0;
    const void *vertexNormalData = nullptr;
    uint32_t vertexNormalStride = 0;
    uint32_t vertexCount = 0;

    const void *indexData = nullptr;
    uint32_t indexCount = 0;
    int indexFormat = 1; // 0 = uint16, 1 = uint32
};

/// A single vertex in the output atlas mesh.
struct Vertex {
    int32_t atlasIndex;   // Which atlas chart this vertex belongs to
    int32_t chartIndex;
    float uv[2];          // UV coordinates in atlas space
    uint32_t xref;        // Index of the corresponding input vertex
};

/// Output mesh produced by xatlas.
struct Mesh {
    uint32_t *indexArray = nullptr;
    uint32_t indexCount = 0;
    Vertex *vertexArray = nullptr;
    uint32_t vertexCount = 0;
};

/// Chart options (unused in placeholder, kept for API compat).
struct ChartOptions {
};

/// Pack options for controlling atlas layout.
struct PackOptions {
    uint32_t padding = 1;
    bool bilinear = true;
    bool blockAlign = false;
    bool bruteForce = false;
    uint32_t resolution = 0;  // 0 = auto
};

/// The atlas object — main handle for xatlas operations.
struct Atlas {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t atlasCount = 0;
    uint32_t chartCount = 0;
    uint32_t meshCount = 0;
    Mesh *meshes = nullptr;
};

/// Create a new atlas instance.
Atlas *Create();

/// Destroy an atlas instance and free all memory.
void Destroy(Atlas *atlas);

/// Add a mesh to the atlas for UV unwrapping.
AddMeshError AddMesh(Atlas *atlas, const MeshDecl &meshDecl);

/// Generate UV coordinates for all added meshes.
/// This is the main entry point that performs parameterization and packing.
void Generate(Atlas *atlas,
              ChartOptions chartOptions = ChartOptions(),
              PackOptions packOptions = PackOptions());

} // namespace xatlas

#endif // XATLAS_H
