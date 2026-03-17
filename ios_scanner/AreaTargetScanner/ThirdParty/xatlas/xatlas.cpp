// xatlas.cpp — Placeholder / Minimal stub implementation
//
// ⚠️  This is a PLACEHOLDER implementation that produces a trivial UV layout.
//     For production use, replace this file and xatlas.h with the actual source from:
//     https://github.com/jpcy/xatlas  (source/xatlas.h, source/xatlas.cpp)
//
// The placeholder generates a simple per-triangle UV layout where each triangle
// is assigned a small rectangular region in the atlas. This is sufficient for
// compilation and basic integration testing, but will NOT produce optimal UV
// packing like the real xatlas library.

#include "xatlas.h"
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace xatlas {

Atlas *Create() {
    Atlas *atlas = new Atlas();
    atlas->width = 0;
    atlas->height = 0;
    atlas->atlasCount = 0;
    atlas->chartCount = 0;
    atlas->meshCount = 0;
    atlas->meshes = nullptr;
    return atlas;
}

void Destroy(Atlas *atlas) {
    if (!atlas) return;
    for (uint32_t i = 0; i < atlas->meshCount; i++) {
        free(atlas->meshes[i].indexArray);
        free(atlas->meshes[i].vertexArray);
    }
    free(atlas->meshes);
    delete atlas;
}

// Internal storage for input mesh data (one mesh supported in placeholder).
struct InternalMeshData {
    float *positions = nullptr;     // 3 floats per vertex
    float *normals = nullptr;       // 3 floats per vertex
    uint32_t *indices = nullptr;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
};

static InternalMeshData g_inputMesh;

AddMeshError AddMesh(Atlas *atlas, const MeshDecl &meshDecl) {
    if (!atlas) return AddMeshError::Error;
    if (meshDecl.vertexCount == 0) return AddMeshError::Error;
    if (meshDecl.indexCount == 0 || meshDecl.indexCount % 3 != 0)
        return AddMeshError::InvalidIndexCount;

    // Store input data
    g_inputMesh.vertexCount = meshDecl.vertexCount;
    g_inputMesh.indexCount = meshDecl.indexCount;

    // Copy positions
    g_inputMesh.positions = (float *)malloc(meshDecl.vertexCount * 3 * sizeof(float));
    for (uint32_t i = 0; i < meshDecl.vertexCount; i++) {
        const uint8_t *src = (const uint8_t *)meshDecl.vertexPositionData
                             + i * meshDecl.vertexPositionStride;
        memcpy(&g_inputMesh.positions[i * 3], src, 3 * sizeof(float));
    }

    // Copy normals if provided
    if (meshDecl.vertexNormalData) {
        g_inputMesh.normals = (float *)malloc(meshDecl.vertexCount * 3 * sizeof(float));
        for (uint32_t i = 0; i < meshDecl.vertexCount; i++) {
            const uint8_t *src = (const uint8_t *)meshDecl.vertexNormalData
                                 + i * meshDecl.vertexNormalStride;
            memcpy(&g_inputMesh.normals[i * 3], src, 3 * sizeof(float));
        }
    }

    // Copy indices
    g_inputMesh.indices = (uint32_t *)malloc(meshDecl.indexCount * sizeof(uint32_t));
    if (meshDecl.indexFormat == 0) {
        // uint16 indices
        const uint16_t *src = (const uint16_t *)meshDecl.indexData;
        for (uint32_t i = 0; i < meshDecl.indexCount; i++) {
            g_inputMesh.indices[i] = (uint32_t)src[i];
        }
    } else {
        memcpy(g_inputMesh.indices, meshDecl.indexData,
               meshDecl.indexCount * sizeof(uint32_t));
    }

    // Validate indices
    for (uint32_t i = 0; i < meshDecl.indexCount; i++) {
        if (g_inputMesh.indices[i] >= meshDecl.vertexCount) {
            free(g_inputMesh.positions);
            free(g_inputMesh.normals);
            free(g_inputMesh.indices);
            g_inputMesh = InternalMeshData();
            return AddMeshError::IndexOutOfRange;
        }
    }

    atlas->meshCount = 1;
    return AddMeshError::Success;
}

void Generate(Atlas *atlas, ChartOptions /*chartOptions*/, PackOptions /*packOptions*/) {
    if (!atlas || atlas->meshCount == 0) return;

    uint32_t triCount = g_inputMesh.indexCount / 3;

    // Placeholder UV strategy: each triangle gets its own 3 output vertices.
    // UVs are laid out in a grid of small triangles across the atlas.
    uint32_t outVertexCount = triCount * 3;
    uint32_t outIndexCount = triCount * 3;

    Vertex *outVertices = (Vertex *)calloc(outVertexCount, sizeof(Vertex));
    uint32_t *outIndices = (uint32_t *)calloc(outIndexCount, sizeof(uint32_t));

    // Determine grid layout: ceil(sqrt(triCount)) columns
    uint32_t cols = (uint32_t)ceil(sqrt((double)triCount));
    if (cols == 0) cols = 1;
    uint32_t rows = (triCount + cols - 1) / cols;

    // Atlas dimensions in pixels — UVs are output in pixel coordinates
    // (the bridge normalizes by dividing by width/height)
    uint32_t atlasW = cols * 32;  // 32 pixels per cell
    uint32_t atlasH = rows * 32;
    if (atlasW < 64) atlasW = 64;
    if (atlasH < 64) atlasH = 64;

    float cellW = (float)atlasW / (float)cols;
    float cellH = (float)atlasH / (float)rows;

    // Add 1-pixel padding inside each cell to avoid seam bleeding
    float pad = 1.0f;

    for (uint32_t t = 0; t < triCount; t++) {
        uint32_t row = t / cols;
        uint32_t col = t % cols;

        float u0 = (float)col * cellW + pad;
        float v0 = (float)row * cellH + pad;
        float cw = cellW - 2.0f * pad;
        float ch = cellH - 2.0f * pad;
        if (cw < 1.0f) cw = 1.0f;
        if (ch < 1.0f) ch = 1.0f;

        // Three vertices of the triangle in UV pixel space (small right triangle)
        uint32_t vi = t * 3;

        outVertices[vi + 0].atlasIndex = 0;
        outVertices[vi + 0].chartIndex = (int32_t)t;
        outVertices[vi + 0].uv[0] = u0;
        outVertices[vi + 0].uv[1] = v0;
        outVertices[vi + 0].xref = g_inputMesh.indices[t * 3 + 0];

        outVertices[vi + 1].atlasIndex = 0;
        outVertices[vi + 1].chartIndex = (int32_t)t;
        outVertices[vi + 1].uv[0] = u0 + cw;
        outVertices[vi + 1].uv[1] = v0;
        outVertices[vi + 1].xref = g_inputMesh.indices[t * 3 + 1];

        outVertices[vi + 2].atlasIndex = 0;
        outVertices[vi + 2].chartIndex = (int32_t)t;
        outVertices[vi + 2].uv[0] = u0;
        outVertices[vi + 2].uv[1] = v0 + ch;
        outVertices[vi + 2].xref = g_inputMesh.indices[t * 3 + 2];

        outIndices[vi + 0] = vi + 0;
        outIndices[vi + 1] = vi + 1;
        outIndices[vi + 2] = vi + 2;
    }

    // Clamp UVs to atlas bounds
    for (uint32_t i = 0; i < outVertexCount; i++) {
        if (outVertices[i].uv[0] > (float)atlasW) outVertices[i].uv[0] = (float)atlasW;
        if (outVertices[i].uv[1] > (float)atlasH) outVertices[i].uv[1] = (float)atlasH;
        if (outVertices[i].uv[0] < 0.0f) outVertices[i].uv[0] = 0.0f;
        if (outVertices[i].uv[1] < 0.0f) outVertices[i].uv[1] = 0.0f;
    }

    // Build output
    atlas->meshes = (Mesh *)calloc(1, sizeof(Mesh));
    atlas->meshes[0].vertexArray = outVertices;
    atlas->meshes[0].vertexCount = outVertexCount;
    atlas->meshes[0].indexArray = outIndices;
    atlas->meshes[0].indexCount = outIndexCount;
    atlas->atlasCount = 1;
    atlas->chartCount = triCount;
    atlas->width = atlasW;
    atlas->height = atlasH;

    // Clean up input data
    free(g_inputMesh.positions);
    free(g_inputMesh.normals);
    free(g_inputMesh.indices);
    g_inputMesh = InternalMeshData();
}

} // namespace xatlas
