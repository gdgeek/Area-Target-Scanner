#import "XAtlasBridge.h"
#include "xatlas.h"

static NSString *const XAtlasBridgeErrorDomain = @"com.areatarget.xatlas";

@implementation XAtlasResult
@end

@implementation XAtlasBridge

+ (nullable XAtlasResult *)unwrapWithVertices:(const float *)vertices
                                  vertexCount:(int)vertexCount
                                      normals:(const float *)normals
                                        faces:(const uint32_t *)faces
                                    faceCount:(int)faceCount
                                        error:(NSError *_Nullable *_Nullable)error {
    // Validate inputs
    if (vertexCount <= 0 || faceCount <= 0 || !vertices || !faces) {
        if (error) {
            *error = [NSError errorWithDomain:XAtlasBridgeErrorDomain
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid input: vertex or face data is empty"}];
        }
        return nil;
    }

    // 1. Create xatlas atlas
    xatlas::Atlas *atlas = xatlas::Create();
    if (!atlas) {
        if (error) {
            *error = [NSError errorWithDomain:XAtlasBridgeErrorDomain
                                         code:2
                                     userInfo:@{NSLocalizedDescriptionKey: @"Failed to create xatlas atlas"}];
        }
        return nil;
    }

    // 2. Set up mesh declaration
    xatlas::MeshDecl meshDecl;
    meshDecl.vertexPositionData = vertices;
    meshDecl.vertexPositionStride = sizeof(float) * 3;
    meshDecl.vertexCount = (uint32_t)vertexCount;

    if (normals) {
        meshDecl.vertexNormalData = normals;
        meshDecl.vertexNormalStride = sizeof(float) * 3;
    }

    meshDecl.indexData = faces;
    meshDecl.indexCount = (uint32_t)faceCount * 3;
    meshDecl.indexFormat = 1; // uint32

    // 3. Add mesh
    xatlas::AddMeshError addResult = xatlas::AddMesh(atlas, meshDecl);
    if (addResult != xatlas::AddMeshError::Success) {
        xatlas::Destroy(atlas);
        if (error) {
            NSString *reason;
            switch (addResult) {
                case xatlas::AddMeshError::IndexOutOfRange:
                    reason = @"Index out of range";
                    break;
                case xatlas::AddMeshError::InvalidIndexCount:
                    reason = @"Invalid index count (must be multiple of 3)";
                    break;
                default:
                    reason = @"Unknown error adding mesh";
                    break;
            }
            *error = [NSError errorWithDomain:XAtlasBridgeErrorDomain
                                         code:3
                                     userInfo:@{NSLocalizedDescriptionKey: reason}];
        }
        return nil;
    }

    // 4. Generate UV layout
    xatlas::Generate(atlas);

    // 5. Validate output
    if (atlas->meshCount == 0 || atlas->meshes[0].vertexCount == 0) {
        xatlas::Destroy(atlas);
        if (error) {
            *error = [NSError errorWithDomain:XAtlasBridgeErrorDomain
                                         code:4
                                     userInfo:@{NSLocalizedDescriptionKey: @"xatlas generated empty output"}];
        }
        return nil;
    }

    // 6. Extract output mesh
    const xatlas::Mesh &outMesh = atlas->meshes[0];
    uint32_t outVertexCount = outMesh.vertexCount;
    uint32_t outIndexCount = outMesh.indexCount;
    uint32_t outFaceCount = outIndexCount / 3;

    // Build result arrays
    NSMutableArray<NSNumber *> *outVertices = [NSMutableArray arrayWithCapacity:outVertexCount * 3];
    NSMutableArray<NSNumber *> *outNormals = [NSMutableArray arrayWithCapacity:outVertexCount * 3];
    NSMutableArray<NSNumber *> *outUVs = [NSMutableArray arrayWithCapacity:outVertexCount * 2];
    NSMutableArray<NSNumber *> *outFaces = [NSMutableArray arrayWithCapacity:outFaceCount * 3];
    NSMutableArray<NSNumber *> *outOriginalIndices = [NSMutableArray arrayWithCapacity:outVertexCount];

    // Determine atlas dimensions for UV normalization
    float atlasWidth = (float)atlas->width;
    float atlasHeight = (float)atlas->height;
    if (atlasWidth <= 0) atlasWidth = 1.0f;
    if (atlasHeight <= 0) atlasHeight = 1.0f;

    for (uint32_t i = 0; i < outVertexCount; i++) {
        const xatlas::Vertex &v = outMesh.vertexArray[i];
        uint32_t origIdx = v.xref;

        // Copy position from original vertex
        [outVertices addObject:@(vertices[origIdx * 3 + 0])];
        [outVertices addObject:@(vertices[origIdx * 3 + 1])];
        [outVertices addObject:@(vertices[origIdx * 3 + 2])];

        // Copy normal from original vertex
        if (normals) {
            [outNormals addObject:@(normals[origIdx * 3 + 0])];
            [outNormals addObject:@(normals[origIdx * 3 + 1])];
            [outNormals addObject:@(normals[origIdx * 3 + 2])];
        } else {
            [outNormals addObject:@(0.0f)];
            [outNormals addObject:@(1.0f)];
            [outNormals addObject:@(0.0f)];
        }

        // Normalize UV coordinates to [0, 1]
        float u = v.uv[0] / atlasWidth;
        float vCoord = v.uv[1] / atlasHeight;
        // Clamp to [0, 1]
        if (u < 0.0f) u = 0.0f;
        if (u > 1.0f) u = 1.0f;
        if (vCoord < 0.0f) vCoord = 0.0f;
        if (vCoord > 1.0f) vCoord = 1.0f;
        [outUVs addObject:@(u)];
        [outUVs addObject:@(vCoord)];

        // Original vertex index
        [outOriginalIndices addObject:@(origIdx)];
    }

    // Copy face indices
    for (uint32_t i = 0; i < outIndexCount; i++) {
        [outFaces addObject:@(outMesh.indexArray[i])];
    }

    // 7. Build result
    XAtlasResult *result = [[XAtlasResult alloc] init];
    result.vertices = [outVertices copy];
    result.normals = [outNormals copy];
    result.uvCoordinates = [outUVs copy];
    result.faces = [outFaces copy];
    result.originalVertexIndices = [outOriginalIndices copy];
    result.vertexCount = (int)outVertexCount;
    result.faceCount = (int)outFaceCount;

    // 8. Cleanup
    xatlas::Destroy(atlas);

    return result;
}

@end
