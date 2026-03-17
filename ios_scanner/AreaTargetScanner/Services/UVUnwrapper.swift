import Foundation
import simd

/// Wraps the xatlas C++ library to perform automatic UV unwrapping
/// on a merged mesh, producing non-overlapping UV coordinates.
final class UVUnwrapper {

    /// Perform automatic UV unwrapping on the given mesh.
    /// - Parameters:
    ///   - vertices: World-space vertex positions
    ///   - normals: Per-vertex normals
    ///   - faces: Triangle face indices
    /// - Returns: A UVMesh with normalized UV coordinates in [0, 1]
    func unwrap(
        vertices: [SIMD3<Float>],
        normals: [SIMD3<Float>],
        faces: [SIMD3<UInt32>]
    ) throws -> UVMesh {
        // 1. Flatten vertices to interleaved [Float] (x, y, z, x, y, z, ...)
        var flatVertices = [Float]()
        flatVertices.reserveCapacity(vertices.count * 3)
        for v in vertices {
            flatVertices.append(v.x)
            flatVertices.append(v.y)
            flatVertices.append(v.z)
        }

        // 2. Flatten normals to interleaved [Float]
        var flatNormals = [Float]()
        flatNormals.reserveCapacity(normals.count * 3)
        for n in normals {
            flatNormals.append(n.x)
            flatNormals.append(n.y)
            flatNormals.append(n.z)
        }

        // 3. Flatten faces to interleaved [UInt32] (i0, i1, i2, ...)
        var flatFaces = [UInt32]()
        flatFaces.reserveCapacity(faces.count * 3)
        for f in faces {
            flatFaces.append(f.x)
            flatFaces.append(f.y)
            flatFaces.append(f.z)
        }

        // 4. Call XAtlasBridge
        let result: XAtlasResult
        do {
            result = try XAtlasBridge.unwrap(
                withVertices: flatVertices,
                vertexCount: Int32(vertices.count),
                normals: flatNormals,
                faces: flatFaces,
                faceCount: Int32(faces.count)
            )
        } catch {
            let reason = error.localizedDescription
            throw TextureMappingError.uvUnwrapFailed(reason: reason)
        }

        let outVertexCount = Int(result.vertexCount)
        let outFaceCount = Int(result.faceCount)

        // 5. Convert result vertices (float3 interleaved) → [SIMD3<Float>]
        var outVertices = [SIMD3<Float>]()
        outVertices.reserveCapacity(outVertexCount)
        for i in 0..<outVertexCount {
            let x = result.vertices[i * 3].floatValue
            let y = result.vertices[i * 3 + 1].floatValue
            let z = result.vertices[i * 3 + 2].floatValue
            outVertices.append(SIMD3<Float>(x, y, z))
        }

        // 6. Convert result normals (float3 interleaved) → [SIMD3<Float>]
        var outNormals = [SIMD3<Float>]()
        outNormals.reserveCapacity(outVertexCount)
        for i in 0..<outVertexCount {
            let nx = result.normals[i * 3].floatValue
            let ny = result.normals[i * 3 + 1].floatValue
            let nz = result.normals[i * 3 + 2].floatValue
            outNormals.append(SIMD3<Float>(nx, ny, nz))
        }

        // 7. Convert result UV coordinates (float2 interleaved) → [SIMD2<Float>]
        //    Already normalized to [0, 1] by the bridge
        var outUVs = [SIMD2<Float>]()
        outUVs.reserveCapacity(outVertexCount)
        for i in 0..<outVertexCount {
            let u = result.uvCoordinates[i * 2].floatValue
            let v = result.uvCoordinates[i * 2 + 1].floatValue
            outUVs.append(SIMD2<Float>(u, v))
        }

        // 8. Convert result faces (uint32 triplets) → [SIMD3<UInt32>]
        var outFaces = [SIMD3<UInt32>]()
        outFaces.reserveCapacity(outFaceCount)
        for i in 0..<outFaceCount {
            let i0 = result.faces[i * 3].uint32Value
            let i1 = result.faces[i * 3 + 1].uint32Value
            let i2 = result.faces[i * 3 + 2].uint32Value
            outFaces.append(SIMD3<UInt32>(i0, i1, i2))
        }

        // 9. Convert original vertex indices → [UInt32]
        var outOriginalIndices = [UInt32]()
        outOriginalIndices.reserveCapacity(outVertexCount)
        for i in 0..<outVertexCount {
            outOriginalIndices.append(result.originalVertexIndices[i].uint32Value)
        }

        return UVMesh(
            vertices: outVertices,
            normals: outNormals,
            uvCoordinates: outUVs,
            faces: outFaces,
            originalVertexIndices: outOriginalIndices
        )
    }
}
