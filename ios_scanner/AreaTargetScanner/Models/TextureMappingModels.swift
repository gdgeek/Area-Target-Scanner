import Foundation
import simd
import CoreGraphics

// MARK: - Simplified Mesh Data (for testing without ARMeshAnchor)

/// Simplified mesh data for testing (avoids ARMeshAnchor dependency).
/// Each instance represents one mesh "anchor" with its own local vertices,
/// faces (indices into local vertices), and a transform to world space.
struct SimpleMeshData {
    let vertices: [SIMD3<Float>]
    let faces: [SIMD3<UInt32>]
    let transform: simd_float4x4
}

// MARK: - Mesh Data Models

/// A unified mesh merged from multiple ARMeshAnchors, with all vertices
/// transformed into world coordinates.
struct MergedMesh {
    /// World-space vertex positions
    var vertices: [SIMD3<Float>]
    /// Per-vertex normals (count must equal vertices.count)
    var normals: [SIMD3<Float>]
    /// Triangle face indices into the vertices array
    var faces: [SIMD3<UInt32>]
}

/// Mesh data after UV unwrapping via xatlas.
/// Vertex count may exceed the original mesh due to UV seam splits.
struct UVMesh {
    /// World-space vertex positions (may be duplicated at seams)
    let vertices: [SIMD3<Float>]
    /// Per-vertex normals
    let normals: [SIMD3<Float>]
    /// Per-vertex UV coordinates, normalized to [0, 1]
    let uvCoordinates: [SIMD2<Float>]
    /// Triangle face indices
    let faces: [SIMD3<UInt32>]
    /// Maps each vertex back to its index in the original MergedMesh
    let originalVertexIndices: [UInt32]
}

/// The final output of the texture mapping pipeline.
struct TexturedMeshResult {
    /// World-space vertex positions
    let vertices: [SIMD3<Float>]
    /// Per-vertex normals
    let normals: [SIMD3<Float>]
    /// Per-vertex texture UV coordinates in [0, 1]
    let uvCoordinates: [SIMD2<Float>]
    /// Triangle face indices
    let faces: [SIMD3<UInt32>]
    /// The rendered texture atlas image
    let textureImage: CGImage
}

// MARK: - Frame Selection

/// Maps a triangle face to its best-matching camera keyframe.
struct FaceFrameAssignment {
    /// Index of the face in the mesh's face array
    let faceIndex: Int
    /// Index of the selected keyframe
    let frameIndex: Int
    /// Quality score (higher is better; 0 when using fallback)
    let score: Float
}

// MARK: - Errors

/// Errors that can occur during the texture mapping pipeline.
enum TextureMappingError: Error, LocalizedError {
    /// No mesh anchors were provided.
    case noMeshData
    /// No keyframe images or camera poses were provided.
    case noKeyframeData
    /// xatlas UV unwrap failed.
    case uvUnwrapFailed(reason: String)
    /// Texture atlas rendering failed.
    case atlasRenderFailed(reason: String)

    var errorDescription: String? {
        switch self {
        case .noMeshData:
            return "No mesh data available for texture mapping."
        case .noKeyframeData:
            return "No keyframe images or camera poses available."
        case .uvUnwrapFailed(let reason):
            return "UV unwrap failed: \(reason)"
        case .atlasRenderFailed(let reason):
            return "Atlas render failed: \(reason)"
        }
    }
}
