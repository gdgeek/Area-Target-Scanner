import Foundation

/// Captured image data associated with a keyframe.
struct CapturedImage {
    /// JPEG image data
    let imageData: Data
    /// Filename for export (e.g. "frame_0000.jpg")
    let filename: String
}

/// The complete result of a scanning session.
/// Contains point cloud data, captured images, camera poses, and intrinsics.
struct ScanResult {
    /// Raw point cloud vertices as (x, y, z, r, g, b, nx, ny, nz) tuples.
    /// Each point has 9 float components: position (3), color (3), normal (3).
    let pointCloudVertices: [[Float]]
    /// List of captured RGB images
    let images: [CapturedImage]
    /// Camera pose for each captured keyframe
    let cameraPoses: [CameraPose]
    /// Camera intrinsic parameters
    let intrinsics: CameraIntrinsics
}
