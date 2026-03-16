import Foundation

/// Camera intrinsic parameters (focal length and principal point).
/// Corresponds to the CameraIntrinsics structure in the design doc.
struct CameraIntrinsics: Codable, Equatable {
    /// Focal length in x (pixels)
    let fx: Float
    /// Focal length in y (pixels)
    let fy: Float
    /// Principal point x (pixels)
    let cx: Float
    /// Principal point y (pixels)
    let cy: Float
    /// Image width in pixels
    let width: Int
    /// Image height in pixels
    let height: Int
}
