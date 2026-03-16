import Foundation

/// Real-time scan progress information.
/// Tracks the number of captured points and estimated coverage area.
struct ScanProgress: Equatable {
    /// Total number of captured point cloud points
    let pointCount: Int
    /// Estimated coverage area in square meters
    let coverageArea: Float
    /// Number of captured keyframes (image + pose pairs)
    let keyframeCount: Int
    /// Whether the scan is currently active
    let isScanning: Bool
}
