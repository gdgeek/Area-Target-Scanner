import SwiftUI

/// Main entry point for the Area Target Scanner iOS app.
///
/// Uses ARKit + LiDAR to capture point clouds, RGB images, and camera poses
/// for downstream 3D reconstruction and AR localization.
@main
struct AreaTargetScannerApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
