import SwiftUI
import ARKit
import RealityKit

/// Live AR camera preview shown during scanning.
/// Shares the ARSession from ARKitScannerService so there's no conflict.
struct ARScanningView: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        arView.session = session

        // Show mesh wireframe overlay so user sees scanned geometry
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            arView.debugOptions.insert(.showSceneUnderstanding)
        }

        return arView
    }

    func updateUIView(_ arView: ARView, context: Context) {}
}
