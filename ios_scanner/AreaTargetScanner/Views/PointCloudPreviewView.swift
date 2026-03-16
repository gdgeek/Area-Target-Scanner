import SwiftUI
import SceneKit

/// A SceneKit-based 3D preview that visualizes the spatial distribution of scanned point cloud data.
///
/// Renders point cloud vertices as colored spheres in a 3D scene with orbit camera controls.
///
/// - Requirements: 2.2
struct PointCloudPreviewView: UIViewRepresentable {
    /// Point cloud vertices, each as [x, y, z, r, g, b, nx, ny, nz].
    let vertices: [[Float]]

    /// Maximum number of points to render for performance.
    private static let maxDisplayPoints = 10_000

    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.scene = SCNScene()
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = true
        scnView.backgroundColor = .black
        scnView.antialiasingMode = .multisampling4X
        return scnView
    }

    func updateUIView(_ scnView: SCNView, context: Context) {
        guard let scene = scnView.scene else { return }

        // Remove previous point cloud node
        scene.rootNode.childNodes
            .filter { $0.name == "pointCloud" }
            .forEach { $0.removeFromParentNode() }

        guard !vertices.isEmpty else { return }

        let pointCloudNode = buildPointCloudNode(from: vertices)
        pointCloudNode.name = "pointCloud"
        scene.rootNode.addChildNode(pointCloudNode)

        // Fit camera to content on first load
        if context.coordinator.isFirstUpdate {
            scnView.pointOfView = makeCamera(fitting: pointCloudNode)
            context.coordinator.isFirstUpdate = false
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var isFirstUpdate = true
    }

    // MARK: - Point Cloud Geometry

    /// Builds a SceneKit node containing the point cloud as custom geometry.
    private func buildPointCloudNode(from vertices: [[Float]]) -> SCNNode {
        let sampled = downsampleForDisplay(vertices)

        var positions: [SCNVector3] = []
        var colors: [SCNVector3] = []
        positions.reserveCapacity(sampled.count)
        colors.reserveCapacity(sampled.count)

        for vertex in sampled {
            guard vertex.count >= 6 else { continue }
            positions.append(SCNVector3(vertex[0], vertex[1], vertex[2]))
            // Use vertex color if available, otherwise default to white
            let r = vertex[3], g = vertex[4], b = vertex[5]
            let hasColor = r > 0 || g > 0 || b > 0
            colors.append(hasColor ? SCNVector3(r, g, b) : SCNVector3(1, 1, 1))
        }

        let positionSource = SCNGeometrySource(vertices: positions)
        let colorData = Data(bytes: colors, count: colors.count * MemoryLayout<SCNVector3>.stride)
        let colorSource = SCNGeometrySource(
            data: colorData,
            semantic: .color,
            vectorCount: colors.count,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: MemoryLayout<SCNVector3>.stride
        )

        let indices = (0..<Int32(positions.count)).map { $0 }
        let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .point,
            primitiveCount: positions.count,
            bytesPerIndex: MemoryLayout<Int32>.size
        )
        element.pointSize = 3
        element.minimumPointScreenSpaceRadius = 1
        element.maximumPointScreenSpaceRadius = 5

        let geometry = SCNGeometry(sources: [positionSource, colorSource], elements: [element])
        return SCNNode(geometry: geometry)
    }

    /// Downsamples vertices to at most `maxDisplayPoints` using uniform stride.
    private func downsampleForDisplay(_ vertices: [[Float]]) -> [[Float]] {
        guard vertices.count > Self.maxDisplayPoints else { return vertices }
        let stride = max(1, vertices.count / Self.maxDisplayPoints)
        return Swift.stride(from: 0, to: vertices.count, by: stride).map { vertices[$0] }
    }

    /// Creates a camera node positioned to frame the given content node.
    private func makeCamera(fitting node: SCNNode) -> SCNNode {
        let (minBound, maxBound) = node.boundingBox
        let center = SCNVector3(
            (minBound.x + maxBound.x) / 2,
            (minBound.y + maxBound.y) / 2,
            (minBound.z + maxBound.z) / 2
        )
        let size = SCNVector3(
            maxBound.x - minBound.x,
            maxBound.y - minBound.y,
            maxBound.z - minBound.z
        )
        let maxDimension = max(size.x, max(size.y, size.z))
        let distance = max(Float(maxDimension) * 1.5, 1.0)

        let camera = SCNCamera()
        camera.automaticallyAdjustsZRange = true

        let cameraNode = SCNNode()
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(
            center.x,
            center.y + distance * 0.3,
            center.z + distance
        )
        cameraNode.look(at: center)
        return cameraNode
    }
}
