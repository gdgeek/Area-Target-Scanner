import SwiftUI
import SceneKit

/// 3D model preview with SceneKit + debug overlay
struct ModelPreviewView: View {
    let fileURL: URL
    @Environment(\.dismiss) private var dismiss
    @State private var loadError: String? = nil
    @State private var loadSuccess = false

    var body: some View {
        ZStack(alignment: .topLeading) {
            // Dark background instead of white
            Color(red: 0.15, green: 0.15, blue: 0.15).ignoresSafeArea()

            SceneModelView(fileURL: fileURL, onLoad: { success, error in
                loadSuccess = success
                loadError = error
            })
            .ignoresSafeArea()

            // Overlay: close button + debug info
            VStack(alignment: .leading, spacing: 8) {
                Button(action: { dismiss() }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 36))
                        .foregroundStyle(.white, .black.opacity(0.6))
                }
                .padding(.top, 56)

                // Debug info
                VStack(alignment: .leading, spacing: 4) {
                    Text("DEBUG 模型预览").font(.caption.weight(.bold)).foregroundStyle(.yellow)
                    Text("URL: \(fileURL.lastPathComponent)").font(.system(size: 10, design: .monospaced)).foregroundStyle(.white)
                    Text("完整路径: \(fileURL.path)").font(.system(size: 9, design: .monospaced)).foregroundStyle(.white.opacity(0.7))
                    if loadSuccess {
                        Text("✅ 加载成功").font(.caption).foregroundStyle(.green)
                    }
                    if let err = loadError {
                        Text("❌ 错误: \(err)").font(.system(size: 10, design: .monospaced)).foregroundStyle(.red)
                    }
                }
                .padding(8)
                .background(.black.opacity(0.7))
                .cornerRadius(8)
            }
            .padding(.leading, 20)
        }
    }
}

struct SceneModelView: UIViewRepresentable {
    let fileURL: URL
    var onLoad: ((Bool, String?) -> Void)?

    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.backgroundColor = UIColor(red: 0.15, green: 0.15, blue: 0.15, alpha: 1)
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = true

        do {
            let scene = try SCNScene(url: fileURL, options: [
                .checkConsistency: true
            ])
            scnView.scene = scene

            // Count nodes for debug
            var nodeCount = 0
            scene.rootNode.enumerateChildNodes { _, _ in nodeCount += 1 }

            // Auto-frame
            let (minVec, maxVec) = scene.rootNode.boundingBox
            let dx = maxVec.x - minVec.x
            let dy = maxVec.y - minVec.y
            let dz = maxVec.z - minVec.z
            let size = max(dx, max(dy, dz))

            if size > 0 {
                let center = SCNVector3(
                    (minVec.x + maxVec.x) / 2,
                    (minVec.y + maxVec.y) / 2,
                    (minVec.z + maxVec.z) / 2
                )
                let cameraNode = SCNNode()
                cameraNode.camera = SCNCamera()
                cameraNode.camera?.automaticallyAdjustsZRange = true
                cameraNode.position = SCNVector3(center.x, center.y, center.z + size * 2)
                cameraNode.look(at: center)
                scene.rootNode.addChildNode(cameraNode)
                scnView.pointOfView = cameraNode
            }

            DispatchQueue.main.async {
                onLoad?(true, "nodes=\(nodeCount), size=\(String(format: "%.2f", size))")
            }
        } catch {
            DispatchQueue.main.async {
                onLoad?(false, error.localizedDescription)
            }
        }

        return scnView
    }

    func updateUIView(_ uiView: SCNView, context: Context) {}
}
