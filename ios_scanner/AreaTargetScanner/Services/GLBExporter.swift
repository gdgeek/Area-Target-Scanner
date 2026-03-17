import Foundation
import ARKit
import ModelIO
import MetalKit

/// Exports ARKit mesh data using Apple's ModelIO framework.
///
/// Strategy: ARMeshAnchor → MDLMesh → MDLAsset → OBJ file.
/// ModelIO handles vertex layout, normals, and index buffers correctly.
/// Also exports USDA format which can be viewed natively on Apple devices.
final class MeshExporter {

    enum ExportError: Error, LocalizedError {
        case noMeshData
        case noMetalDevice
        case exportFailed(String)

        var errorDescription: String? {
            switch self {
            case .noMeshData:
                return "没有可导出的网格数据，请确保设备支持 LiDAR 并已扫描足够区域"
            case .noMetalDevice:
                return "无法获取 Metal 设备"
            case .exportFailed(let reason):
                return "模型导出失败: \(reason)"
            }
        }
    }

    /// Exports mesh anchors to OBJ and USDA files in the output directory.
    /// - Parameters:
    ///   - meshAnchors: ARMeshAnchors collected during scanning
    ///   - outputDir: Directory path to write model files
    /// - Returns: Array of exported file paths
    @discardableResult
    static func export(meshAnchors: [ARMeshAnchor], to outputDir: String) throws -> [String] {
        guard !meshAnchors.isEmpty else {
            throw ExportError.noMeshData
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ExportError.noMetalDevice
        }

        let fm = FileManager.default
        let dirURL = URL(fileURLWithPath: outputDir)
        if !fm.fileExists(atPath: outputDir) {
            try fm.createDirectory(at: dirURL, withIntermediateDirectories: true)
        }

        let asset = MDLAsset()

        for anchor in meshAnchors {
            let mdlMesh = anchor.geometry.toMDLMesh(
                device: device,
                transform: anchor.transform
            )
            asset.add(mdlMesh)
        }

        var exportedFiles: [String] = []

        // Export OBJ (universally compatible)
        let objURL = dirURL.appendingPathComponent("model.obj")
        do {
            if fm.fileExists(atPath: objURL.path) {
                try fm.removeItem(at: objURL)
            }
            try asset.export(to: objURL)
            exportedFiles.append(objURL.path)
        } catch {
            throw ExportError.exportFailed("OBJ: \(error.localizedDescription)")
        }

        // Export USDA (native Apple format, viewable in Quick Look)
        let usdaURL = dirURL.appendingPathComponent("model.usda")
        if MDLAsset.canExportFileExtension("usda") {
            do {
                if fm.fileExists(atPath: usdaURL.path) {
                    try fm.removeItem(at: usdaURL)
                }
                try asset.export(to: usdaURL)
                exportedFiles.append(usdaURL.path)
            } catch {
                // USDA export is optional, don't fail the whole export
            }
        }

        return exportedFiles
    }
}

// MARK: - ARMeshGeometry → MDLMesh

extension ARMeshGeometry {

    /// Converts ARMeshGeometry to MDLMesh with world-space transform applied.
    ///
    /// Based on Apple Developer Forums verified approach:
    /// - Uses vertices.stride for correct buffer layout
    /// - Uses faces.bytesPerIndex * faces.count * faces.indexCountPerPrimitive for index data
    /// - Applies world transform so all meshes are in the same coordinate space
    func toMDLMesh(device: MTLDevice, transform: simd_float4x4) -> MDLMesh {
        let allocator = MTKMeshBufferAllocator(device: device)

        // -- Vertices: apply world transform --
        let vCount = vertices.count
        let vStride = vertices.stride
        let srcPtr = vertices.buffer.contents()

        // Allocate transformed vertex data
        var transformedData = Data(count: vStride * vCount)
        transformedData.withUnsafeMutableBytes { destRaw in
            let destBase = destRaw.baseAddress!
            for i in 0..<vCount {
                let srcBase = srcPtr.advanced(by: vertices.offset + i * vStride)
                let px = srcBase.assumingMemoryBound(to: Float.self).pointee
                let py = srcBase.advanced(by: 4).assumingMemoryBound(to: Float.self).pointee
                let pz = srcBase.advanced(by: 8).assumingMemoryBound(to: Float.self).pointee

                let local = SIMD4<Float>(px, py, pz, 1.0)
                let world = transform * local

                let dstBase = destBase.advanced(by: i * vStride)
                dstBase.assumingMemoryBound(to: Float.self).pointee = world.x
                dstBase.advanced(by: 4).assumingMemoryBound(to: Float.self).pointee = world.y
                dstBase.advanced(by: 8).assumingMemoryBound(to: Float.self).pointee = world.z

                // Copy any remaining stride bytes (padding) as-is
                if vStride > 12 {
                    let remaining = vStride - 12
                    let srcExtra = srcBase.advanced(by: 12)
                    let dstExtra = dstBase.advanced(by: 12)
                    dstExtra.copyMemory(from: srcExtra, byteCount: remaining)
                }
            }
        }

        let vertexBuffer = allocator.newBuffer(with: transformedData, type: .vertex)

        // -- Indices --
        let idxByteCount = faces.bytesPerIndex * faces.count * faces.indexCountPerPrimitive
        let indexData = Data(bytes: faces.buffer.contents(), count: idxByteCount)
        let indexBuffer = allocator.newBuffer(with: indexData, type: .index)

        let submesh = MDLSubmesh(
            indexBuffer: indexBuffer,
            indexCount: faces.count * faces.indexCountPerPrimitive,
            indexType: .uInt32,
            geometryType: .triangles,
            material: nil
        )

        // -- Vertex descriptor --
        let vertexDescriptor = MDLVertexDescriptor()
        vertexDescriptor.attributes[0] = MDLVertexAttribute(
            name: MDLVertexAttributePosition,
            format: .float3,
            offset: 0,
            bufferIndex: 0
        )
        vertexDescriptor.layouts[0] = MDLVertexBufferLayout(stride: vStride)

        let mdlMesh = MDLMesh(
            vertexBuffer: vertexBuffer,
            vertexCount: vCount,
            descriptor: vertexDescriptor,
            submeshes: [submesh]
        )

        // Let ModelIO compute proper normals
        mdlMesh.addNormals(withAttributeNamed: MDLVertexAttributeNormal, creaseThreshold: 0.5)

        return mdlMesh
    }
}
