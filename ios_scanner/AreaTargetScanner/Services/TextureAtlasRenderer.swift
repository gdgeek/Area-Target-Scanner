import Foundation
import CoreGraphics
import ImageIO
import simd

/// Rasterizes camera frame images onto a texture atlas by scanning each
/// triangle face in UV space and sampling colors from the assigned keyframe.
final class TextureAtlasRenderer {

    private let projector = TextureProjector()

    /// Render the texture atlas image.
    /// - Parameters:
    ///   - uvMesh: The UV-unwrapped mesh
    ///   - faceAssignments: Per-face camera frame assignments
    ///   - images: Keyframe JPEG image data
    ///   - cameraPoses: Keyframe camera poses
    ///   - intrinsics: Camera intrinsic parameters
    ///   - atlasSize: Atlas dimensions in pixels (atlasSize × atlasSize)
    /// - Returns: The rendered texture atlas as a CGImage
    func renderAtlas(
        uvMesh: UVMesh,
        faceAssignments: [FaceFrameAssignment],
        images: [CapturedImage],
        cameraPoses: [CameraPose],
        intrinsics: CameraIntrinsics,
        atlasSize: Int
    ) throws -> CGImage {
        // Auto-downgrade atlas size if memory allocation fails
        let sizes = downgradeSizes(from: atlasSize)

        for size in sizes {
            if let result = try? renderAtlasWithSize(
                size: size,
                uvMesh: uvMesh,
                faceAssignments: faceAssignments,
                images: images,
                cameraPoses: cameraPoses,
                intrinsics: intrinsics
            ) {
                return result
            }
            print("[TextureAtlasRenderer] Warning: Failed to allocate \(size)×\(size) atlas, trying smaller size")
        }

        throw TextureMappingError.atlasRenderFailed(reason: "Failed to allocate atlas at any size")
    }

    // MARK: - Private Implementation

    /// Compute the downgrade sequence: e.g. 4096 → 2048 → 1024
    private func downgradeSizes(from atlasSize: Int) -> [Int] {
        var sizes: [Int] = []
        var current = atlasSize
        while current >= 1024 {
            sizes.append(current)
            current /= 2
        }
        if sizes.isEmpty {
            sizes.append(atlasSize)
        }
        return sizes
    }

    /// Core rendering logic for a specific atlas size.
    private func renderAtlasWithSize(
        size: Int,
        uvMesh: UVMesh,
        faceAssignments: [FaceFrameAssignment],
        images: [CapturedImage],
        cameraPoses: [CameraPose],
        intrinsics: CameraIntrinsics
    ) throws -> CGImage {
        // 1. Create bitmap context (RGBA8, atlasSize × atlasSize)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw TextureMappingError.atlasRenderFailed(reason: "CGContext allocation failed for \(size)×\(size)")
        }

        // 2. Fill with default gray (RGB 128, 128, 128)
        context.setFillColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))

        // Get raw pixel buffer for direct pixel writes
        guard let pixelData = context.data else {
            throw TextureMappingError.atlasRenderFailed(reason: "Cannot access pixel buffer")
        }
        let buffer = pixelData.bindMemory(to: UInt8.self, capacity: size * size * 4)

        // 3. Pre-decode only referenced keyframe images (on-demand)
        let decodedImages = preDecodeReferencedImages(
            faceAssignments: faceAssignments,
            images: images
        )

        // Track which pixels were written (for dilation/bleeding)
        var writtenMask = [Bool](repeating: false, count: size * size)
        var rasterizedPixelCount = 0
        var projectionFailCount = 0

        // 4. Rasterize each face
        for i in 0..<uvMesh.faces.count {
            guard i < faceAssignments.count else { break }
            let face = uvMesh.faces[i]
            let assignment = faceAssignments[i]

            // Get decoded source image for this face's assigned frame
            guard let sourcePixels = decodedImages[assignment.frameIndex] else {
                continue // Skip faces with failed image decode
            }

            let pose = cameraPoses[assignment.frameIndex]

            // UV coordinates scaled to atlas pixels
            let uv0 = uvMesh.uvCoordinates[Int(face.x)] * Float(size)
            let uv1 = uvMesh.uvCoordinates[Int(face.y)] * Float(size)
            let uv2 = uvMesh.uvCoordinates[Int(face.z)] * Float(size)

            // 3D world coordinates of face vertices
            let p0 = uvMesh.vertices[Int(face.x)]
            let p1 = uvMesh.vertices[Int(face.y)]
            let p2 = uvMesh.vertices[Int(face.z)]

            // Scanline rasterize the triangle in UV space
            rasterizeTriangle(
                uv0: uv0, uv1: uv1, uv2: uv2,
                p0: p0, p1: p1, p2: p2,
                pose: pose,
                intrinsics: intrinsics,
                sourcePixels: sourcePixels,
                buffer: buffer,
                writtenMask: &writtenMask,
                atlasSize: size,
                rasterizedPixelCount: &rasterizedPixelCount,
                projectionFailCount: &projectionFailCount
            )
        }

        // 5. Dilate texture (padding/bleeding, 2 pixels)
        print("[TextureAtlasRenderer] Rasterized \(rasterizedPixelCount) pixels, \(projectionFailCount) projection failures, \(decodedImages.count) decoded images")
        dilateTexture(buffer: buffer, writtenMask: &writtenMask, atlasSize: size, iterations: 2)

        // 6. Return the rendered image
        guard let image = context.makeImage() else {
            throw TextureMappingError.atlasRenderFailed(reason: "Failed to create CGImage from context")
        }
        return image
    }

    // MARK: - Image Decoding

    /// Decoded image pixel data for efficient sampling.
    struct DecodedImage {
        let pixels: UnsafeMutablePointer<UInt8>
        let width: Int
        let height: Int
        let bytesPerRow: Int
        let context: CGContext // retain to keep pixels alive

        /// Sample a color at normalized coordinates using bilinear interpolation.
        /// u,v are in [0,1] where (0,0) is top-left of the image.
        /// CGBitmapContext pixel buffer has row 0 at the top of the image
        /// (memory order matches CGImage row order), so no Y-flip is needed.
        func sampleBilinear(u: Float, v: Float) -> (UInt8, UInt8, UInt8, UInt8) {
            let fx = u * Float(width - 1)
            let fy = v * Float(height - 1)

            let x0 = Int(fx)
            let y0 = Int(fy)
            let x1 = min(x0 + 1, width - 1)
            let y1 = min(y0 + 1, height - 1)

            let dx = fx - Float(x0)
            let dy = fy - Float(y0)

            let c00 = pixelAt(x: x0, y: y0)
            let c10 = pixelAt(x: x1, y: y0)
            let c01 = pixelAt(x: x0, y: y1)
            let c11 = pixelAt(x: x1, y: y1)

            func lerp(_ a: UInt8, _ b: UInt8, _ t: Float) -> UInt8 {
                UInt8(clamping: Int(Float(a) * (1 - t) + Float(b) * t + 0.5))
            }

            let r = lerp(
                lerp(c00.0, c10.0, dx),
                lerp(c01.0, c11.0, dx),
                dy
            )
            let g = lerp(
                lerp(c00.1, c10.1, dx),
                lerp(c01.1, c11.1, dx),
                dy
            )
            let b = lerp(
                lerp(c00.2, c10.2, dx),
                lerp(c01.2, c11.2, dx),
                dy
            )
            let a = lerp(
                lerp(c00.3, c10.3, dx),
                lerp(c01.3, c11.3, dx),
                dy
            )

            return (r, g, b, a)
        }

        private func pixelAt(x: Int, y: Int) -> (UInt8, UInt8, UInt8, UInt8) {
            let offset = y * bytesPerRow + x * 4
            return (pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3])
        }
    }

    /// Pre-decode only the JPEG images that are actually referenced by face assignments.
    /// Skips frames with failed JPEG decode and logs a warning.
    private func preDecodeReferencedImages(
        faceAssignments: [FaceFrameAssignment],
        images: [CapturedImage]
    ) -> [Int: DecodedImage] {
        // Collect unique referenced frame indices
        var referencedIndices = Set<Int>()
        for assignment in faceAssignments {
            referencedIndices.insert(assignment.frameIndex)
        }

        var decoded: [Int: DecodedImage] = [:]

        for frameIndex in referencedIndices {
            guard frameIndex < images.count else { continue }
            let imageData = images[frameIndex].imageData

            // Decode JPEG from Data
            guard let provider = CGDataProvider(data: imageData as CFData),
                  let cgImage = CGImage(
                    jpegDataProviderSource: provider,
                    decode: nil,
                    shouldInterpolate: true,
                    intent: .defaultIntent
                  )
            else {
                print("[TextureAtlasRenderer] Warning: Failed to decode JPEG for frame \(frameIndex) (\(images[frameIndex].filename)), skipping")
                continue
            }

            // Render into an RGBA context for pixel access
            let w = cgImage.width
            let h = cgImage.height
            let bytesPerRow = w * 4
            let colorSpace = CGColorSpaceCreateDeviceRGB()

            guard let ctx = CGContext(
                data: nil,
                width: w,
                height: h,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ), let pixelPtr = ctx.data else {
                print("[TextureAtlasRenderer] Warning: Failed to create pixel context for frame \(frameIndex), skipping")
                continue
            }

            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

            decoded[frameIndex] = DecodedImage(
                pixels: pixelPtr.bindMemory(to: UInt8.self, capacity: bytesPerRow * h),
                width: w,
                height: h,
                bytesPerRow: bytesPerRow,
                context: ctx
            )
        }

        return decoded
    }

    // MARK: - Triangle Rasterization

    /// Scanline rasterize a triangle in UV space, sampling colors from the source image.
    private func rasterizeTriangle(
        uv0: SIMD2<Float>, uv1: SIMD2<Float>, uv2: SIMD2<Float>,
        p0: SIMD3<Float>, p1: SIMD3<Float>, p2: SIMD3<Float>,
        pose: CameraPose,
        intrinsics: CameraIntrinsics,
        sourcePixels: DecodedImage,
        buffer: UnsafeMutablePointer<UInt8>,
        writtenMask: inout [Bool],
        atlasSize: Int,
        rasterizedPixelCount: inout Int,
        projectionFailCount: inout Int
    ) {
        // Compute bounding box of the triangle in atlas pixel space
        let minX = max(0, Int(floor(min(uv0.x, min(uv1.x, uv2.x)))))
        let maxX = min(atlasSize - 1, Int(ceil(max(uv0.x, max(uv1.x, uv2.x)))))
        let minY = max(0, Int(floor(min(uv0.y, min(uv1.y, uv2.y)))))
        let maxY = min(atlasSize - 1, Int(ceil(max(uv0.y, max(uv1.y, uv2.y)))))

        if minX > maxX || minY > maxY { return }

        // Precompute barycentric denominator
        let denom = (uv1.y - uv2.y) * (uv0.x - uv2.x) + (uv2.x - uv1.x) * (uv0.y - uv2.y)
        if abs(denom) < 1e-10 { return } // Degenerate triangle
        let invDenom = 1.0 / denom

        for y in minY...maxY {
            for x in minX...maxX {
                let px = Float(x) + 0.5
                let py = Float(y) + 0.5

                // Compute barycentric coordinates
                let alpha = ((uv1.y - uv2.y) * (px - uv2.x) + (uv2.x - uv1.x) * (py - uv2.y)) * invDenom
                let beta = ((uv2.y - uv0.y) * (px - uv2.x) + (uv0.x - uv2.x) * (py - uv2.y)) * invDenom
                let gamma = 1.0 - alpha - beta

                // Check if pixel is inside triangle
                if alpha >= 0 && beta >= 0 && gamma >= 0 {
                    // Interpolate 3D world point
                    let worldPoint = alpha * p0 + beta * p1 + gamma * p2

                    // Project to camera image
                    guard let imageUV = projector.projectVertex(
                        worldPoint: worldPoint,
                        cameraPose: pose,
                        intrinsics: intrinsics
                    ) else {
                        projectionFailCount += 1
                        continue
                    }

                    // Sample color from source image using bilinear interpolation
                    let (r, g, b, _) = sourcePixels.sampleBilinear(u: imageUV.x, v: imageUV.y)

                    // Write pixel to atlas
                    // UV convention: v=0 at bottom of texture image.
                    // CGContext memory: row 0 is at lowest address, but CGContext
                    // coordinate y=0 is at the bottom. When makeImage() creates a
                    // CGImage, memory row 0 becomes image row 0 (top of image).
                    // So UV y=0 (bottom) would map to image top without flipping.
                    // We must flip Y so that UV v=0 lands at the image bottom.
                    let flippedY = atlasSize - 1 - y
                    let offset = (flippedY * atlasSize + x) * 4
                    buffer[offset] = r
                    buffer[offset + 1] = g
                    buffer[offset + 2] = b
                    buffer[offset + 3] = 255

                    writtenMask[flippedY * atlasSize + x] = true
                    rasterizedPixelCount += 1
                }
            }
        }
    }

    // MARK: - Texture Dilation (Padding/Bleeding)

    /// Dilate the texture atlas to fill gaps at UV seam boundaries.
    /// For each unwritten pixel adjacent to a written pixel, copy the nearest
    /// written neighbor's color. Repeat for the specified number of iterations.
    private func dilateTexture(
        buffer: UnsafeMutablePointer<UInt8>,
        writtenMask: inout [Bool],
        atlasSize: Int,
        iterations: Int
    ) {
        let offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for _ in 0..<iterations {
            var newPixels: [(x: Int, y: Int, r: UInt8, g: UInt8, b: UInt8)] = []

            for y in 0..<atlasSize {
                for x in 0..<atlasSize {
                    if writtenMask[y * atlasSize + x] { continue }

                    // Check neighbors for written pixels
                    for (dx, dy) in offsets {
                        let nx = x + dx
                        let ny = y + dy
                        if nx >= 0 && nx < atlasSize && ny >= 0 && ny < atlasSize {
                            if writtenMask[ny * atlasSize + nx] {
                                let srcOffset = (ny * atlasSize + nx) * 4
                                newPixels.append((
                                    x: x, y: y,
                                    r: buffer[srcOffset],
                                    g: buffer[srcOffset + 1],
                                    b: buffer[srcOffset + 2]
                                ))
                                break
                            }
                        }
                    }
                }
            }

            // Apply new pixels
            for pixel in newPixels {
                let offset = (pixel.y * atlasSize + pixel.x) * 4
                buffer[offset] = pixel.r
                buffer[offset + 1] = pixel.g
                buffer[offset + 2] = pixel.b
                buffer[offset + 3] = 255
                writtenMask[pixel.y * atlasSize + pixel.x] = true
            }
        }
    }
}
