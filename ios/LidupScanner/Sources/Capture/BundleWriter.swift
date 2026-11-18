import Foundation
import ARKit
import CoreImage
import ImageIO
import Metal
import UIKit
import simd

/// Wrapper seguro para despachar `CVPixelBuffer` de foto hi-res para codificação JPEG em background.
private final class PixelBufferBox: @unchecked Sendable {
    let buffer: CVPixelBuffer
    init(_ buffer: CVPixelBuffer) {
        self.buffer = buffer
    }
}

/// Cópia segura dos buffers Metal de um `ARMeshAnchor` para escrita em background.
private struct MeshAnchorCopy: Sendable {
    let transform: simd_float4x4
    let vertexData: Data
    let vertexCount: Int
    let vertexStride: Int
    let vertexOffset: Int
    let faceData: Data
    let faceCount: Int
    let bytesPerIndex: Int
}

/// Grava pacotes no formato `lidup_bundle_v1` de forma thread-safe em fila serial dedicada.
final class BundleWriter: @unchecked Sendable {
    let sessionId: String
    let bundleDirectory: URL

    private let imagesDirectory: URL
    private let depthDirectory: URL
    private let confidenceDirectory: URL
    private let manifestURL: URL
    private let framesURL: URL
    private let gnssURL: URL
    private let imuURL: URL
    private let meshURL: URL
    private let gcpsURL: URL

    private let writeQueue = DispatchQueue(label: "com.lidup.scanner.bundleWriter", qos: .userInitiated)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    private var framesHandle: FileHandle?
    private var gnssHandle: FileHandle?
    private var imuHandle: FileHandle?
    private var manifest: BundleManifest
    private var gcpFile = GCPFile(points: [], scaleBars: [])

    /// Alinha o referencial da câmera ARKit (+Y cima, -Z frente) ao padrão OpenCV/Open3D (+Y baixo, +Z frente).
    private static let qAC = simd_quatf(ix: 1.0, iy: 0.0, iz: 0.0, r: 0.0)

    init(sectorName: String, config: CaptureConfig, hasLidar: Bool) throws {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HHmmss"
        let timestampPrefix = formatter.string(from: Date())
        let cleanSector = sectorName
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: " ", with: "_")
            .filter { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" }
        let safeSector = cleanSector.isEmpty ? "setor" : cleanSector

        self.sessionId = "\(timestampPrefix)_\(safeSector)"

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        self.bundleDirectory = docs.appendingPathComponent(sessionId, isDirectory: true)
        self.imagesDirectory = bundleDirectory.appendingPathComponent("images", isDirectory: true)
        self.depthDirectory = bundleDirectory.appendingPathComponent("depth", isDirectory: true)
        self.confidenceDirectory = bundleDirectory.appendingPathComponent("confidence", isDirectory: true)
        self.manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        self.framesURL = bundleDirectory.appendingPathComponent("frames.csv")
        self.gnssURL = bundleDirectory.appendingPathComponent("gnss.csv")
        self.imuURL = bundleDirectory.appendingPathComponent("imu.csv")
        self.meshURL = bundleDirectory.appendingPathComponent("anchors_mesh.obj")
        self.gcpsURL = bundleDirectory.appendingPathComponent("gcps.json")

        let fm = FileManager.default
        try fm.createDirectory(at: imagesDirectory, withIntermediateDirectories: true)
        try fm.createDirectory(at: depthDirectory, withIntermediateDirectories: true)
        try fm.createDirectory(at: confidenceDirectory, withIntermediateDirectories: true)

        let iso = ISO8601DateFormatter().string(from: Date())
        self.manifest = BundleManifest(
            schemaVersion: "lidup_bundle_v1",
            sessionId: sessionId,
            sectorName: safeSector,
            createdAtIso: iso,
            device: DeviceInfo(
                model: Self.deviceIdentifier(),
                systemVersion: UIDevice.current.systemVersion,
                hasLidar: hasLidar
            ),
            captureConfig: config,
            gnssSource: "apple_core_location"
        )

        let framesHeader = "frame_id,timestamp_s,image_file,depth_file,confidence_file,tx,ty,tz,qx,qy,qz,qw,fx,fy,cx,cy,sharpness,tracking_state,lat,lon,alt_m,h_acc_m,v_acc_m,depth_tx,depth_ty,depth_tz,depth_qx,depth_qy,depth_qz,depth_qw,img_w,img_h,exposure_s,iso,lens_pos\n"
        try framesHeader.write(to: framesURL, atomically: true, encoding: .utf8)
        self.framesHandle = try FileHandle(forWritingTo: framesURL)
        try self.framesHandle?.seekToEnd()

        let gnssHeader = "timestamp_s,lat,lon,alt_ellipsoidal_m,h_acc_m,v_acc_m,speed_mps,course_deg,fix_quality,source\n"
        try gnssHeader.write(to: gnssURL, atomically: true, encoding: .utf8)
        self.gnssHandle = try FileHandle(forWritingTo: gnssURL)
        try self.gnssHandle?.seekToEnd()

        let imuHeader = "timestamp_s,gx,gy,gz,ax,ay,az,grav_x,grav_y,grav_z,qx,qy,qz,qw\n"
        try imuHeader.write(to: imuURL, atomically: true, encoding: .utf8)
        self.imuHandle = try FileHandle(forWritingTo: imuURL)
        try self.imuHandle?.seekToEnd()

        try persistManifestSync()
        try persistGCPsSync()
    }

    func appendGNSS(_ sample: GNSSSample) {
        writeQueue.async { [weak self] in
            guard let self, let handle = self.gnssHandle else { return }
            let line = String(
                format: "%.6f,%.8f,%.8f,%.3f,%.3f,%.3f,%.3f,%.2f,%d,%@\n",
                sample.timestampS,
                sample.lat,
                sample.lon,
                sample.altEllipsoidalM,
                sample.hAccM,
                sample.vAccM,
                sample.speedMps,
                sample.courseDeg,
                sample.fixQuality,
                sample.source
            )
            if let data = line.data(using: .utf8) {
                try? handle.write(contentsOf: data)
            }
        }
    }

    func appendIMU(_ sample: IMUSample) {
        writeQueue.async { [weak self] in
            guard let self, let handle = self.imuHandle else { return }
            let line = String(
                format: "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                sample.timestampS,
                sample.gx, sample.gy, sample.gz,
                sample.ax, sample.ay, sample.az,
                sample.gravX, sample.gravY, sample.gravZ,
                sample.qx, sample.qy, sample.qz, sample.qw
            )
            if let data = line.data(using: .utf8) {
                try? handle.write(contentsOf: data)
            }
        }
    }

    func appendKeyframe(
        frameIndex: Int,
        timestampS: Double,
        pixelBuffer: CVPixelBuffer?,
        imageSizeHint: CGSize? = nil,
        transform: simd_float4x4,
        depthTransform: simd_float4x4? = nil,
        intrinsics: simd_float3x3,
        sharpness: Float,
        trackingStateLabel: String,
        depthSnapshot: DepthSnapshot? = nil,
        rawDepth: RawDepthSnapshot? = nil,
        gnss: GNSSSample?,
        exposureSeconds: Double? = nil,
        iso: Float? = nil,
        lensPosition: Float? = nil
    ) {
        let frameId = String(format: "%06d", frameIndex)
        let bufferBox = pixelBuffer.map { PixelBufferBox($0) }
        let finalImgW = pixelBuffer.map { CVPixelBufferGetWidth($0) } ?? Int(imageSizeHint?.width ?? 0)
        let finalImgH = pixelBuffer.map { CVPixelBufferGetHeight($0) } ?? Int(imageSizeHint?.height ?? 0)

        writeQueue.async { [weak self] in
            guard let self else { return }

            var encodedJPEG: Data? = nil
            if let box = bufferBox {
                // Codificação JPEG 12 MP fora da Main Thread preservando orientação nativa e intrinsics exatos.
                let ciImage = CIImage(cvPixelBuffer: box.buffer)
                if let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
                   let jpegData = self.ciContext.jpegRepresentation(
                    of: ciImage,
                    colorSpace: colorSpace,
                    options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.95]
                   ) {
                    encodedJPEG = jpegData
                } else {
                    return
                }
            }

            let resolvedDepth = depthSnapshot ?? rawDepth?.convert()

            if encodedJPEG != nil && self.manifest.captureConfig.imageWidth == 0 {
                self.manifest.captureConfig.imageWidth = finalImgW
                self.manifest.captureConfig.imageHeight = finalImgH
                if let resolvedDepth {
                    self.manifest.captureConfig.depthWidth = resolvedDepth.width
                    self.manifest.captureConfig.depthHeight = resolvedDepth.height
                }
                try? self.persistManifestSync()
            } else if self.manifest.captureConfig.depthWidth == 0, let resolvedDepth {
                self.manifest.captureConfig.depthWidth = resolvedDepth.width
                self.manifest.captureConfig.depthHeight = resolvedDepth.height
                try? self.persistManifestSync()
            }

            var relImage = ""
            if let jpegData = encodedJPEG {
                relImage = "images/\(frameId).jpg"
                let imgURL = self.bundleDirectory.appendingPathComponent(relImage)
                do {
                    try jpegData.write(to: imgURL, options: .atomic)
                } catch {
                    return
                }
            }

            var relDepth = ""
            var relConf = ""
            if let resolvedDepth,
               let depthData = Png16Encoder.encode16BitDepthPNG(
                width: resolvedDepth.width,
                height: resolvedDepth.height,
                values: resolvedDepth.depthMillimeters
               ) {
                relDepth = "depth/\(frameId).png"
                try? depthData.write(to: self.bundleDirectory.appendingPathComponent(relDepth), options: .atomic)

                if let conf = resolvedDepth.confidence,
                   let confData = Png16Encoder.encode8BitGrayPNG(
                    width: resolvedDepth.width,
                    height: resolvedDepth.height,
                    values: conf
                   ) {
                    relConf = "confidence/\(frameId).png"
                    try? confData.write(to: self.bundleDirectory.appendingPathComponent(relConf), options: .atomic)
                }
            }

            let t = transform.columns.3
            let q = (simd_quatf(transform) * Self.qAC).vector
            let dTransform = depthTransform ?? transform
            let dt = dTransform.columns.3
            let dq = (simd_quatf(dTransform) * Self.qAC).vector

            let fx = intrinsics.columns.0.x
            let fy = intrinsics.columns.1.y
            let cx = intrinsics.columns.2.x
            let cy = intrinsics.columns.2.y

            let latStr = gnss.map { String(format: "%.8f", $0.lat) } ?? ""
            let lonStr = gnss.map { String(format: "%.8f", $0.lon) } ?? ""
            let altStr = gnss.map { String(format: "%.3f", $0.altEllipsoidalM) } ?? ""
            let hAccStr = gnss.map { String(format: "%.3f", $0.hAccM) } ?? ""
            let vAccStr = gnss.map { String(format: "%.3f", $0.vAccM) } ?? ""
            let expStr = exposureSeconds.map { String(format: "%.6f", $0) } ?? ""
            let isoStr = iso.map { String(format: "%.1f", $0) } ?? ""
            let lensStr = lensPosition.map { String(format: "%.4f", $0) } ?? ""

            let line = "\(frameId),\(String(format: "%.6f", timestampS)),\(relImage),\(relDepth),\(relConf),"
                + "\(t.x),\(t.y),\(t.z),\(q.x),\(q.y),\(q.z),\(q.w),"
                + "\(fx),\(fy),\(cx),\(cy),\(sharpness),\(trackingStateLabel),"
                + "\(latStr),\(lonStr),\(altStr),\(hAccStr),\(vAccStr),"
                + "\(dt.x),\(dt.y),\(dt.z),\(dq.x),\(dq.y),\(dq.z),\(dq.w),\(finalImgW),\(finalImgH),"
                + "\(expStr),\(isoStr),\(lensStr)\n"

            if let data = line.data(using: .utf8) {
                try? self.framesHandle?.write(contentsOf: data)
            }
        }
    }

    func addGCP(_ point: GCPPoint) {
        writeQueue.async { [weak self] in
            guard let self else { return }
            self.gcpFile.points.append(point)
            try? self.persistGCPsSync()
        }
    }

    func addScaleBar(_ bar: ScaleBar) {
        writeQueue.async { [weak self] in
            guard let self else { return }
            self.gcpFile.scaleBars.append(bar)
            try? self.persistGCPsSync()
        }
    }

    func saveARMeshAnchors(_ anchors: [ARMeshAnchor]) {
        guard !anchors.isEmpty else { return }
        var copies: [MeshAnchorCopy] = []
        copies.reserveCapacity(anchors.count)
        for anchor in anchors {
            let geom = anchor.geometry
            let verts = geom.vertices
            let faces = geom.faces
            guard verts.count > 0, faces.count > 0,
                  verts.buffer.length > 0, faces.buffer.length > 0,
                  faces.indexCountPerPrimitive == 3 else {
                continue
            }
            let vData = Data(bytes: verts.buffer.contents(), count: verts.buffer.length)
            let fData = Data(bytes: faces.buffer.contents(), count: faces.buffer.length)
            copies.append(MeshAnchorCopy(
                transform: anchor.transform,
                vertexData: vData,
                vertexCount: verts.count,
                vertexStride: verts.stride,
                vertexOffset: verts.offset,
                faceData: fData,
                faceCount: faces.count,
                bytesPerIndex: faces.bytesPerIndex
            ))
        }
        guard !copies.isEmpty else { return }
        let anchorCopies = copies

        writeQueue.async { [weak self] in
            guard let self else { return }
            var out = "# Lidup ARKit Mesh Reconstruction (mundo ARKit +Y cima)\n"
            var vertexBase = 1

            for mesh in anchorCopies {
                mesh.vertexData.withUnsafeBytes { vRaw in
                    for i in 0..<mesh.vertexCount {
                        let byteOff = mesh.vertexOffset + i * mesh.vertexStride
                        guard byteOff + 12 <= vRaw.count else { break }
                        let lx = vRaw.loadUnaligned(fromByteOffset: byteOff, as: Float.self)
                        let ly = vRaw.loadUnaligned(fromByteOffset: byteOff + 4, as: Float.self)
                        let lz = vRaw.loadUnaligned(fromByteOffset: byteOff + 8, as: Float.self)
                        let worldPos = mesh.transform * SIMD4<Float>(lx, ly, lz, 1.0)
                        out += String(format: "v %.5f %.5f %.5f\n", worldPos.x, worldPos.y, worldPos.z)
                    }
                }

                mesh.faceData.withUnsafeBytes { fRaw in
                    for f in 0..<mesh.faceCount {
                        let triOffset = f * 3
                        if mesh.bytesPerIndex == 4 {
                            let byteOff = triOffset * 4
                            guard byteOff + 12 <= fRaw.count else { break }
                            let i0 = Int(fRaw.loadUnaligned(fromByteOffset: byteOff, as: UInt32.self)) + vertexBase
                            let i1 = Int(fRaw.loadUnaligned(fromByteOffset: byteOff + 4, as: UInt32.self)) + vertexBase
                            let i2 = Int(fRaw.loadUnaligned(fromByteOffset: byteOff + 8, as: UInt32.self)) + vertexBase
                            out += "f \(i0) \(i1) \(i2)\n"
                        } else if mesh.bytesPerIndex == 2 {
                            let byteOff = triOffset * 2
                            guard byteOff + 6 <= fRaw.count else { break }
                            let i0 = Int(fRaw.loadUnaligned(fromByteOffset: byteOff, as: UInt16.self)) + vertexBase
                            let i1 = Int(fRaw.loadUnaligned(fromByteOffset: byteOff + 2, as: UInt16.self)) + vertexBase
                            let i2 = Int(fRaw.loadUnaligned(fromByteOffset: byteOff + 4, as: UInt16.self)) + vertexBase
                            out += "f \(i0) \(i1) \(i2)\n"
                        }
                    }
                }
                vertexBase += mesh.vertexCount
            }

            if let objData = out.data(using: .utf8) {
                try? objData.write(to: self.meshURL, options: .atomic)
            }
        }
    }

    func finish(completion: @escaping @Sendable () -> Void) {
        writeQueue.async { [weak self] in
            guard let self else {
                completion()
                return
            }
            try? self.persistManifestSync()
            try? self.persistGCPsSync()
            try? self.framesHandle?.close()
            try? self.gnssHandle?.close()
            try? self.imuHandle?.close()
            self.framesHandle = nil
            self.gnssHandle = nil
            self.imuHandle = nil
            completion()
        }
    }

    // MARK: - Helpers

    private func persistManifestSync() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(manifest)
        try data.write(to: manifestURL, options: .atomic)
    }

    private func persistGCPsSync() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(gcpFile)
        try data.write(to: gcpsURL, options: .atomic)
    }

    private static func deviceIdentifier() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        return withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0) ?? "iPhone"
            }
        }
    }
}
