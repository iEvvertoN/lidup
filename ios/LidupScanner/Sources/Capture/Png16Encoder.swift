import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import UniformTypeIdentifiers

/// Cópia leve dos buffers de profundidade e confiança para evitar reter `ARFrame` em memória.
struct DepthSnapshot: Sendable {
    let width: Int
    let height: Int
    let depthMillimeters: [UInt16]
    let confidence: [UInt8]?

    static func capture(from depthMap: CVPixelBuffer, confidenceMap: CVPixelBuffer?) -> DepthSnapshot? {
        guard CVPixelBufferGetPixelFormatType(depthMap) == kCVPixelFormatType_DepthFloat32 else {
            return nil
        }

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

        var mmValues = [UInt16](repeating: 0, count: width * height)
        for y in 0..<height {
            let rowPtr = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: Float32.self)
            let rowOffset = y * width
            for x in 0..<width {
                let meters = rowPtr[x]
                if meters.isFinite && meters > 0 {
                    let mm = (meters * 1000.0).rounded()
                    mmValues[rowOffset + x] = UInt16(min(max(mm, 0), Float(UInt16.max)))
                }
            }
        }

        var confValues: [UInt8]? = nil
        if let confidenceMap,
           CVPixelBufferGetPixelFormatType(confidenceMap) == kCVPixelFormatType_OneComponent8 {
            CVPixelBufferLockBaseAddress(confidenceMap, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly) }

            let cW = CVPixelBufferGetWidth(confidenceMap)
            let cH = CVPixelBufferGetHeight(confidenceMap)
            let cStride = CVPixelBufferGetBytesPerRow(confidenceMap)
            if cW == width && cH == height, let cBase = CVPixelBufferGetBaseAddress(confidenceMap) {
                var arr = [UInt8](repeating: 0, count: width * height)
                for y in 0..<cH {
                    let rowPtr = cBase.advanced(by: y * cStride).assumingMemoryBound(to: UInt8.self)
                    let rowOffset = y * cW
                    for x in 0..<cW {
                        arr[rowOffset + x] = rowPtr[x]
                    }
                }
                confValues = arr
            }
        }

        return DepthSnapshot(
            width: width,
            height: height,
            depthMillimeters: mmValues,
            confidence: confValues
        )
    }

    /// Amostra rapidamente a distância central (m) sem copiar todo o buffer, para orientação do operador.
    static func centerDepthMeters(from depthMap: CVPixelBuffer) -> Float? {
        guard CVPixelBufferGetPixelFormatType(depthMap) == kCVPixelFormatType_DepthFloat32 else {
            return nil
        }
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        guard width >= 5, height >= 5, let base = CVPixelBufferGetBaseAddress(depthMap) else {
            return nil
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
        let cy = height / 2
        let cx = width / 2

        var sum: Float = 0
        var count = 0
        for dy in -2...2 {
            let rowPtr = base.advanced(by: (cy + dy) * bytesPerRow).assumingMemoryBound(to: Float32.self)
            for dx in -2...2 {
                let d = rowPtr[cx + dx]
                if d.isFinite && d > 0.05 && d < 12.0 {
                    sum += d
                    count += 1
                }
            }
        }
        return count > 0 ? (sum / Float(count)) : nil
    }
}

/// Codifica PNG 16-bit grayscale (profundidade em mm) e PNG 8-bit grayscale (confiança 0/1/2) em Swift puro.
enum Png16Encoder {
    static func encode16BitDepthPNG(width: Int, height: Int, values: [UInt16]) -> Data? {
        guard values.count == width * height else { return nil }
        let bytesPerRow = width * MemoryLayout<UInt16>.size
        let byteData = values.withUnsafeBytes { Data($0) }
        guard let provider = CGDataProvider(data: byteData as CFData) else { return nil }

        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(
            rawValue: CGBitmapInfo.byteOrder16Little.rawValue | CGImageAlphaInfo.none.rawValue
        )

        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 16,
            bitsPerPixel: 16,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            return nil
        }

        return encodePNG(cgImage: cgImage)
    }

    static func encode8BitGrayPNG(width: Int, height: Int, values: [UInt8]) -> Data? {
        guard values.count == width * height else { return nil }
        let bytesPerRow = width
        let byteData = Data(values)
        guard let provider = CGDataProvider(data: byteData as CFData) else { return nil }

        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)

        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            return nil
        }

        return encodePNG(cgImage: cgImage)
    }

    private static func encodePNG(cgImage: CGImage) -> Data? {
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            return nil
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return out as Data
    }
}
