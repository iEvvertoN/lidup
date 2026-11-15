import CoreVideo

/// Avalia nitidez relativa por energia de gradiente horizontal na luminância.
/// Baseado na técnica usada no `ObjectScanner` para descartar quadros com motion blur antes de salvar.
enum SharpnessMeter {
    static func score(_ buffer: CVPixelBuffer) -> Float? {
        let format = CVPixelBufferGetPixelFormatType(buffer)

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        guard width > 8, height > 8 else { return nil }

        switch format {
        case kCVPixelFormatType_32BGRA:
            guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
            return gradientEnergy(
                base: base,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                width: width,
                height: height,
                bytesPerPixel: 4,
                channelOffset: 1
            )

        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
             kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            guard let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return nil }
            return gradientEnergy(
                base: base,
                bytesPerRow: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0),
                width: CVPixelBufferGetWidthOfPlane(buffer, 0),
                height: CVPixelBufferGetHeightOfPlane(buffer, 0),
                bytesPerPixel: 1,
                channelOffset: 0
            )

        default:
            return nil
        }
    }

    private static func gradientEnergy(
        base: UnsafeMutableRawPointer,
        bytesPerRow: Int,
        width: Int,
        height: Int,
        bytesPerPixel: Int,
        channelOffset: Int
    ) -> Float {
        var total: Double = 0
        var samples = 0

        for row in stride(from: 0, to: height, by: 4) {
            let rowBase = base.advanced(by: row * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            for column in stride(from: 0, to: width - 1, by: 2) {
                let left = Int(rowBase[column * bytesPerPixel + channelOffset])
                let right = Int(rowBase[(column + 1) * bytesPerPixel + channelOffset])
                let diff = Double(right - left)
                total += diff * diff
                samples += 1
            }
        }

        return samples > 0 ? Float(total / Double(samples)) : 0
    }
}
