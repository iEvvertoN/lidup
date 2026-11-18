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
            return LidupRustCore.sharpnessLaplacianBGRA(
                bgraBase: UnsafeRawPointer(base),
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                width: width,
                height: height,
                sampleStep: 4
            )

        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
             kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            guard let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return nil }
            let planeW = CVPixelBufferGetWidthOfPlane(buffer, 0)
            let planeH = CVPixelBufferGetHeightOfPlane(buffer, 0)
            let planeStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
            return LidupRustCore.sharpnessLaplacian(
                yPlaneBase: UnsafeRawPointer(base),
                bytesPerRow: planeStride,
                width: planeW,
                height: planeH,
                sampleStep: 4
            )

        default:
            return nil
        }
    }
}
