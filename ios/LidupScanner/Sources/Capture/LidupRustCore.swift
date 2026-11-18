import Foundation
import simd

/// Wrapper Swift seguro sobre a biblioteca estática Rust `lidup_core` (compilada para `aarch64-apple-ios`).
enum LidupRustCore {
    static var abiVersion: UInt32 {
        lidup_core_abi_version()
    }

    static func convertFloat32DepthToMillimeters(
        depthRaw: Data,
        bytesPerRow: Int,
        width: Int,
        height: Int
    ) -> [UInt16]? {
        guard width > 0, height > 0, bytesPerRow >= width * 4 else { return nil }
        var mmValues = [UInt16](repeating: 0, count: width * height)
        let rc: Int32 = depthRaw.withUnsafeBytes { rawBuf in
            guard let srcPtr = rawBuf.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return -1
            }
            return mmValues.withUnsafeMutableBufferPointer { dstBuf in
                guard let dstPtr = dstBuf.baseAddress else { return -1 }
                return lidup_depth_float32_to_mm_u16(
                    srcPtr,
                    depthRaw.count,
                    Int32(bytesPerRow),
                    Int32(width),
                    Int32(height),
                    dstPtr
                )
            }
        }
        return rc == 0 ? mmValues : nil
    }

    static func centerDepthMeters(
        depthBase: UnsafeRawPointer,
        bytesPerRow: Int,
        width: Int,
        height: Int
    ) -> Float? {
        guard width >= 5, height >= 5, bytesPerRow >= width * 4 else { return nil }
        let ptr = depthBase.assumingMemoryBound(to: UInt8.self)
        let val = lidup_center_depth_m_f32(ptr, bytesPerRow * height, Int32(bytesPerRow), Int32(width), Int32(height))
        return val >= 0 ? val : nil
    }

    static func sharpnessLaplacian(
        yPlaneBase: UnsafeRawPointer,
        bytesPerRow: Int,
        width: Int,
        height: Int,
        sampleStep: Int = 2
    ) -> Float {
        guard bytesPerRow >= width, width >= 4, height >= 4 else { return 0 }
        let ptr = yPlaneBase.assumingMemoryBound(to: UInt8.self)
        return lidup_sharpness_laplacian_u8(
            ptr,
            bytesPerRow * height,
            Int32(bytesPerRow),
            Int32(width),
            Int32(height),
            Int32(max(1, sampleStep))
        )
    }

    static func sharpnessLaplacianBGRA(
        bgraBase: UnsafeRawPointer,
        bytesPerRow: Int,
        width: Int,
        height: Int,
        sampleStep: Int = 4
    ) -> Float {
        guard bytesPerRow >= width * 4, width >= 4, height >= 4 else { return 0 }
        let ptr = bgraBase.assumingMemoryBound(to: UInt8.self)
        return lidup_sharpness_laplacian_bgra_u8(
            ptr,
            bytesPerRow * height,
            Int32(bytesPerRow),
            Int32(width),
            Int32(height),
            Int32(max(1, sampleStep))
        )
    }

    static func weightedGNSSAverage(
        lat: [Double],
        lon: [Double],
        alt: [Double],
        hAcc: [Double],
        vAcc: [Double]
    ) -> (count: Int, lat: Double, lon: Double, alt: Double, hAccMean: Double, vAccMean: Double)? {
        let n = min(lat.count, lon.count, alt.count, hAcc.count, vAcc.count)
        guard n > 0 else { return nil }
        var outLat = 0.0
        var outLon = 0.0
        var outAlt = 0.0
        var outHAcc = 0.0
        var outVAcc = 0.0
        let validCount: Int32 = lat.withUnsafeBufferPointer { laP in
            lon.withUnsafeBufferPointer { loP in
                alt.withUnsafeBufferPointer { alP in
                    hAcc.withUnsafeBufferPointer { haP in
                        vAcc.withUnsafeBufferPointer { vaP in
                            guard let laB = laP.baseAddress,
                                  let loB = loP.baseAddress,
                                  let alB = alP.baseAddress,
                                  let haB = haP.baseAddress,
                                  let vaB = vaP.baseAddress else { return 0 }
                            return lidup_weighted_gnss_average(
                                laB, loB, alB, haB, vaB, Int32(n),
                                &outLat, &outLon, &outAlt, &outHAcc, &outVAcc
                            )
                        }
                    }
                }
            }
        }
        guard validCount > 0 else { return nil }
        return (Int(validCount), outLat, outLon, outAlt, outHAcc, outVAcc)
    }

    static func se3HasMovedEnough(
        from previous: simd_float4x4,
        to current: simd_float4x4,
        minDistM: Float,
        minAngleDeg: Float
    ) -> Bool {
        let pArr: [Float] = [
            previous.columns.0.x, previous.columns.0.y, previous.columns.0.z, previous.columns.0.w,
            previous.columns.1.x, previous.columns.1.y, previous.columns.1.z, previous.columns.1.w,
            previous.columns.2.x, previous.columns.2.y, previous.columns.2.z, previous.columns.2.w,
            previous.columns.3.x, previous.columns.3.y, previous.columns.3.z, previous.columns.3.w,
        ]
        let cArr: [Float] = [
            current.columns.0.x, current.columns.0.y, current.columns.0.z, current.columns.0.w,
            current.columns.1.x, current.columns.1.y, current.columns.1.z, current.columns.1.w,
            current.columns.2.x, current.columns.2.y, current.columns.2.z, current.columns.2.w,
            current.columns.3.x, current.columns.3.y, current.columns.3.z, current.columns.3.w,
        ]
        return pArr.withUnsafeBufferPointer { pPtr in
            cArr.withUnsafeBufferPointer { cPtr in
                guard let pBase = pPtr.baseAddress, let cBase = cPtr.baseAddress else { return true }
                return lidup_se3_moved_enough(pBase, cBase, minDistM, minAngleDeg) != 0
            }
        }
    }
}
