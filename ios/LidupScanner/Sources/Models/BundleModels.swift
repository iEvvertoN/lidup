import Foundation
import simd

/// Configuração de captura e metadados gravados no `manifest.json` de cada sessão (`lidup_bundle_v1`).
struct BundleManifest: Codable {
    let schemaVersion: String
    let sessionId: String
    let sectorName: String
    let createdAtIso: String
    let device: DeviceInfo
    var captureConfig: CaptureConfig
    var gnssSource: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case sessionId = "session_id"
        case sectorName = "sector_name"
        case createdAtIso = "created_at_iso"
        case device
        case captureConfig = "capture_config"
        case gnssSource = "gnss_source"
    }
}

struct DeviceInfo: Codable {
    let model: String
    let systemVersion: String
    let hasLidar: Bool

    enum CodingKeys: String, CodingKey {
        case model
        case systemVersion = "system_version"
        case hasLidar = "has_lidar"
    }
}

struct CaptureConfig: Codable {
    var minTranslationM: Float
    var minRotationRad: Float
    var videoStabilizationDisabled: Bool
    var focusLocked: Bool
    var imageWidth: Int
    var imageHeight: Int
    var depthWidth: Int
    var depthHeight: Int
    var depthUnit: String

    enum CodingKeys: String, CodingKey {
        case minTranslationM = "min_translation_m"
        case minRotationRad = "min_rotation_rad"
        case videoStabilizationDisabled = "video_stabilization_disabled"
        case focusLocked = "focus_locked"
        case imageWidth = "image_width"
        case imageHeight = "image_height"
        case depthWidth = "depth_width"
        case depthHeight = "depth_height"
        case depthUnit = "depth_unit"
    }
}

struct GNSSSample: Codable, Sendable {
    let timestampS: Double
    let lat: Double
    let lon: Double
    let altEllipsoidalM: Double
    let hAccM: Double
    let vAccM: Double
    let speedMps: Double
    let courseDeg: Double
    /// 0 = inválido, 1 = GPS autônomo, 4 = RTK Fix, 5 = RTK Float
    let fixQuality: Int
    let source: String
}

struct IMUSample: Sendable {
    let timestampS: Double
    let gx: Double
    let gy: Double
    let gz: Double
    let ax: Double
    let ay: Double
    let az: Double
    let gravX: Double
    let gravY: Double
    let gravZ: Double
    let qx: Double
    let qy: Double
    let qz: Double
    let qw: Double
}

struct GCPPoint: Codable, Identifiable {
    let id: String
    let kind: String
    let timestampS: Double
    let sampleCount: Int
    let lat: Double?
    let lon: Double?
    let altEllipsoidalM: Double?
    let hAccMeanM: Double?
    let vAccMeanM: Double?
    let arkitXYZ: [Float]
    let frameId: String?
    let notes: String

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case timestampS = "timestamp_s"
        case sampleCount = "sample_count"
        case lat
        case lon
        case altEllipsoidalM = "alt_ellipsoidal_m"
        case hAccMeanM = "h_acc_mean_m"
        case vAccMeanM = "v_acc_mean_m"
        case arkitXYZ = "arkit_xyz"
        case frameId = "frame_id"
        case notes
    }
}

struct ScaleBar: Codable, Identifiable {
    let id: String
    let pointAId: String
    let pointBId: String
    let measuredDistanceM: Double

    enum CodingKeys: String, CodingKey {
        case id
        case pointAId = "point_a_id"
        case pointBId = "point_b_id"
        case measuredDistanceM = "measured_distance_m"
    }
}

struct GCPFile: Codable {
    var points: [GCPPoint]
    var scaleBars: [ScaleBar]

    enum CodingKeys: String, CodingKey {
        case points
        case scaleBars = "scale_bars"
    }
}

struct CapturedBundleSummary: Identifiable, Hashable {
    let id: String
    let url: URL
    let sectorName: String
    let createdAt: Date
    let frameCount: Int
    let gcpCount: Int
    let sizeBytes: Int64
}
