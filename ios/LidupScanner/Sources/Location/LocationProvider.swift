import Foundation
import CoreLocation
import Combine

/// Abstração para provedor GNSS: começa com CoreLocation e permite plugar NMEA/RTK via BLE/TCP no futuro.
@MainActor
protocol LocationProviding: AnyObject {
    var latestSample: GNSSSample? { get }
    var authorizationStatus: CLAuthorizationStatus { get }
    var onSample: ((GNSSSample) -> Void)? { get set }

    func requestPermission()
    func startUpdating()
    func stopUpdating()
}

/// Resultado da média estacionária ponderada por precisão para marcação de estacas/GCPs.
struct StationaryGNSSAverage {
    let sampleCount: Int
    let lat: Double
    let lon: Double
    let altEllipsoidalM: Double
    let hAccMeanM: Double
    let vAccMeanM: Double
}

@MainActor
final class CoreLocationProvider: NSObject, ObservableObject, LocationProviding, @preconcurrency CLLocationManagerDelegate {
    @Published private(set) var latestSample: GNSSSample?
    @Published private(set) var authorizationStatus: CLAuthorizationStatus = .notDetermined

    var onSample: ((GNSSSample) -> Void)?

    private let manager = CLLocationManager()

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBestForNavigation
        manager.distanceFilter = kCLDistanceFilterNone
        manager.activityType = .otherNavigation
        authorizationStatus = manager.authorizationStatus
    }

    func requestPermission() {
        manager.requestWhenInUseAuthorization()
    }

    func startUpdating() {
        if manager.authorizationStatus == .notDetermined {
            manager.requestWhenInUseAuthorization()
        }
        manager.startUpdatingLocation()
    }

    func stopUpdating() {
        manager.stopUpdatingLocation()
    }

    /// Calcula média ponderada pela variância inversa da acurácia horizontal/vertical (via motor Rust).
    static func computeWeightedAverage(samples: [GNSSSample]) -> StationaryGNSSAverage? {
        guard !samples.isEmpty else { return nil }
        guard let avg = LidupRustCore.weightedGNSSAverage(
            lat: samples.map(\.lat),
            lon: samples.map(\.lon),
            alt: samples.map(\.altEllipsoidalM),
            hAcc: samples.map(\.hAccM),
            vAcc: samples.map(\.vAccM)
        ) else {
            return nil
        }
        return StationaryGNSSAverage(
            sampleCount: avg.count,
            lat: avg.lat,
            lon: avg.lon,
            altEllipsoidalM: avg.alt,
            hAccMeanM: avg.hAccMean,
            vAccMeanM: avg.vAccMean
        )
    }

    // MARK: - CLLocationManagerDelegate

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        authorizationStatus = manager.authorizationStatus
        if authorizationStatus == .authorizedWhenInUse || authorizationStatus == .authorizedAlways {
            manager.startUpdatingLocation()
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        for loc in locations where loc.horizontalAccuracy >= 0 {
            let sample = GNSSSample(
                timestampS: loc.timestamp.timeIntervalSince1970,
                lat: loc.coordinate.latitude,
                lon: loc.coordinate.longitude,
                altEllipsoidalM: loc.ellipsoidalAltitude,
                hAccM: loc.horizontalAccuracy,
                vAccM: loc.verticalAccuracy,
                speedMps: max(0, loc.speed),
                courseDeg: loc.course >= 0 ? loc.course : -1,
                fixQuality: 1,
                source: "apple_core_location"
            )
            latestSample = sample
            onSample?(sample)
        }
    }
}
