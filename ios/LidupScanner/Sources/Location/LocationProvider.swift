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

    /// Calcula média ponderada pela variância inversa da acurácia horizontal/vertical.
    static func computeWeightedAverage(samples: [GNSSSample]) -> StationaryGNSSAverage? {
        let valid = samples.filter { $0.hAccM > 0 && $0.hAccM < 100 }
        guard !valid.isEmpty else { return nil }

        var wLatSum = 0.0
        var wLonSum = 0.0
        var wHorizTotal = 0.0
        var wAltSum = 0.0
        var wVertTotal = 0.0
        var hAccSum = 0.0
        var vAccSum = 0.0
        var vCount = 0

        for s in valid {
            let wh = 1.0 / max(s.hAccM * s.hAccM, 0.25)
            wLatSum += s.lat * wh
            wLonSum += s.lon * wh
            wHorizTotal += wh
            hAccSum += s.hAccM

            if s.vAccM > 0 {
                let wv = 1.0 / max(s.vAccM * s.vAccM, 0.25)
                wAltSum += s.altEllipsoidalM * wv
                wVertTotal += wv
                vAccSum += s.vAccM
                vCount += 1
            }
        }

        let meanAlt = wVertTotal > 0 ? (wAltSum / wVertTotal) : (valid.map(\.altEllipsoidalM).reduce(0, +) / Double(valid.count))
        let meanVAcc = vCount > 0 ? (vAccSum / Double(vCount)) : -1.0

        return StationaryGNSSAverage(
            sampleCount: valid.count,
            lat: wLatSum / wHorizTotal,
            lon: wLonSum / wHorizTotal,
            altEllipsoidalM: meanAlt,
            hAccMeanM: hAccSum / Double(valid.count),
            vAccMeanM: meanVAcc
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
