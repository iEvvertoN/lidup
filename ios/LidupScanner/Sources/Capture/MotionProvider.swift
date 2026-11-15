import Foundation
import CoreMotion

/// Provedor inercial de alta frequência (100 Hz) via `CMMotionManager` para log contínuo
/// e sincronização de disparo nos momentos de menor rotação (inspirado em micro-drones).
final class MotionProvider: @unchecked Sendable {
    private let manager = CMMotionManager()
    private let queue: OperationQueue = {
        let q = OperationQueue()
        q.name = "com.lidup.scanner.imu"
        q.maxConcurrentOperationCount = 1
        q.qualityOfService = .userInitiated
        return q
    }()

    private let lock = NSLock()
    private var _latestAngularSpeedRadS: Float = 0
    private weak var _imuWriter: BundleWriter?

    var onSample: (@Sendable (IMUSample) -> Void)?
    var onMetrics: (@Sendable (Float) -> Void)?

    private var sampleCounter = 0

    func setIMUWriter(_ writer: BundleWriter?) {
        lock.lock()
        _imuWriter = writer
        lock.unlock()
    }

    var latestAngularSpeedRadS: Float {
        lock.lock()
        defer { lock.unlock() }
        return _latestAngularSpeedRadS
    }

    func startUpdating() {
        guard manager.isDeviceMotionAvailable else { return }
        guard !manager.isDeviceMotionActive else { return }
        manager.deviceMotionUpdateInterval = 0.01 // 100 Hz
        manager.startDeviceMotionUpdates(using: .xArbitraryCorrectedZVertical, to: queue) { [weak self] motion, _ in
            guard let self, let motion else { return }
            let rate = motion.rotationRate
            let acc = motion.userAcceleration
            let grav = motion.gravity
            let q = motion.attitude.quaternion

            let gyroNorm = Float(sqrt(rate.x * rate.x + rate.y * rate.y + rate.z * rate.z))
            self.lock.lock()
            self._latestAngularSpeedRadS = gyroNorm
            let activeWriter = self._imuWriter
            self.sampleCounter += 1
            let shouldNotifyUI = (self.sampleCounter % 10) == 0
            self.lock.unlock()

            let sample = IMUSample(
                timestampS: motion.timestamp,
                gx: rate.x,
                gy: rate.y,
                gz: rate.z,
                ax: acc.x,
                ay: acc.y,
                az: acc.z,
                gravX: grav.x,
                gravY: grav.y,
                gravZ: grav.z,
                qx: q.x,
                qy: q.y,
                qz: q.z,
                qw: q.w
            )
            activeWriter?.appendIMU(sample)
            self.onSample?(sample)
            if shouldNotifyUI {
                self.onMetrics?(gyroNorm)
            }
        }
    }

    func stopUpdating() {
        if manager.isDeviceMotionActive {
            manager.stopDeviceMotionUpdates()
        }
        lock.lock()
        _latestAngularSpeedRadS = 0
        lock.unlock()
    }
}
