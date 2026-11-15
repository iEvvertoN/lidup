import Foundation
import ARKit
import AVFoundation
import Combine
import simd

enum CapturePreset: String, CaseIterable, Identifiable {
    case terrain = "terrain"
    case object3DGS = "object_3dgs"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .terrain: return "Terreno Rural (5–7 ha)"
        case .object3DGS: return "Objeto / Galpão (3DGS)"
        }
    }

    var shortTitle: String {
        switch self {
        case .terrain: return "Terreno (0.8m)"
        case .object3DGS: return "Objeto / 3DGS (0.25m)"
        }
    }

    var minTranslationM: Float {
        switch self {
        case .terrain: return 0.80
        case .object3DGS: return 0.25
        }
    }

    var minRotationDeg: Float {
        switch self {
        case .terrain: return 12.0
        case .object3DGS: return 8.0
        }
    }
}

/// Controla sessão ARKit + LiDAR, disparo espacial de fotos em alta resolução, filtro de nitidez e estacas GCP.
@MainActor
final class ARCaptureController: NSObject, ObservableObject, @preconcurrency ARSessionDelegate {
    @Published var sectorName: String = "galpao_norte"
    @Published var capturePreset: CapturePreset = .terrain
    @Published var minTranslationM: Float = 0.80
    @Published var minRotationDeg: Float = 12.0
    @Published var lockCameraOptics: Bool = true
    @Published var blurRejectionEnabled: Bool = true
    @Published var showLidarMeshOverlay: Bool = true

    @Published private(set) var isRecording: Bool = false
    @Published private(set) var savedFrameCount: Int = 0
    @Published private(set) var rejectedBlurCount: Int = 0
    @Published private(set) var walkedDistanceM: Float = 0
    @Published private(set) var trackingLabel: String = "Aguardando ARKit"
    @Published private(set) var trackingReady: Bool = false
    @Published private(set) var hasLidarSupport: Bool = false
    @Published private(set) var hasMeshReconstructionSupport: Bool = false
    @Published private(set) var activeVideoResolution: String = "-"
    @Published private(set) var gcpPoints: [GCPPoint] = []
    @Published private(set) var scaleBars: [ScaleBar] = []

    // Estado da coleta estacionária de estaca (GCP)
    @Published private(set) var isCollectingGCP: Bool = false
    @Published private(set) var gcpProgress: Double = 0
    @Published private(set) var gcpSampleCount: Int = 0
    @Published var lastStatusMessage: String?

    let arSession = ARSession()
    let locationProvider: CoreLocationProvider

    private var writer: BundleWriter?
    private var lastKeyframeTransform: simd_float4x4?
    private var lastTrackedPosition: SIMD3<Float>?
    private var latestCameraTransform: simd_float4x4?
    private var isCapturingStill = false
    private var bestSharpness: Float = 0
    private var frameCounter = 0

    // Buffer temporário para média estacionária de GCP
    private var gcpCollectedSamples: [GNSSSample] = []
    private var gcpTask: Task<Void, Never>?

    init(locationProvider: CoreLocationProvider) {
        self.locationProvider = locationProvider
        super.init()
        self.arSession.delegate = self
        self.hasLidarSupport = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        self.hasMeshReconstructionSupport = ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)

        self.locationProvider.onSample = { [weak self] sample in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if self.isRecording {
                    self.writer?.appendGNSS(sample)
                }
                if self.isCollectingGCP {
                    self.gcpCollectedSamples.append(sample)
                    self.gcpSampleCount = self.gcpCollectedSamples.count
                }
            }
        }
    }

    func startARSession() {
        locationProvider.startUpdating()

        guard ARWorldTrackingConfiguration.isSupported else {
            trackingLabel = "ARKit não suportado neste dispositivo"
            trackingReady = false
            return
        }

        let config = ARWorldTrackingConfiguration()
        // Seleciona o formato recomendado para liberar fotos em resolução máxima via `captureHighResolutionFrame`.
        if let hiResFormat = ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing {
            config.videoFormat = hiResFormat
        }
        activeVideoResolution = "\(Int(config.videoFormat.imageResolution.width))×\(Int(config.videoFormat.imageResolution.height))"

        var semantics: ARConfiguration.FrameSemantics = []
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            semantics.insert(.sceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            semantics.insert(.smoothedSceneDepth)
        }
        config.frameSemantics = semantics
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }
        // Em ambientes internos/3DGS evita desvios da bússola perto de metal e paredes; em campo aberto mantém norte magnético.
        config.worldAlignment = (capturePreset == .object3DGS) ? .gravity : .gravityAndHeading

        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
        applyOpticsLockIfNeeded(lock: lockCameraOptics)
    }

    func applyPreset(_ preset: CapturePreset) {
        let alignmentChanged = (capturePreset != preset)
        capturePreset = preset
        minTranslationM = preset.minTranslationM
        minRotationDeg = preset.minRotationDeg
        if alignmentChanged && !isRecording {
            startARSession()
        }
        lastStatusMessage = String(format: "Modo %@: %.2fm / %.0f°", preset.shortTitle, minTranslationM, minRotationDeg)
    }

    func pauseARSession() {
        arSession.pause()
        locationProvider.stopUpdating()
    }

    func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    func startRecording() {
        let minRotRad = minRotationDeg * .pi / 180.0
        let cfg = CaptureConfig(
            minTranslationM: minTranslationM,
            minRotationRad: minRotRad,
            videoStabilizationDisabled: true,
            focusLocked: lockCameraOptics,
            imageWidth: 0,
            imageHeight: 0,
            depthWidth: 0,
            depthHeight: 0,
            depthUnit: "millimeters_uint16"
        )

        do {
            writer = try BundleWriter(
                sectorName: sectorName,
                config: cfg,
                hasLidar: hasLidarSupport
            )
            savedFrameCount = 0
            rejectedBlurCount = 0
            walkedDistanceM = 0
            frameCounter = 0
            bestSharpness = 0
            lastKeyframeTransform = nil
            lastTrackedPosition = nil
            gcpPoints = []
            scaleBars = []
            isRecording = true
            applyOpticsLockIfNeeded(lock: lockCameraOptics)
            lastStatusMessage = "Gravação iniciada em \(writer?.sessionId ?? "")"
        } catch {
            lastStatusMessage = "Erro ao iniciar pacote: \(error.localizedDescription)"
        }
    }

    func stopRecording() {
        isRecording = false
        gcpTask?.cancel()
        isCollectingGCP = false

        guard let currentWriter = writer else { return }
        writer = nil
        currentWriter.finish { [weak self] in
            Task { @MainActor [weak self] in
                self?.lastStatusMessage = "Pacote salvo: \(currentWriter.sessionId)"
            }
        }
    }

    /// Trava foco e exposição na câmera primária do ARKit para evitar focus breathing entre frames.
    func applyOpticsLockIfNeeded(lock: Bool) {
        guard let device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera else {
            return
        }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }

            if lock {
                if device.isFocusModeSupported(.locked) {
                    device.focusMode = .locked
                }
                if device.isExposureModeSupported(.locked) {
                    device.exposureMode = .locked
                }
            } else {
                if device.isFocusModeSupported(.continuousAutoFocus) {
                    device.focusMode = .continuousAutoFocus
                }
                if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposureMode = .continuousAutoExposure
                }
            }
        } catch {
            lastStatusMessage = "Aviso óptica: \(error.localizedDescription)"
        }
    }

    /// Inicia coleta estacionária de GNSS (padrão 15s) para registrar estaca / GCP com média ponderada.
    func startStationaryGCP(id: String, notes: String, durationSeconds: Int = 15) {
        guard isRecording, !isCollectingGCP else { return }
        gcpCollectedSamples = []
        gcpSampleCount = 0
        gcpProgress = 0
        isCollectingGCP = true

        let anchorFrameId: String? = savedFrameCount > 0 ? String(format: "%06d", savedFrameCount - 1) : nil
        let arkitPos: [Float] = {
            if let t = latestCameraTransform?.columns.3 {
                return [t.x, t.y, t.z]
            }
            return [0, 0, 0]
        }()

        gcpTask = Task { [weak self] in
            guard let self else { return }
            let totalTicks = max(1, durationSeconds * 4)
            for tick in 1...totalTicks {
                if Task.isCancelled { return }
                try? await Task.sleep(for: .milliseconds(250))
                self.gcpProgress = Double(tick) / Double(totalTicks)
            }

            self.isCollectingGCP = false
            let avg = CoreLocationProvider.computeWeightedAverage(samples: self.gcpCollectedSamples)
            let point = GCPPoint(
                id: id,
                kind: avg != nil ? "stationary_gnss_avg" : "arkit_only_stake",
                timestampS: Date().timeIntervalSince1970,
                sampleCount: avg?.sampleCount ?? 0,
                lat: avg?.lat,
                lon: avg?.lon,
                altEllipsoidalM: avg?.altEllipsoidalM,
                hAccMeanM: avg?.hAccMeanM,
                vAccMeanM: avg?.vAccMeanM,
                arkitXYZ: arkitPos,
                frameId: anchorFrameId,
                notes: notes
            )
            self.gcpPoints.append(point)
            self.writer?.addGCP(point)
            if let avg {
                self.lastStatusMessage = String(format: "%@ salva (%d amostras, ±%.1fm)", id, avg.sampleCount, avg.hAccMeanM)
            } else {
                self.lastStatusMessage = "\(id) salva (sem fix GNSS; gravada em ARKit)"
            }
        }
    }

    func cancelStationaryGCP() {
        gcpTask?.cancel()
        isCollectingGCP = false
        gcpProgress = 0
    }

    func registerScaleBar(id: String, pointAId: String, pointBId: String, distanceMeters: Double) {
        let bar = ScaleBar(id: id, pointAId: pointAId, pointBId: pointBId, measuredDistanceM: distanceMeters)
        scaleBars.append(bar)
        writer?.addScaleBar(bar)
        lastStatusMessage = String(format: "Trena %@ (%.2fm) salva", id, distanceMeters)
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        latestCameraTransform = frame.camera.transform

        let (label, ready) = Self.describeTracking(frame.camera.trackingState)
        trackingLabel = label
        trackingReady = ready

        guard isRecording, ready, let activeWriter = writer else { return }

        let currentTransform = frame.camera.transform
        let currentPos = SIMD3<Float>(currentTransform.columns.3.x, currentTransform.columns.3.y, currentTransform.columns.3.z)
        if let prevPos = lastTrackedPosition {
            let step = simd_distance(prevPos, currentPos)
            if step < 2.0 {
                walkedDistanceM += step
            }
        }
        lastTrackedPosition = currentPos

        guard !isCapturingStill else { return }
        if let lastT = lastKeyframeTransform, !hasMovedEnough(from: lastT, to: currentTransform) {
            return
        }

        // Usa `sceneDepth` instantâneo do mesmo `ARFrame` e guarda sua pose exata `t0` para não dessincronizar ao girar.
        let depthSourceT0 = frame.sceneDepth ?? frame.smoothedSceneDepth
        let depthSnapshotT0: DepthSnapshot? = depthSourceT0.flatMap {
            DepthSnapshot.capture(from: $0.depthMap, confidenceMap: $0.confidenceMap)
        }
        let depthTransformT0 = currentTransform
        let gnssSnapshot = locationProvider.latestSample
        let fallbackIntrinsics = frame.camera.intrinsics
        let fallbackTimestamp = frame.timestamp

        lastKeyframeTransform = currentTransform
        isCapturingStill = true

        session.captureHighResolutionFrame { [weak self] hiResFrame, _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                defer { self.isCapturingStill = false }
                guard self.isRecording else { return }

                let liveNow = self.arSession.currentFrame
                guard let chosenBuffer = hiResFrame?.capturedImage ?? liveNow?.capturedImage else {
                    return
                }
                let chosenTransform = hiResFrame?.camera.transform ?? liveNow?.camera.transform ?? currentTransform
                let chosenIntrinsics = hiResFrame?.camera.intrinsics ?? liveNow?.camera.intrinsics ?? fallbackIntrinsics
                let chosenTimestamp = hiResFrame?.timestamp ?? liveNow?.timestamp ?? fallbackTimestamp

                var chosenDepth = depthSnapshotT0
                var chosenDepthTransform = depthTransformT0
                if let liveNow,
                   abs(liveNow.timestamp - chosenTimestamp) < 0.045,
                   let liveDepth = liveNow.sceneDepth ?? liveNow.smoothedSceneDepth,
                   let freshDepth = DepthSnapshot.capture(from: liveDepth.depthMap, confidenceMap: liveDepth.confidenceMap) {
                    chosenDepth = freshDepth
                    chosenDepthTransform = liveNow.camera.transform
                }

                let sharpness = SharpnessMeter.score(chosenBuffer) ?? 0
                if self.blurRejectionEnabled && self.savedFrameCount >= 5 {
                    if self.bestSharpness > 0 && sharpness < self.bestSharpness * 0.45 {
                        self.rejectedBlurCount += 1
                        return
                    }
                }
                if sharpness > self.bestSharpness {
                    self.bestSharpness = sharpness
                }

                let idx = self.frameCounter
                self.frameCounter += 1
                self.savedFrameCount += 1

                activeWriter.appendKeyframe(
                    frameIndex: idx,
                    timestampS: chosenTimestamp,
                    pixelBuffer: chosenBuffer,
                    transform: chosenTransform,
                    depthTransform: chosenDepthTransform,
                    intrinsics: chosenIntrinsics,
                    sharpness: sharpness,
                    trackingStateLabel: "normal",
                    depthSnapshot: chosenDepth,
                    gnss: gnssSnapshot
                )
            }
        }
    }

    // MARK: - Helpers

    private func hasMovedEnough(from previous: simd_float4x4, to current: simd_float4x4) -> Bool {
        let p0 = SIMD3<Float>(previous.columns.3.x, previous.columns.3.y, previous.columns.3.z)
        let p1 = SIMD3<Float>(current.columns.3.x, current.columns.3.y, current.columns.3.z)
        if simd_distance(p0, p1) >= minTranslationM {
            return true
        }

        let f0 = simd_normalize(-SIMD3<Float>(previous.columns.2.x, previous.columns.2.y, previous.columns.2.z))
        let f1 = simd_normalize(-SIMD3<Float>(current.columns.2.x, current.columns.2.y, current.columns.2.z))
        let cosine = max(-1.0 as Float, min(1.0 as Float, simd_dot(f0, f1)))
        let minRotRad = minRotationDeg * .pi / 180.0
        return acos(cosine) >= minRotRad
    }

    private static func describeTracking(_ state: ARCamera.TrackingState) -> (String, Bool) {
        switch state {
        case .normal:
            return ("Tracking OK", true)
        case .notAvailable:
            return ("ARKit indisponível", false)
        case .limited(let reason):
            switch reason {
            case .initializing: return ("Inicializando...", false)
            case .excessiveMotion: return ("Movimento rápido!", false)
            case .insufficientFeatures: return ("Poucos detalhes visuais", false)
            case .relocalizing: return ("Relocalizando...", false)
            @unknown default: return ("Tracking limitado", false)
            }
        }
    }
}
