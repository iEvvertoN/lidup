import SwiftUI

struct CaptureView: View {
    @ObservedObject var controller: ARCaptureController
    @ObservedObject var locationProvider: CoreLocationProvider

    @State private var showGCPDialog = false
    @State private var showScaleDialog = false
    @State private var showSettings = false

    @State private var stakeName = "ESTACA-01"
    @State private var stakeNotes = ""
    @State private var scaleBarId = "TRENA-01"
    @State private var pointAId = "ESTACA-01"
    @State private var pointBId = "ESTACA-02"
    @State private var scaleDistanceText = "10.0"

    var body: some View {
        ZStack {
            ARPreviewContainer(
                session: controller.arSession,
                showMeshOverlay: controller.showLidarMeshOverlay
            )
            .ignoresSafeArea()

            VStack(spacing: 12) {
                topStatusBar
                Spacer()
                if controller.isCollectingGCP {
                    gcpProgressBanner
                }
                if let msg = controller.lastStatusMessage {
                    Text(msg)
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                }
                bottomControls
            }
            .padding()
        }
        .onAppear {
            controller.startARSession()
        }
        .onDisappear {
            if !controller.isRecording {
                controller.pauseARSession()
            }
        }
        .sheet(isPresented: $showSettings) {
            settingsSheet
        }
        .sheet(isPresented: $showGCPDialog) {
            gcpSheet
        }
        .sheet(isPresented: $showScaleDialog) {
            scaleSheet
        }
    }

    private var topStatusBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(controller.trackingLabel, systemImage: controller.trackingReady ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .font(.caption.bold())
                    .foregroundStyle(controller.trackingReady ? .green : .yellow)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.black.opacity(0.65), in: Capsule())

                Spacer()

                if let gnss = locationProvider.latestSample {
                    Label(String(format: "GPS ±%.1fm", gnss.hAccM), systemImage: "location.fill")
                        .font(.caption.bold())
                        .foregroundStyle(gnss.hAccM <= 5 ? .green : .orange)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.black.opacity(0.65), in: Capsule())
                } else {
                    Label("Sem GPS", systemImage: "location.slash")
                        .font(.caption.bold())
                        .foregroundStyle(.red)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.black.opacity(0.65), in: Capsule())
                }

                if controller.hasMeshReconstructionSupport {
                    Button {
                        controller.showLidarMeshOverlay.toggle()
                    } label: {
                        Image(systemName: controller.showLidarMeshOverlay ? "cube.transparent.fill" : "cube.transparent")
                            .font(.body.bold())
                            .foregroundStyle(controller.showLidarMeshOverlay ? .cyan : .white.opacity(0.75))
                            .padding(8)
                            .background(.black.opacity(0.65), in: Circle())
                    }
                    .accessibilityLabel("Alternar malha LiDAR")
                }

                Button {
                    showSettings = true
                } label: {
                    Image(systemName: "slider.horizontal.3")
                        .font(.body.bold())
                        .foregroundStyle(.white)
                        .padding(8)
                        .background(.black.opacity(0.65), in: Circle())
                }
                .disabled(controller.isRecording)
            }

            HStack(spacing: 8) {
                metricPill(title: "Fotos", value: "\(controller.savedFrameCount)")
                metricPill(title: "Depth+", value: "\(controller.savedDepthOnlyCount)")
                metricPill(title: "Dist", value: String(format: "%.1fm", controller.walkedDistanceM))
                metricPill(
                    title: "LiDAR",
                    value: controller.centerDepthM.map { String(format: "%.2fm", $0) } ?? "-"
                )
                metricPill(title: "Gyro", value: String(format: "%.0f°/s", controller.angularSpeedDegS))
            }
        }
    }

    private func metricPill(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.75))
            Text(value)
                .font(.caption.monospacedDigit().bold())
                .foregroundStyle(.white)
                .lineLimit(1)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.black.opacity(0.60), in: RoundedRectangle(cornerRadius: 10))
    }

    private var gcpProgressBanner: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Coletando média GPS da estaca (\(controller.gcpSampleCount) amostras)...")
                    .font(.caption.bold())
                    .foregroundStyle(.white)
                Spacer()
                Button("Cancelar") {
                    controller.cancelStationaryGCP()
                }
                .font(.caption.bold())
                .foregroundStyle(.red)
            }
            ProgressView(value: controller.gcpProgress)
                .tint(.green)
        }
        .padding(12)
        .background(.black.opacity(0.8), in: RoundedRectangle(cornerRadius: 12))
    }

    private var bottomControls: some View {
        VStack(spacing: 12) {
            if !controller.isRecording {
                HStack(spacing: 8) {
                    ForEach(CapturePreset.allCases) { preset in
                        let selected = controller.capturePreset == preset
                        Button {
                            controller.applyPreset(preset)
                        } label: {
                            Text(preset.shortTitle)
                                .font(.caption.bold())
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                                .background(
                                    selected ? Color.cyan.opacity(0.85) : Color.black.opacity(0.65),
                                    in: RoundedRectangle(cornerRadius: 10)
                                )
                                .foregroundStyle(selected ? .black : .white)
                        }
                    }
                }
            }

            if controller.isRecording {
                HStack(spacing: 12) {
                    Button {
                        stakeName = String(format: "ESTACA-%02d", controller.gcpPoints.count + 1)
                        stakeNotes = ""
                        showGCPDialog = true
                    } label: {
                        Label("Estaca (GCP)", systemImage: "mappin.and.ellipse")
                            .font(.subheadline.bold())
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(.blue.opacity(0.9), in: RoundedRectangle(cornerRadius: 12))
                            .foregroundStyle(.white)
                    }
                    .disabled(controller.isCollectingGCP)

                    Button {
                        scaleBarId = String(format: "TRENA-%02d", controller.scaleBars.count + 1)
                        showScaleDialog = true
                    } label: {
                        Label("Trena", systemImage: "ruler")
                            .font(.subheadline.bold())
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(.orange.opacity(0.9), in: RoundedRectangle(cornerRadius: 12))
                            .foregroundStyle(.white)
                    }
                }
            }

            Button {
                controller.toggleRecording()
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: controller.isRecording ? "stop.circle.fill" : "record.circle")
                        .font(.title2)
                    Text(controller.isRecording ? "Encerrar Bloco" : "Iniciar Captura")
                        .font(.headline)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(controller.isRecording ? .red : .green, in: RoundedRectangle(cornerRadius: 16))
                .foregroundStyle(.white)
            }
        }
    }

    private var settingsSheet: some View {
        NavigationStack {
            Form {
                Section("Identificação do Bloco") {
                    TextField("Nome do setor (ex: galpao_norte)", text: $controller.sectorName)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                Section("Modo de Captura") {
                    Picker("Preset", selection: Binding(
                        get: { controller.capturePreset },
                        set: { controller.applyPreset($0) }
                    )) {
                        ForEach(CapturePreset.allCases) { preset in
                            Text(preset.title).tag(preset)
                        }
                    }
                    if controller.hasMeshReconstructionSupport {
                        Toggle("Exibir malha LiDAR ao vivo", isOn: $controller.showLidarMeshOverlay)
                    }
                }
                Section("Gatilho Espacial (ARKit)") {
                    HStack {
                        Text("Distância mínima")
                        Spacer()
                        Text(String(format: "%.2f m", controller.minTranslationM))
                    }
                    Slider(value: $controller.minTranslationM, in: 0.08...2.5, step: 0.02)

                    HStack {
                        Text("Rotação mínima")
                        Spacer()
                        Text(String(format: "%.0f°", controller.minRotationDeg))
                    }
                    Slider(value: $controller.minRotationDeg, in: 3...30, step: 1)
                }
                Section("Física & Sensores (Micro-drone)") {
                    Toggle("Disparo no vale da IMU (menos blur)", isOn: $controller.imuStabilizedCapture)
                    Toggle("Depth contínuo entre fotos 12 MP", isOn: $controller.highDensityDepthEnabled)
                }
                Section("Ótica & Qualidade") {
                    Toggle("Travar foco e exposição", isOn: $controller.lockCameraOptics)
                    Toggle("Descartar fotos borradas (blur)", isOn: $controller.blurRejectionEnabled)
                    HStack {
                        Text("LiDAR disponível")
                        Spacer()
                        Text(controller.hasLidarSupport ? "Sim" : "Não")
                            .foregroundStyle(controller.hasLidarSupport ? .green : .orange)
                    }
                    HStack {
                        Text("Resolução configurada")
                        Spacer()
                        Text(controller.activeVideoResolution)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Configuração de Campo")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("OK") { showSettings = false }
                }
            }
        }
    }

    private var gcpSheet: some View {
        NavigationStack {
            Form {
                Section("Identificação da Estaca / Marco") {
                    TextField("ID (ex: ESTACA-01)", text: $stakeName)
                    TextField("Observação (ex: canto NE platô)", text: $stakeNotes)
                }
                Section {
                    Text("Fique parado sobre o ponto por 15 segundos para coletar a média ponderada de GPS junto com a pose ARKit.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Marcar Estaca (GCP)")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancelar") { showGCPDialog = false }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Iniciar 15s") {
                        showGCPDialog = false
                        controller.startStationaryGCP(id: stakeName, notes: stakeNotes, durationSeconds: 15)
                    }
                }
            }
        }
    }

    private var scaleSheet: some View {
        NavigationStack {
            Form {
                Section("Medida Real com Trena") {
                    TextField("ID", text: $scaleBarId)
                    TextField("Ponto A (ex: ESTACA-01)", text: $pointAId)
                    TextField("Ponto B (ex: ESTACA-02)", text: $pointBId)
                    TextField("Distância medida (m)", text: $scaleDistanceText)
                        .keyboardType(.decimalPad)
                }
            }
            .navigationTitle("Registrar Trena")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancelar") { showScaleDialog = false }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Salvar") {
                        let normalized = scaleDistanceText.replacingOccurrences(of: ",", with: ".")
                        if let dist = Double(normalized), dist > 0 {
                            controller.registerScaleBar(
                                id: scaleBarId,
                                pointAId: pointAId,
                                pointBId: pointBId,
                                distanceMeters: dist
                            )
                        }
                        showScaleDialog = false
                    }
                }
            }
        }
    }
}
