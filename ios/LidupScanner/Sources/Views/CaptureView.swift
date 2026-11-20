import SwiftUI
import UIKit

struct CaptureView: View {
    @ObservedObject var controller: ARCaptureController
    @ObservedObject var locationProvider: CoreLocationProvider
    @ObservedObject var externalStorage: ExternalStorageManager

    @State private var showGCPDialog = false
    @State private var showScaleDialog = false
    @State private var showSettings = false
    @State private var showFolderPicker = false
    @State private var showSettingsFolderPicker = false
    @State private var showRenameAlert = false
    @State private var draftSectorName = ""

    @State private var stakeName = "ESTACA-01"
    @State private var stakeNotes = ""
    @State private var scaleBarId = "TRENA-01"
    @State private var pointAId = "ESTACA-01"
    @State private var pointBId = "ESTACA-02"
    @State private var scaleDistanceText = "10.0"

    private let lidarRed = Color(red: 1.0, green: 0.24, blue: 0.20)
    private let cardBg = Color.black.opacity(0.72)
    private let borderColor = Color.white.opacity(0.14)

    var body: some View {
        ZStack {
            ARPreviewContainer(
                session: controller.arSession,
                showMeshOverlay: controller.showLidarMeshOverlay
            )
            .ignoresSafeArea()

            centerReticle
                .allowsHitTesting(false)

            VStack(spacing: 10) {
                topStatusBar
                Spacer()
                if controller.isCollectingGCP {
                    gcpProgressBanner
                }
                if let msg = controller.lastStatusMessage {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(controller.isRecording ? lidarRed : .white.opacity(0.7))
                            .frame(width: 6, height: 6)
                        Text(msg)
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.white)
                            .lineLimit(2)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(cardBg, in: Capsule())
                    .overlay(Capsule().stroke(borderColor, lineWidth: 1))
                }
                bottomControls
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
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
        .sheet(isPresented: $showFolderPicker) {
            ExternalFolderPicker { result in
                showFolderPicker = false
                handlePickedFolder(result)
            }
        }
        .alert("Nome do Bloco / Setor", isPresented: $showRenameAlert) {
            TextField("ex: boi_nelore_01", text: $draftSectorName)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            Button("Cancelar", role: .cancel) {}
            Button("Salvar") {
                let cleaned = draftSectorName.trimmingCharacters(in: .whitespacesAndNewlines)
                if !cleaned.isEmpty {
                    controller.sectorName = cleaned
                }
            }
        } message: {
            Text("Defina o nome da pasta que será criada para esta captura.")
        }
    }

    // MARK: - Mira Central + Profundidade ao Vivo

    private var centerReticle: some View {
        let movingFast = controller.isRecording && controller.angularSpeedDegS > 35.0
        let ringColor: Color = movingFast ? .orange : .white.opacity(0.75)

        return VStack(spacing: 6) {
            ZStack {
                Circle()
                    .stroke(ringColor, lineWidth: 1.2)
                    .frame(width: 26, height: 26)
                Circle()
                    .fill(controller.showLidarMeshOverlay ? lidarRed : .white)
                    .frame(width: 4, height: 4)
            }
            if let depthM = controller.centerDepthM {
                Text(String(format: "%.2f m", depthM))
                    .font(.caption2.monospacedDigit().bold())
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.black.opacity(0.65), in: Capsule())
                    .overlay(Capsule().stroke(borderColor, lineWidth: 0.8))
            }
        }
    }

    // MARK: - Barra Superior

    private var topStatusBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                HStack(spacing: 6) {
                    Circle()
                        .fill(controller.trackingReady ? Color.green : Color.yellow)
                        .frame(width: 7, height: 7)
                    Text(controller.trackingLabel)
                        .font(.caption.bold())
                        .foregroundStyle(.white)
                        .lineLimit(1)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(cardBg, in: Capsule())
                .overlay(Capsule().stroke(borderColor, lineWidth: 1))

                if controller.isRecording, let startDate = controller.recordingStartDate {
                    TimelineView(.periodic(from: startDate, by: 1.0)) { context in
                        let secs = max(0, Int(context.date.timeIntervalSince(startDate)))
                        HStack(spacing: 5) {
                            Circle()
                                .fill(lidarRed)
                                .frame(width: 7, height: 7)
                            Text(String(format: "%02d:%02d", secs / 60, secs % 60))
                                .font(.caption.monospacedDigit().bold())
                                .foregroundStyle(.white)
                        }
                        .padding(.horizontal, 9)
                        .padding(.vertical, 6)
                        .background(lidarRed.opacity(0.24), in: Capsule())
                        .overlay(Capsule().stroke(lidarRed.opacity(0.75), lineWidth: 1))
                    }
                }

                Spacer()

                if let gnss = locationProvider.latestSample {
                    Label(String(format: "±%.1fm", gnss.hAccM), systemImage: "location.fill")
                        .font(.caption2.monospacedDigit().bold())
                        .foregroundStyle(gnss.hAccM <= 5 ? .green : .orange)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 6)
                        .background(cardBg, in: Capsule())
                        .overlay(Capsule().stroke(borderColor, lineWidth: 1))
                } else {
                    Label("GPS --", systemImage: "location.slash")
                        .font(.caption2.bold())
                        .foregroundStyle(.white.opacity(0.65))
                        .padding(.horizontal, 9)
                        .padding(.vertical, 6)
                        .background(cardBg, in: Capsule())
                        .overlay(Capsule().stroke(borderColor, lineWidth: 1))
                }

                if controller.hasMeshReconstructionSupport {
                    Button {
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()
                        controller.showLidarMeshOverlay.toggle()
                    } label: {
                        Image(systemName: controller.showLidarMeshOverlay ? "circle.grid.3x3.fill" : "circle.grid.3x3")
                            .font(.subheadline.bold())
                            .foregroundStyle(controller.showLidarMeshOverlay ? lidarRed : .white.opacity(0.55))
                            .frame(width: 34, height: 34)
                            .background(cardBg, in: Circle())
                            .overlay(
                                Circle().stroke(
                                    controller.showLidarMeshOverlay ? lidarRed.opacity(0.75) : borderColor,
                                    lineWidth: 1
                                )
                            )
                    }
                    .accessibilityLabel("Alternar pontos vermelhos LiDAR")
                }

                Button {
                    showSettings = true
                } label: {
                    Image(systemName: "slider.horizontal.3")
                        .font(.subheadline.bold())
                        .foregroundStyle(.white)
                        .frame(width: 34, height: 34)
                        .background(cardBg, in: Circle())
                        .overlay(Circle().stroke(borderColor, lineWidth: 1))
                }
                .disabled(controller.isRecording)
                .opacity(controller.isRecording ? 0.4 : 1.0)
            }

            HStack(spacing: 0) {
                telemetryCell(
                    label: controller.lidarOnlyMode ? "LIDAR" : "FOTOS / DEPTH",
                    value: controller.lidarOnlyMode
                        ? "\(controller.savedDepthOnlyCount) pts"
                        : "\(controller.savedFrameCount) / \(controller.savedDepthOnlyCount)"
                )
                dividerLine
                telemetryCell(
                    label: "PERCURSO",
                    value: String(format: "%.1f m", controller.walkedDistanceM)
                )
                dividerLine
                telemetryCell(
                    label: "GIRO",
                    value: String(format: "%.0f°/s", controller.angularSpeedDegS),
                    highlight: controller.angularSpeedDegS > 35.0
                )
                dividerLine
                telemetryCell(
                    label: "DESTINO",
                    value: controller.isRecording
                        ? (externalStorage.hasConfiguredExternalFolder && controller.activeStorageLabel.hasPrefix("USB") ? "USB OTG" : "iPhone")
                        : (externalStorage.hasConfiguredExternalFolder ? "USB OTG" : "iPhone"),
                    accentGreen: externalStorage.hasConfiguredExternalFolder
                )
            }
            .padding(.vertical, 7)
            .background(cardBg, in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(borderColor, lineWidth: 1))
        }
    }

    private var dividerLine: some View {
        Rectangle()
            .fill(borderColor)
            .frame(width: 1, height: 24)
    }

    private func telemetryCell(
        label: String,
        value: String,
        highlight: Bool = false,
        accentGreen: Bool = false
    ) -> some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.system(size: 9, weight: .bold))
                .tracking(0.6)
                .foregroundStyle(.white.opacity(0.55))
                .lineLimit(1)
            Text(value)
                .font(.caption.monospacedDigit().bold())
                .foregroundStyle(highlight ? .orange : (accentGreen ? .green : .white))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity)
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
                .foregroundStyle(lidarRed)
            }
            ProgressView(value: controller.gcpProgress)
                .tint(.green)
        }
        .padding(12)
        .background(cardBg, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(borderColor, lineWidth: 1))
    }

    // MARK: - Painel Inferior

    private var bottomControls: some View {
        VStack(spacing: 10) {
            if !controller.isRecording {
                HStack(spacing: 8) {
                    Button {
                        draftSectorName = controller.sectorName
                        showRenameAlert = true
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "tag.fill")
                                .font(.caption2)
                                .foregroundStyle(.white.opacity(0.65))
                            Text(controller.sectorName)
                                .font(.caption.monospaced().bold())
                                .foregroundStyle(.white)
                                .lineLimit(1)
                            Image(systemName: "pencil")
                                .font(.caption2)
                                .foregroundStyle(.white.opacity(0.55))
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(cardBg, in: Capsule())
                        .overlay(Capsule().stroke(borderColor, lineWidth: 1))
                    }

                    Spacer()

                    Button {
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()
                        controller.lidarOnlyMode.toggle()
                        controller.lastStatusMessage = controller.lidarOnlyMode
                            ? "Modo Só LiDAR ativo (~90–97% menor)"
                            : "Modo Fotos 12 MP + LiDAR ativo"
                    } label: {
                        Label(
                            controller.lidarOnlyMode ? "Só LiDAR" : "Fotos + LiDAR",
                            systemImage: controller.lidarOnlyMode ? "dot.radiowaves.left.and.right" : "camera.fill"
                        )
                        .font(.caption.bold())
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(
                            controller.lidarOnlyMode ? lidarRed.opacity(0.88) : cardBg,
                            in: Capsule()
                        )
                        .overlay(
                            Capsule().stroke(
                                controller.lidarOnlyMode ? lidarRed : borderColor,
                                lineWidth: 1
                            )
                        )
                        .foregroundStyle(.white)
                    }
                }

                HStack(spacing: 6) {
                    ForEach(CapturePreset.allCases) { preset in
                        let selected = controller.capturePreset == preset
                        Button {
                            UIImpactFeedbackGenerator(style: .light).impactOccurred()
                            controller.applyPreset(preset)
                        } label: {
                            Text(preset.shortTitle)
                                .font(.caption.bold())
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 9)
                                .background(
                                    selected ? Color.white : Color.clear,
                                    in: RoundedRectangle(cornerRadius: 10)
                                )
                                .foregroundStyle(selected ? .black : .white.opacity(0.78))
                        }
                    }
                }
                .padding(4)
                .background(cardBg, in: RoundedRectangle(cornerRadius: 13))
                .overlay(RoundedRectangle(cornerRadius: 13).stroke(borderColor, lineWidth: 1))

                HStack(spacing: 10) {
                    Button {
                        showFolderPicker = true
                    } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 5) {
                                Image(systemName: "externaldrive.fill")
                                    .font(.caption2)
                                Text(externalStorage.hasConfiguredExternalFolder ? "USB OTG" : "MEMÓRIA")
                                    .font(.system(size: 9, weight: .bold))
                                    .tracking(0.5)
                            }
                            .foregroundStyle(externalStorage.hasConfiguredExternalFolder ? .green : .white.opacity(0.6))
                            Text(externalStorage.folderName ?? "iPhone Interno")
                                .font(.caption.bold())
                                .foregroundStyle(.white)
                                .lineLimit(1)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(cardBg, in: RoundedRectangle(cornerRadius: 14))
                        .overlay(
                            RoundedRectangle(cornerRadius: 14).stroke(
                                externalStorage.hasConfiguredExternalFolder ? Color.green.opacity(0.55) : borderColor,
                                lineWidth: 1
                            )
                        )
                    }

                    Button {
                        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                        controller.toggleRecording()
                    } label: {
                        HStack(spacing: 10) {
                            ZStack {
                                Circle()
                                    .stroke(.white, lineWidth: 2)
                                    .frame(width: 24, height: 24)
                                Circle()
                                    .fill(lidarRed)
                                    .frame(width: 14, height: 14)
                            }
                            Text("Iniciar")
                                .font(.headline.bold())
                                .foregroundStyle(.black)
                        }
                        .padding(.horizontal, 24)
                        .padding(.vertical, 14)
                        .background(.white, in: RoundedRectangle(cornerRadius: 16))
                    }
                }
            } else {
                HStack(spacing: 10) {
                    Button {
                        stakeName = String(format: "ESTACA-%02d", controller.gcpPoints.count + 1)
                        stakeNotes = ""
                        showGCPDialog = true
                    } label: {
                        Label("Estaca", systemImage: "mappin.and.ellipse")
                            .font(.caption.bold())
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 13)
                            .background(cardBg, in: RoundedRectangle(cornerRadius: 14))
                            .overlay(RoundedRectangle(cornerRadius: 14).stroke(borderColor, lineWidth: 1))
                            .foregroundStyle(.white)
                    }
                    .disabled(controller.isCollectingGCP)

                    Button {
                        scaleBarId = String(format: "TRENA-%02d", controller.scaleBars.count + 1)
                        showScaleDialog = true
                    } label: {
                        Label("Trena", systemImage: "ruler")
                            .font(.caption.bold())
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 13)
                            .background(cardBg, in: RoundedRectangle(cornerRadius: 14))
                            .overlay(RoundedRectangle(cornerRadius: 14).stroke(borderColor, lineWidth: 1))
                            .foregroundStyle(.white)
                    }

                    Button {
                        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
                        controller.toggleRecording()
                    } label: {
                        HStack(spacing: 8) {
                            RoundedRectangle(cornerRadius: 3)
                                .fill(.white)
                                .frame(width: 14, height: 14)
                            Text("Encerrar")
                                .font(.subheadline.bold())
                        }
                        .padding(.horizontal, 20)
                        .padding(.vertical, 13)
                        .background(lidarRed, in: RoundedRectangle(cornerRadius: 14))
                        .foregroundStyle(.white)
                    }
                }
            }
        }
    }

    // MARK: - Sheets

    private var settingsSheet: some View {
        NavigationStack {
            Form {
                Section("Identificação do Bloco") {
                    TextField("Nome do setor (ex: galpao_norte)", text: $controller.sectorName)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                Section {
                    Picker("Preset", selection: Binding(
                        get: { controller.capturePreset },
                        set: { controller.applyPreset($0) }
                    )) {
                        ForEach(CapturePreset.allCases) { preset in
                            Text(preset.title).tag(preset)
                        }
                    }
                    if controller.hasMeshReconstructionSupport {
                        Toggle("Pontos vermelhos LiDAR ao vivo", isOn: $controller.showLidarMeshOverlay)
                    }
                    Toggle("Modo Só LiDAR (sem fotos 12 MP)", isOn: $controller.lidarOnlyMode)
                    Toggle("Salvar malha ARMesh (anchors_mesh.obj)", isOn: $controller.saveAnchorMesh)
                } header: {
                    Text("Modo de Captura")
                } footer: {
                    Text("O Modo Só LiDAR reduz o peso da pasta em ~90% a 97% e libera captura mais rápida de quadros para animais ou ambientes amplos.")
                }

                Section {
                    HStack {
                        Text("Pasta atual")
                        Spacer()
                        Text(externalStorage.folderName ?? "Memória do iPhone")
                            .foregroundStyle(externalStorage.hasConfiguredExternalFolder ? .green : .secondary)
                    }
                    Button {
                        showSettingsFolderPicker = true
                    } label: {
                        Label("Selecionar Pendrive OTG / Pasta...", systemImage: "externaldrive.fill")
                    }
                    if externalStorage.hasConfiguredExternalFolder {
                        Button("Voltar para Memória do iPhone", role: .destructive) {
                            externalStorage.clearExternalFolder()
                            controller.lastStatusMessage = "Gravação definida para Memória do iPhone"
                        }
                    }
                } header: {
                    Text("Destino de Gravação (Pendrive OTG / Arquivos)")
                } footer: {
                    Text("Escolha uma pasta dentro do seu pendrive USB-C no app Arquivos. Se o pendrive for removido antes de iniciar, o app grava automaticamente no iPhone.")
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
            .sheet(isPresented: $showSettingsFolderPicker) {
                ExternalFolderPicker { result in
                    showSettingsFolderPicker = false
                    handlePickedFolder(result)
                }
            }
        }
    }

    private func handlePickedFolder(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            do {
                try externalStorage.saveExternalFolder(url: url)
                controller.lastStatusMessage = "Destino OTG selecionado: \(url.lastPathComponent)"
            } catch {
                controller.lastStatusMessage = "Pasta sem permissão de escrita: \(error.localizedDescription)"
            }
        case .failure(let err):
            controller.lastStatusMessage = "Erro ao selecionar pasta: \(err.localizedDescription)"
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
