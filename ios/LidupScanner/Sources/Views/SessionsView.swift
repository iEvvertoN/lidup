import SwiftUI
import UIKit

struct SessionsView: View {
    @ObservedObject var externalStorage: ExternalStorageManager
    @State private var bundles: [CapturedBundleSummary] = []
    @State private var shareURL: URL?
    @State private var statusBanner: String?
    @State private var showFolderPicker = false

    private let lidarRed = Color(red: 1.0, green: 0.24, blue: 0.20)

    private var totalBytes: Int64 {
        bundles.reduce(0) { $0 + $1.sizeBytes }
    }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack(spacing: 12) {
                        summaryBox(title: "PACOTES", value: "\(bundles.count)")
                        summaryBox(
                            title: "ESPAÇO",
                            value: ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file)
                        )
                        summaryBox(
                            title: "DESTINO OTG",
                            value: externalStorage.folderName ?? "Não definido",
                            isGreen: externalStorage.hasConfiguredExternalFolder
                        )
                    }
                    .listRowInsets(EdgeInsets(top: 8, leading: 0, bottom: 8, trailing: 0))
                    .listRowBackground(Color.clear)
                }

                if let statusBanner {
                    Section {
                        Text(statusBanner)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                if bundles.isEmpty {
                    ContentUnavailableView(
                        "Nenhum bloco capturado",
                        systemImage: "cube.transparent",
                        description: Text("Os pacotes gravados em campo (iPhone ou Pendrive OTG) aparecerão aqui.")
                    )
                } else {
                    Section("Capturas Gravadas") {
                        ForEach(bundles) { item in
                            VStack(alignment: .leading, spacing: 8) {
                                HStack(spacing: 6) {
                                    Text(item.sectorName)
                                        .font(.headline)
                                    badgePill(
                                        item.isExternal ? "USB OTG" : "iPhone",
                                        color: item.isExternal ? .green : .secondary
                                    )
                                    if item.frameCount == 0 && item.depthCount > 0 {
                                        badgePill("SÓ LIDAR", color: lidarRed)
                                    }
                                    Spacer()
                                    Text(ByteCountFormatter.string(fromByteCount: item.sizeBytes, countStyle: .file))
                                        .font(.caption.monospacedDigit().bold())
                                        .foregroundStyle(.secondary)
                                }

                                Text(item.id)
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)

                                HStack(spacing: 14) {
                                    Label(
                                        item.frameCount == 0 && item.depthCount > 0
                                            ? "\(item.depthCount) mapas depth"
                                            : "\(item.frameCount) fotos · \(item.depthCount) depth",
                                        systemImage: item.frameCount == 0 ? "dot.radiowaves.left.and.right" : "photo.stack"
                                    )
                                    Label("\(item.gcpCount) estacas", systemImage: "mappin")
                                }
                                .font(.caption)
                                .foregroundStyle(.white.opacity(0.82))

                                HStack(spacing: 8) {
                                    if !item.isExternal && externalStorage.hasConfiguredExternalFolder {
                                        Button {
                                            copyToExternal(item)
                                        } label: {
                                            Label("Copiar p/ USB", systemImage: "externaldrive.badge.plus")
                                                .font(.caption.bold())
                                        }
                                        .buttonStyle(.borderedProminent)
                                        .tint(.green)
                                    }

                                    Button {
                                        shareURL = item.url
                                    } label: {
                                        Label("Exportar", systemImage: "square.and.arrow.up")
                                            .font(.caption.bold())
                                    }
                                    .buttonStyle(.bordered)
                                }
                            }
                            .padding(.vertical, 6)
                        }
                        .onDelete(perform: deleteBundles)
                    }
                }
            }
            .navigationTitle("Pacotes LidUp")
            .toolbar {
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button {
                        showFolderPicker = true
                    } label: {
                        Image(systemName: "externaldrive.fill")
                    }
                    Button {
                        reloadBundles()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .onAppear {
                reloadBundles()
            }
            .sheet(isPresented: $showFolderPicker) {
                ExternalFolderPicker { result in
                    showFolderPicker = false
                    switch result {
                    case .success(let url):
                        do {
                            try externalStorage.saveExternalFolder(url: url)
                            statusBanner = "Destino USB selecionado: \(url.lastPathComponent)"
                            reloadBundles()
                        } catch {
                            statusBanner = "Pasta sem permissão de escrita: \(error.localizedDescription)"
                        }
                    case .failure(let err):
                        statusBanner = "Erro ao selecionar pasta: \(err.localizedDescription)"
                    }
                }
            }
            .sheet(item: Binding(
                get: { shareURL.map { ShareableURL(url: $0) } },
                set: { shareURL = $0?.url }
            )) { item in
                ActivityShareView(activityItems: [item.url])
            }
        }
    }

    private func summaryBox(title: String, value: String, isGreen: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 10, weight: .bold))
                .tracking(0.6)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.monospacedDigit().bold())
                .foregroundStyle(isGreen ? .green : .white)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 12))
    }

    private func badgePill(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .bold))
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(color.opacity(0.20), in: Capsule())
            .foregroundStyle(color)
    }

    private func reloadBundles() {
        var loaded: [CapturedBundleSummary] = []
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        loaded.append(contentsOf: scanBundles(in: docs, isExternal: false))

        if let ext = externalStorage.resolveReadableFolderForListing() {
            defer {
                if ext.scoped {
                    ext.url.stopAccessingSecurityScopedResource()
                }
            }
            loaded.append(contentsOf: scanBundles(in: ext.url, isExternal: true))
        }

        bundles = loaded.sorted { $0.createdAt > $1.createdAt }
    }

    private func scanBundles(in rootDir: URL, isExternal: Bool) -> [CapturedBundleSummary] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: rootDir,
            includingPropertiesForKeys: [.isDirectoryKey, .creationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var loaded: [CapturedBundleSummary] = []
        let decoder = JSONDecoder()

        for url in entries {
            let manifestURL = url.appendingPathComponent("manifest.json")
            guard let mData = try? Data(contentsOf: manifestURL),
                  let manifest = try? decoder.decode(BundleManifest.self, from: mData) else {
                continue
            }

            let imagesDir = url.appendingPathComponent("images")
            let depthDir = url.appendingPathComponent("depth")
            let frameCount = (try? fm.contentsOfDirectory(atPath: imagesDir.path).count) ?? 0
            let depthCount = (try? fm.contentsOfDirectory(atPath: depthDir.path).count) ?? 0

            let gcpsURL = url.appendingPathComponent("gcps.json")
            let gcpCount: Int = {
                guard let gData = try? Data(contentsOf: gcpsURL),
                      let gFile = try? decoder.decode(GCPFile.self, from: gData) else {
                    return 0
                }
                return gFile.points.count
            }()

            let created = (try? url.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date()
            let size = directorySize(url: url)

            loaded.append(
                CapturedBundleSummary(
                    id: manifest.sessionId,
                    url: url,
                    sectorName: manifest.sectorName,
                    createdAt: created,
                    frameCount: frameCount,
                    depthCount: depthCount,
                    gcpCount: gcpCount,
                    sizeBytes: size,
                    isExternal: isExternal
                )
            )
        }
        return loaded
    }

    private func copyToExternal(_ item: CapturedBundleSummary) {
        guard let ext = externalStorage.acquireWritableRootURL() else {
            statusBanner = "Pendrive OTG não conectado ou sem permissão de escrita."
            return
        }
        defer {
            if ext.scoped {
                ext.url.stopAccessingSecurityScopedResource()
            }
        }
        let destURL = ext.url.appendingPathComponent(item.url.lastPathComponent, isDirectory: true)
        do {
            if FileManager.default.fileExists(atPath: destURL.path) {
                try FileManager.default.removeItem(at: destURL)
            }
            try FileManager.default.copyItem(at: item.url, to: destURL)
            statusBanner = "Copiado para USB: \(item.id)"
            reloadBundles()
        } catch {
            statusBanner = "Falha ao copiar para USB: \(error.localizedDescription)"
        }
    }

    private func deleteBundles(at offsets: IndexSet) {
        for index in offsets {
            let item = bundles[index]
            if item.isExternal, let ext = externalStorage.acquireWritableRootURL() {
                try? FileManager.default.removeItem(at: item.url)
                if ext.scoped {
                    ext.url.stopAccessingSecurityScopedResource()
                }
            } else {
                try? FileManager.default.removeItem(at: item.url)
            }
        }
        bundles.remove(atOffsets: offsets)
    }

    private func directorySize(url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += Int64(size)
            }
        }
        return total
    }
}

private struct ShareableURL: Identifiable {
    let id = UUID()
    let url: URL
}

private struct ActivityShareView: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
