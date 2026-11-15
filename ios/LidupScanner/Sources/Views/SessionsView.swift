import SwiftUI
import UIKit

struct SessionsView: View {
    @State private var bundles: [CapturedBundleSummary] = []
    @State private var shareURL: URL?

    var body: some View {
        NavigationStack {
            List {
                if bundles.isEmpty {
                    ContentUnavailableView(
                        "Nenhum bloco capturado",
                        systemImage: "cube.transparent",
                        description: Text("Os pacotes gravados em campo aparecerão aqui e também no app Arquivos do iPhone.")
                    )
                } else {
                    ForEach(bundles) { item in
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text(item.sectorName)
                                    .font(.headline)
                                Spacer()
                                Text(ByteCountFormatter.string(fromByteCount: item.sizeBytes, countStyle: .file))
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                            Text(item.id)
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                            HStack(spacing: 16) {
                                Label("\(item.frameCount) frames", systemImage: "photo.stack")
                                Label("\(item.gcpCount) estacas", systemImage: "mappin")
                                Spacer()
                                Button {
                                    shareURL = item.url
                                } label: {
                                    Label("Exportar", systemImage: "square.and.arrow.up")
                                        .font(.caption.bold())
                                }
                                .buttonStyle(.bordered)
                            }
                            .font(.caption)
                        }
                        .padding(.vertical, 4)
                    }
                    .onDelete(perform: deleteBundles)
                }
            }
            .navigationTitle("Pacotes (`lidup`)")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
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
            .sheet(item: Binding(
                get: { shareURL.map { ShareableURL(url: $0) } },
                set: { shareURL = $0?.url }
            )) { item in
                ActivityShareView(activityItems: [item.url])
            }
        }
    }

    private func reloadBundles() {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first!
        guard let entries = try? fm.contentsOfDirectory(
            at: docs,
            includingPropertiesForKeys: [.isDirectoryKey, .creationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            bundles = []
            return
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
            let frameCount = (try? fm.contentsOfDirectory(atPath: imagesDir.path).count) ?? 0

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
                    gcpCount: gcpCount,
                    sizeBytes: size
                )
            )
        }

        bundles = loaded.sorted { $0.createdAt > $1.createdAt }
    }

    private func deleteBundles(at offsets: IndexSet) {
        for index in offsets {
            let item = bundles[index]
            try? FileManager.default.removeItem(at: item.url)
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
