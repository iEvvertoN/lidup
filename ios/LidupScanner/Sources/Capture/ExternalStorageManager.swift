import Foundation
import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// Gerencia bookmark security-scoped no iOS para gravar e listar pacotes diretamente
/// em pasta externa escolhida pelo usuário (ex.: pendrive USB-C / OTG no app Arquivos).
@MainActor
final class ExternalStorageManager: ObservableObject {
    private enum Keys {
        static let bookmark = "lidup.externalFolderBookmark"
        static let name = "lidup.externalFolderName"
    }

    @Published private(set) var folderName: String?
    @Published private(set) var hasConfiguredExternalFolder: Bool = false

    init() {
        let savedData = UserDefaults.standard.data(forKey: Keys.bookmark)
        let savedName = UserDefaults.standard.string(forKey: Keys.name)
        self.hasConfiguredExternalFolder = (savedData != nil)
        self.folderName = savedName
    }

    func saveExternalFolder(url: URL) throws {
        let scoped = url.startAccessingSecurityScopedResource()
        defer {
            if scoped {
                url.stopAccessingSecurityScopedResource()
            }
        }

        // Valida permissão real de escrita na unidade selecionada (ex.: pendrive OTG montado).
        let probeURL = url.appendingPathComponent(".lidup_write_test_\(UUID().uuidString)")
        try Data("ok".utf8).write(to: probeURL, options: .atomic)
        try? FileManager.default.removeItem(at: probeURL)

        let bookmark = try url.bookmarkData(
            options: [],
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
        UserDefaults.standard.set(bookmark, forKey: Keys.bookmark)
        UserDefaults.standard.set(url.lastPathComponent, forKey: Keys.name)
        folderName = url.lastPathComponent
        hasConfiguredExternalFolder = true
    }

    func clearExternalFolder() {
        UserDefaults.standard.removeObject(forKey: Keys.bookmark)
        UserDefaults.standard.removeObject(forKey: Keys.name)
        folderName = nil
        hasConfiguredExternalFolder = false
    }

    /// Resolve o diretório externo e confirma que está montado e gravável neste momento.
    /// O chamador (ou `BundleWriter.finish`) é responsável por chamar `stopAccessingSecurityScopedResource()` se `scoped == true`.
    func acquireWritableRootURL() -> (url: URL, scoped: Bool)? {
        guard let data = UserDefaults.standard.data(forKey: Keys.bookmark) else {
            return nil
        }
        var isStale = false
        guard let resolved = try? URL(
            resolvingBookmarkData: data,
            options: [],
            relativeTo: nil,
            bookmarkDataIsStale: &isStale
        ) else {
            return nil
        }

        let scoped = resolved.startAccessingSecurityScopedResource()
        let probeURL = resolved.appendingPathComponent(".lidup_probe_\(UUID().uuidString)")
        do {
            try Data("1".utf8).write(to: probeURL, options: .atomic)
            try? FileManager.default.removeItem(at: probeURL)
        } catch {
            if scoped {
                resolved.stopAccessingSecurityScopedResource()
            }
            return nil
        }

        if isStale, let refreshed = try? resolved.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil) {
            UserDefaults.standard.set(refreshed, forKey: Keys.bookmark)
        }
        return (resolved, scoped)
    }

    /// Resolve o diretório externo para leitura/listagem na aba Pacotes.
    func resolveReadableFolderForListing() -> (url: URL, scoped: Bool)? {
        guard let data = UserDefaults.standard.data(forKey: Keys.bookmark) else {
            return nil
        }
        var isStale = false
        guard let resolved = try? URL(
            resolvingBookmarkData: data,
            options: [],
            relativeTo: nil,
            bookmarkDataIsStale: &isStale
        ) else {
            return nil
        }
        let scoped = resolved.startAccessingSecurityScopedResource()
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: resolved.path, isDirectory: &isDir), isDir.boolValue else {
            if scoped {
                resolved.stopAccessingSecurityScopedResource()
            }
            return nil
        }
        return (resolved, scoped)
    }
}

/// Seletor nativo de pasta (app Arquivos / pendrives USB-C / OTG) usando `UIDocumentPickerViewController`.
struct ExternalFolderPicker: UIViewControllerRepresentable {
    var onPick: (Result<URL, Error>) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onPick: onPick)
    }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.folder])
        picker.allowsMultipleSelection = false
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    final class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (Result<URL, Error>) -> Void

        init(onPick: @escaping (Result<URL, Error>) -> Void) {
            self.onPick = onPick
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            if let first = urls.first {
                onPick(.success(first))
            }
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {}
    }
}
