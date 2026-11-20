import SwiftUI

@main
struct LidupScannerApp: App {
    @StateObject private var locationProvider: CoreLocationProvider
    @StateObject private var captureController: ARCaptureController

    init() {
        let loc = CoreLocationProvider()
        _locationProvider = StateObject(wrappedValue: loc)
        _captureController = StateObject(wrappedValue: ARCaptureController(locationProvider: loc))
    }

    var body: some Scene {
        WindowGroup {
            TabView {
                CaptureView(
                    controller: captureController,
                    locationProvider: locationProvider,
                    externalStorage: captureController.externalStorage
                )
                    .tabItem {
                        Label("Capturar", systemImage: "camera.metering.center.weighted")
                    }

                SessionsView(externalStorage: captureController.externalStorage)
                    .tabItem {
                        Label("Pacotes", systemImage: "folder")
                    }
            }
            .preferredColorScheme(.dark)
            .tint(.white)
        }
    }
}
