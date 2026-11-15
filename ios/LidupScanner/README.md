# LidupScanner (iOS)

App nativo em Swift/SwiftUI para coleta rural de **fotos de alta resolução + LiDAR (`sceneDepth`) + poses ARKit + GNSS/Estacas (GCPs)** no formato [`lidup_bundle_v1`](../../docs/bundle_spec.md).

## Recursos implementados
* `ARWorldTrackingConfiguration` com `recommendedVideoFormatForHighResolutionFrameCapturing` + `.sceneDepth` / `.smoothedSceneDepth`.
* Gatilho espacial automático por distância (`minTranslationM`) e rotação (`minRotationDeg`) apenas com `trackingState == .normal`.
* Filtro de nitidez (`SharpnessMeter`) para descartar quadros com *motion blur*.
* Trava opcional de foco e exposição na câmera primária do ARKit (`configurableCaptureDeviceForPrimaryCamera`) para evitar *focus breathing*.
* Codificação nativa em Swift puro de profundidade em PNG 16-bit (mm) e mapa de confiança em PNG 8-bit (`Png16Encoder`).
* Log contínuo de GNSS (`gnss.csv`) e coleta estacionária ponderada de estacas/GCPs (`gcps.json`) + medidas de trena.
* `UIFileSharingEnabled` habilitado no `Info.plist` para copiar os pacotes direto pelo app **Arquivos** ou via USB no Windows/Finder.

## Como gerar o `.xcodeproj` e compilar (macOS / CI)

1. Instale o [XcodeGen](https://github.com/yonaskolb/XcodeGen):
   ```bash
   brew install xcodegen
   ```
2. Gere o projeto Xcode e abra:
   ```bash
   cd ios/LidupScanner
   xcodegen generate
   open LidupScanner.xcodeproj
   ```
3. No Xcode, selecione o target **LidupScanner → Signing & Capabilities**, escolha seu Apple ID Team (funciona com conta gratuita para instalar via cabo no seu iPhone) e ajuste o `Bundle Identifier` se necessário.
