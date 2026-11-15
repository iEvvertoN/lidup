# Estudo Técnico dos Repositórios iOS de Referência (`references/`)

Este documento resume a análise direta do código-fonte dos projetos clonados em `references/` e define o que aproveitar, o que evitar e o que adicionar no app **LidupScanner** (`ios/`) para uso em **topografia rural e reconstrução 3D de zonas de obra**.

---

## 1. Comparativo dos projetos analisados

### A. `references/stray-scanner` (`strayrobots/scanner`)
* **Pontos fortes (aproveitar como espinha dorsal de dados):**
  * Arquitetura limpa de gravação assíncrona com `DispatchSemaphore(value: 3)` para não reter `ARFrame` em memória (`DatasetEncoder.swift`).
  * Exporta um dataset completo e padronizado: `odometry.csv` (pose 6-DoF + intrínsecas por frame `fx, fy, cx, cy` + centro de distorção), `depth/*.png` (16-bit mm), `confidence/*.png` (0, 1, 2), `distortion/*.bin` (tabela radial `Float32`) e `imu.csv` via `CMMotionManager` (`RecordSessionViewController.swift`, `OdometryEncoder.swift`).
  * Já inclui script Python de integração TSDF com Open3D (`scripts/integrate.py`), mostrando exatamente a convenção de eixos (`q_AC` em `OdometryEncoder.swift`).
* **Limitações para o nosso caso rural:**
  * Captura apenas o stream de vídeo (`frame.capturedImage` -> `rgb.mp4`), que tem resolução de vídeo (~2–3 MP), bem abaixo do sensor fotográfico.
  * Amostragem apenas por divisor de FPS fixo (`[1, 2, 4, 12, 60]`): se o operador parar para desviar de um obstáculo ou conferir uma estaca, grava dezenas de frames idênticos; se andar rápido, perde controle de sobreposição.
  * Não registra `CoreLocation` (GPS/altitude/acurácia) nem pontos de controle (GCPs/estacas).
  * Não rejeita frames borrados (*motion blur*).

### B. `references/object-scanner` (`burakSahinkaya/ObjectScanner`)
* **Descobertas técnicas críticas medidas em dispositivo real (`docs/what-limits-quality.md`):**
  * `PhotogrammetrySession` no iOS é limitado a `.reduced` (~25.000 triângulos). Portanto, **não** vale a pena tentar reconstruir o modelo final dentro do iPhone — o iPhone deve atuar como **coletor de alta qualidade** (e preview leve), deixando a reconstrução pesada para o PC.
  * Textura visual é entrada da fotogrametria: paredes lisas ou pasto uniforme sem marcadores fazem o solver perder correspondência; o LiDAR entra justamente para dar escala métrica e suporte onde falta textura.
* **Padrões de código excelentes para reutilizar:**
  * **Gating espacial de keyframes (`RoomKeyframeCollector.swift`):** em vez de FPS fixo, só dispara quando a câmera deslocou `>= minimumTranslation` ou girou `>= minimumRotation` com `trackingState == .normal`.
  * **Fotos em alta resolução dentro da `ARSession` (`RoomKeyframeCollector.swift`):** usa `ARSession.captureHighResolutionFrame`, observando um detalhe fundamental: a resolução entregue depende do `videoFormat` ativo da `ARWorldTrackingConfiguration` — por isso é essencial selecionar `recommendedVideoFormatForHighResolutionFrameCapturing` na configuração da sessão!
  * **Controle de câmera e nitidez (`PhotoCaptureCoordinator.swift` & `SharpnessMeter.swift`):**
    * Desligar estabilização de vídeo (`preferredVideoStabilizationMode = .off`), pois estabilização deforma os pixels em relação às intrínsecas publicadas.
    * Travar/gerenciar foco, exposição e balanço de branco para evitar *focus breathing* (que altera `fx`/`fy` no meio da caminhada).
    * Rejeitar frames com *motion blur* usando energia de gradiente na luminância (`SharpnessMeter.score`).

### C. `references/lidar-depth-map-capture` (`ioridev/LiDAR-Depth-Map-Capture-for-iOS`) & `references/ios-depth-point-cloud` (`Waley-Z/ios-depth-point-cloud`)
* Mostram captura direta de `depthMap` (`Float32`) e `confidenceMap` do `ARFrame` (inclusive salvamento em TIFF `Float32` ou projeção em nuvem de pontos via Metal).
* Reforçam a regra de sempre filtrar pontos com `confidence == .low (0)` antes de acumular geometria.

### D. `references/arkit-markers` (`laanlabs/ARKitMarkers`)
* Demonstra detecção de marcadores **ArUco** integrados ao `ARWorldTrackingConfiguration`.
* **Aplicação direta no terreno rural:** imprimir placas ArUco (ex.: 20×20 cm ou 30×30 cm) para colocar no chão ajuda em duas frentes:
  1. Dá **pontos âncora únicos** em terreno de pasto/terra uniforme;
  2. Permite **registrar/unir blocos 3D vizinhos** e validar **escala métrica** no processador Python (usando OpenCV `aruco` no Windows, sem sequer depender de OpenCV compilado dentro do iOS se quisermos manter o app Swift leve no primeiro momento!).

---

## 2. Arquitetura Alvo do App `LidupScanner` (`ios/`)

Unindo o melhor dessas referências para o cenário rural (blocos de 30×40 m + caminhamento macro):

1. **Sessão ARKit configurada para fotogrametria + LiDAR:**
   * `ARWorldTrackingConfiguration` com `recommendedVideoFormatForHighResolutionFrameCapturing` (priorizando alta resolução no `captureHighResolutionFrame`) + `frameSemantics = [.sceneDepth, .smoothedSceneDepth]`.
   * Preferência por trava de foco/exposição ou modo de obturador rápido para evitar *motion blur*.
2. **Gatilho Inteligente por Deslocamento + Nitidez:**
   * Captura automática de *keyframe* a cada `Δd` configurável (ex.: **0,8 m a 1,2 m** ao caminhar com bastão) ou `Δθ` (ex.: **12°**), **somente** quando `trackingState == .normal` e `SharpnessMeter` acima do limiar.
   * Modo paralelo de **Snapshot Manual / Estaca (GCP)**: botão para parar 15–30 s em cima de um marco/estaca e gravar a média das leituras de GPS (`CLLocationManager`) + foto + profundidade + pose ARKit.
3. **Coleta de Georreferenciamento (`CoreLocation` + futuro NMEA/RTK):**
   * Log contínuo em `gnss.csv` (`timestamp, lat, lon, alt_ellipsoidal, h_acc, v_acc, speed, course`).
   * Arquitetura preparada com protocolo `LocationProvider` para plugar depois um receptor externo BLE/TCP (NMEA/RTK) sem mudar o restante do app.
4. **Exportação direta para o `lidup/processor`:**
   * Cada bloco/setor salvo em uma pasta autocontida (`lidup_bundle_v1`), exportável via Compartilhar / Arquivos / cabo USB.
