import SwiftUI
import ARKit
import SceneKit
import Metal

/// Exibe o feed ao vivo da `ARSession` usando `ARSCNView` e renderiza os pontos LiDAR em vermelho
/// em tempo real sem interferir no `arSession.delegate`.
struct ARPreviewContainer: UIViewRepresentable {
    let session: ARSession
    var showMeshOverlay: Bool

    func makeCoordinator() -> Coordinator {
        Coordinator(showMeshOverlay: showMeshOverlay)
    }

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = session
        view.delegate = context.coordinator
        view.automaticallyUpdatesLighting = false
        view.rendersCameraGrain = false
        view.rendersMotionBlur = false
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        if uiView.session !== session {
            uiView.session = session
        }
        if context.coordinator.showMeshOverlay != showMeshOverlay {
            context.coordinator.showMeshOverlay = showMeshOverlay
            for node in uiView.scene.rootNode.childNodes {
                node.isHidden = !showMeshOverlay
            }
        }
    }

    final class Coordinator: NSObject, ARSCNViewDelegate {
        var showMeshOverlay: Bool

        private lazy var lidarPointMaterial: SCNMaterial = {
            let mat = SCNMaterial()
            mat.diffuse.contents = UIColor(red: 1.0, green: 0.20, blue: 0.18, alpha: 0.92)
            mat.emission.contents = UIColor(red: 0.95, green: 0.12, blue: 0.12, alpha: 0.45)
            mat.lightingModel = .constant
            mat.isDoubleSided = true
            return mat
        }()

        init(showMeshOverlay: Bool) {
            self.showMeshOverlay = showMeshOverlay
            super.init()
        }

        func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
            guard let meshAnchor = anchor as? ARMeshAnchor else { return nil }
            let node = SCNNode()
            node.geometry = SCNGeometry.lidarPointsFromMeshAnchor(meshAnchor, material: lidarPointMaterial)
            node.isHidden = !showMeshOverlay
            return node
        }

        func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
            guard let meshAnchor = anchor as? ARMeshAnchor else { return }
            node.geometry = SCNGeometry.lidarPointsFromMeshAnchor(meshAnchor, material: lidarPointMaterial)
            node.isHidden = !showMeshOverlay
        }
    }
}

private extension SCNGeometry {
    static func lidarPointsFromMeshAnchor(_ meshAnchor: ARMeshAnchor, material: SCNMaterial) -> SCNGeometry? {
        let geometry = meshAnchor.geometry
        let vertices = geometry.vertices
        guard vertices.count > 0, vertices.buffer.length > 0 else {
            return nil
        }

        // Copia o buffer de vértices para evitar corrida caso o ARKit recicle o buffer Metal.
        let vertexData = Data(bytes: vertices.buffer.contents(), count: vertices.buffer.length)
        let vertexSource = SCNGeometrySource(
            data: vertexData,
            semantic: .vertex,
            vectorCount: vertices.count,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: vertices.offset,
            dataStride: vertices.stride
        )

        var indices = [UInt32](repeating: 0, count: vertices.count)
        for i in 0..<vertices.count {
            indices[i] = UInt32(i)
        }
        let element = SCNGeometryElement(indices: indices, primitiveType: .point)
        element.pointSize = 4.5
        element.minimumPointScreenSpaceRadius = 2.2
        element.maximumPointScreenSpaceRadius = 5.5

        let scnGeometry = SCNGeometry(sources: [vertexSource], elements: [element])
        scnGeometry.materials = [material]
        return scnGeometry
    }
}
