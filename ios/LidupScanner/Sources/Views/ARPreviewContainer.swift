import SwiftUI
import ARKit
import SceneKit
import Metal

/// Exibe o feed ao vivo da `ARSession` usando `ARSCNView` e renderiza a malha LiDAR em tempo real sem interferir no `arSession.delegate`.
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

        private lazy var meshMaterial: SCNMaterial = {
            let mat = SCNMaterial()
            mat.fillMode = .lines
            mat.diffuse.contents = UIColor.systemCyan.withAlphaComponent(0.65)
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
            node.geometry = SCNGeometry.fromMeshAnchor(meshAnchor, material: meshMaterial)
            node.isHidden = !showMeshOverlay
            return node
        }

        func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
            guard let meshAnchor = anchor as? ARMeshAnchor else { return }
            node.geometry = SCNGeometry.fromMeshAnchor(meshAnchor, material: meshMaterial)
            node.isHidden = !showMeshOverlay
        }
    }
}

private extension SCNGeometry {
    static func fromMeshAnchor(_ meshAnchor: ARMeshAnchor, material: SCNMaterial) -> SCNGeometry? {
        let geometry = meshAnchor.geometry
        let vertices = geometry.vertices
        let faces = geometry.faces
        guard vertices.count > 0, faces.count > 0, faces.buffer.length > 0 else {
            return nil
        }

        let vertexSource = SCNGeometrySource(
            buffer: vertices.buffer,
            vertexFormat: vertices.format,
            semantic: .vertex,
            vertexCount: vertices.count,
            dataOffset: vertices.offset,
            dataStride: vertices.stride
        )

        let normals = geometry.normals
        let normalSource = SCNGeometrySource(
            buffer: normals.buffer,
            vertexFormat: normals.format,
            semantic: .normal,
            vertexCount: normals.count,
            dataOffset: normals.offset,
            dataStride: normals.stride
        )

        // Copia os índices para evitar leitura concorrente caso o ARKit recicle o buffer Metal.
        let faceData = Data(bytes: faces.buffer.contents(), count: faces.buffer.length)
        let element = SCNGeometryElement(
            data: faceData,
            primitiveType: .triangles,
            primitiveCount: faces.count,
            bytesPerIndex: faces.bytesPerIndex
        )

        let scnGeometry = SCNGeometry(sources: [vertexSource, normalSource], elements: [element])
        scnGeometry.materials = [material]
        return scnGeometry
    }
}
