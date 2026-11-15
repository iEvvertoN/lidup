from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .bundle import LidupBundle, load_bundle, reconstruct_lidar_point_cloud
from .terrain import export_ply


_ARKIT_TO_ENG_3X3 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# OpenCV (+X dir, +Y baixo, +Z frente) -> NeRF/OpenGL (+X dir, +Y cima, +Z trás)
_OPENCV_TO_OPENGL_4X4 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def _link_or_copy(src: Path, dst: Path, *, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if not copy_images:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def export_gaussian_splatting_dataset(
    bundle_or_dir: LidupBundle | str | Path,
    out_dir: str | Path,
    *,
    min_confidence: int = 1,
    max_depth_m: float = 6.0,
    voxel_size_m: float = 0.02,
    copy_images: bool = False,
) -> dict[str, Any]:
    """
    Exporta um pacote `lidup_bundle_v1` pronto para treinamento de Gaussian Splatting (3DGS):
      - `transforms.json` (Nerfstudio / Splatfacto / Brush)
      - `sparse/0/{cameras.txt, images.txt, points3D.txt}` (COLMAP para Postshot / LichtFeld / gsplat)
      - `points3D.ply` (nuvem inicial densa do LiDAR em coordenadas Z-para-cima)
      - `images/` (hardlink ou cópia das fotos 12 MP)
    """
    bundle = bundle_or_dir if isinstance(bundle_or_dir, LidupBundle) else load_bundle(bundle_or_dir)
    if not bundle.frames:
        raise ValueError("Pacote não possui frames válidos para exportar para Gaussian Splatting.")

    root_out = Path(out_dir)
    sparse_dir = root_out / "sparse" / "0"
    images_out = root_out / "images"
    root_out.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    pcd_arkit = reconstruct_lidar_point_cloud(
        bundle,
        min_confidence=min_confidence,
        max_depth_m=max_depth_m,
        voxel_size_m=voxel_size_m,
    )
    pts_arkit = np.asarray(pcd_arkit.points, dtype=np.float64)
    rgb = np.asarray(pcd_arkit.colors, dtype=np.float64) if pcd_arkit.has_colors() else None
    pts_eng = pts_arkit @ _ARKIT_TO_ENG_3X3.T if len(pts_arkit) > 0 else np.empty((0, 3), dtype=np.float64)

    ply_path = export_ply(pts_eng, rgb, out_path=root_out / "points3D.ply")

    t_arkit_to_eng = np.eye(4, dtype=np.float64)
    t_arkit_to_eng[:3, :3] = _ARKIT_TO_ENG_3X3

    cfg = bundle.manifest.get("capture_config", {})
    default_w = int(cfg.get("image_width") or 0)
    default_h = int(cfg.get("image_height") or 0)

    ns_frames: list[dict[str, Any]] = []
    colmap_cameras: list[str] = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    colmap_images: list[str] = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]

    first_meta: tuple[int, int, float, float, float, float] | None = None

    for idx, frame in enumerate(bundle.frames, start=1):
        src_img = bundle.root / frame.image_file
        if not src_img.exists():
            continue

        img_name = src_img.name
        _link_or_copy(src_img, images_out / img_name, copy_images=copy_images)

        img_w, img_h = default_w, default_h
        if img_w <= 0 or img_h <= 0:
            with Image.open(src_img) as im:
                img_w, img_h = im.size

        if first_meta is None:
            first_meta = (img_w, img_h, frame.fx, frame.fy, frame.cx, frame.cy)

        # Mundo Z-up (Engenharia) <- câmera OpenCV
        t_wc_eng = t_arkit_to_eng @ frame.t_wc
        # Nerfstudio / Brush usa câmera OpenGL/NeRF
        c2w_nerf = t_wc_eng @ _OPENCV_TO_OPENGL_4X4

        frame_entry: dict[str, Any] = {
            "file_path": f"images/{img_name}",
            "transform_matrix": c2w_nerf.tolist(),
            "fl_x": frame.fx,
            "fl_y": frame.fy,
            "cx": frame.cx,
            "cy": frame.cy,
            "w": img_w,
            "h": img_h,
            "sharpness": frame.sharpness,
        }
        if frame.depth_file and (bundle.root / frame.depth_file).exists():
            dst_depth = root_out / "depth" / Path(frame.depth_file).name
            _link_or_copy(bundle.root / frame.depth_file, dst_depth, copy_images=copy_images)
            frame_entry["depth_file_path"] = f"depth/{dst_depth.name}"

        ns_frames.append(frame_entry)

        # COLMAP usa extrínsecos world-to-camera (OpenCV) com quaternion (QW, QX, QY, QZ)
        r_wc = t_wc_eng[:3, :3]
        t_wc = t_wc_eng[:3, 3]
        r_cw = r_wc.T
        t_cw = -r_cw @ t_wc
        qx, qy, qz, qw = Rotation.from_matrix(r_cw).as_quat()

        colmap_cameras.append(f"{idx} PINHOLE {img_w} {img_h} {frame.fx:.6f} {frame.fy:.6f} {frame.cx:.6f} {frame.cy:.6f}")
        colmap_images.append(
            f"{idx} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {t_cw[0]:.8f} {t_cw[1]:.8f} {t_cw[2]:.8f} {idx} {img_name}"
        )
        colmap_images.append("")

    if first_meta is None:
        raise ValueError("Nenhuma imagem RGB válida encontrada para exportar.")

    w0, h0, fx0, fy0, cx0, cy0 = first_meta
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": fx0,
        "fl_y": fy0,
        "cx": cx0,
        "cy": cy0,
        "w": w0,
        "h": h0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "ply_file_path": "points3D.ply",
        "frames": ns_frames,
    }
    transforms_path = root_out / "transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")

    (sparse_dir / "cameras.txt").write_text("\n".join(colmap_cameras) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(colmap_images) + "\n", encoding="utf-8")

    points3d_lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    rgb_u8 = (
        np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        if rgb is not None and len(rgb) == len(pts_eng)
        else np.full((len(pts_eng), 3), 180, dtype=np.uint8)
    )
    for pid, (pt, col) in enumerate(zip(pts_eng, rgb_u8), start=1):
        points3d_lines.append(f"{pid} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])} 0.0")
    (sparse_dir / "points3D.txt").write_text("\n".join(points3d_lines) + "\n", encoding="utf-8")

    return {
        "out_dir": str(root_out),
        "transforms_json": str(transforms_path),
        "points3d_ply": str(ply_path),
        "colmap_sparse_dir": str(sparse_dir),
        "frame_count": len(ns_frames),
        "point_count": int(len(pts_eng)),
    }
