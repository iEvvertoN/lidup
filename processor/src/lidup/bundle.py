from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial.transform import Rotation


@dataclass
class FrameRecord:
    frame_id: str
    timestamp_s: float
    image_file: str
    depth_file: str
    confidence_file: str
    t_wc: np.ndarray  # (4, 4) float64, câmera OpenCV (+X dir, +Y baixo, +Z frente) -> mundo ARKit (+Y cima)
    fx: float
    fy: float
    cx: float
    cy: float
    sharpness: float
    tracking_state: str
    lat: float | None
    lon: float | None
    alt_m: float | None
    h_acc_m: float | None
    v_acc_m: float | None
    depth_t_wc: np.ndarray | None = None


@dataclass
class LidupBundle:
    root: Path
    manifest: dict[str, Any]
    frames: list[FrameRecord]
    gcps: dict[str, Any]


def _opt_float(val: str | None) -> float | None:
    if val is None:
        return None
    s = val.strip()
    if not s:
        return None
    return float(s)


def load_bundle(bundle_dir: str | Path) -> LidupBundle:
    """Lê e valida um diretório no formato `lidup_bundle_v1`."""
    root = Path(bundle_dir)
    manifest_path = root / "manifest.json"
    frames_path = root / "frames.csv"
    gcps_path = root / "gcps.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json não encontrado em {root}")
    if not frames_path.exists():
        raise FileNotFoundError(f"frames.csv não encontrado em {root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "lidup_bundle_v1":
        raise ValueError(f"Versão de schema não suportada: {manifest.get('schema_version')}")

    gcps: dict[str, Any] = {"points": [], "scale_bars": []}
    if gcps_path.exists():
        gcps = json.loads(gcps_path.read_text(encoding="utf-8"))

    frames: list[FrameRecord] = []
    with frames_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_wc = np.eye(4, dtype=np.float64)
            quat = [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])]
            t_wc[:3, :3] = Rotation.from_quat(quat).as_matrix()
            t_wc[:3, 3] = [float(row["tx"]), float(row["ty"]), float(row["tz"])]

            depth_t_wc: np.ndarray | None = None
            if all(_opt_float(row.get(k)) is not None for k in ("depth_tx", "depth_ty", "depth_tz", "depth_qx", "depth_qy", "depth_qz", "depth_qw")):
                depth_t_wc = np.eye(4, dtype=np.float64)
                d_quat = [
                    float(row["depth_qx"]),
                    float(row["depth_qy"]),
                    float(row["depth_qz"]),
                    float(row["depth_qw"]),
                ]
                depth_t_wc[:3, :3] = Rotation.from_quat(d_quat).as_matrix()
                depth_t_wc[:3, 3] = [float(row["depth_tx"]), float(row["depth_ty"]), float(row["depth_tz"])]

            frames.append(
                FrameRecord(
                    frame_id=row["frame_id"].strip(),
                    timestamp_s=float(row["timestamp_s"]),
                    image_file=row["image_file"].strip(),
                    depth_file=row["depth_file"].strip(),
                    confidence_file=row["confidence_file"].strip(),
                    t_wc=t_wc,
                    fx=float(row["fx"]),
                    fy=float(row["fy"]),
                    cx=float(row["cx"]),
                    cy=float(row["cy"]),
                    sharpness=float(row["sharpness"]),
                    tracking_state=row["tracking_state"].strip(),
                    lat=_opt_float(row.get("lat")),
                    lon=_opt_float(row.get("lon")),
                    alt_m=_opt_float(row.get("alt_m")),
                    h_acc_m=_opt_float(row.get("h_acc_m")),
                    v_acc_m=_opt_float(row.get("v_acc_m")),
                    depth_t_wc=depth_t_wc,
                )
            )

    return LidupBundle(root=root, manifest=manifest, frames=frames, gcps=gcps)


def reconstruct_lidar_point_cloud(
    bundle: LidupBundle,
    *,
    min_confidence: int = 2,
    min_depth_m: float = 0.30,
    max_depth_m: float = 4.50,
    voxel_size_m: float = 0.03,
    frame_stride: int = 1,
    edge_jump_m: float = 0.12,
    refine_poses: bool = False,
) -> o3d.geometry.PointCloud:
    """
    Reconstrói a nuvem de pontos 3D no referencial local do ARKit a partir dos mapas
    de profundidade LiDAR filtrados por confiança, descontinuidade de borda e coloridos
    pelas fotos RGB (reprojetando quando a pose do depth diferir da foto 12 MP).
    """
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    cfg = bundle.manifest.get("capture_config", {})
    ref_img_w = int(cfg.get("image_width") or 0)
    ref_img_h = int(cfg.get("image_height") or 0)

    icp_map: o3d.geometry.PointCloud | None = None
    icp_voxel = max(0.02, float(voxel_size_m) if voxel_size_m > 0 else 0.02)

    for idx, frame in enumerate(bundle.frames):
        if idx % max(1, frame_stride) != 0:
            continue
        if not frame.depth_file:
            continue

        depth_path = bundle.root / frame.depth_file
        if not depth_path.exists():
            continue

        depth_mm = np.asarray(Image.open(depth_path), dtype=np.uint16)
        depth_m = depth_mm.astype(np.float64) / 1000.0
        d_h, d_w = depth_m.shape[:2]

        mask = (depth_m >= min_depth_m) & (depth_m <= max_depth_m)

        if frame.confidence_file:
            conf_path = bundle.root / frame.confidence_file
            if conf_path.exists():
                conf = np.asarray(Image.open(conf_path), dtype=np.uint8)
                if conf.shape == depth_m.shape:
                    mask &= conf >= min_confidence

        if edge_jump_m > 0 and np.any(mask):
            d_hi = maximum_filter(np.where(mask, depth_m, 0.0), size=3)
            d_lo = minimum_filter(np.where(mask, depth_m, 1e6), size=3)
            mask &= (d_hi - d_lo) <= edge_jump_m

        if not np.any(mask):
            continue

        rgb_arr: np.ndarray | None = None
        img_w = ref_img_w
        img_h = ref_img_h
        pose_depth = frame.depth_t_wc if frame.depth_t_wc is not None else frame.t_wc
        poses_differ = frame.depth_t_wc is not None and not np.allclose(frame.depth_t_wc, frame.t_wc, atol=1e-4)

        if frame.image_file:
            img_path = bundle.root / frame.image_file
            if img_path.exists():
                with Image.open(img_path) as im:
                    rgb_im = im.convert("RGB")
                    img_w, img_h = rgb_im.size
                    if (img_w, img_h) != (d_w, d_h):
                        rgb_im = rgb_im.resize((d_w, d_h), Image.Resampling.BILINEAR)
                    rgb_arr = np.asarray(rgb_im, dtype=np.float64) / 255.0

        if img_w <= 0 or img_h <= 0:
            # Fallback para dimensões aproximadas a partir do centro principal
            img_w = max(d_w, int(round(frame.cx * 2.0)))
            img_h = max(d_h, int(round(frame.cy * 2.0)))

        sx = d_w / float(img_w)
        sy = d_h / float(img_h)
        fx = frame.fx * sx
        fy = frame.fy * sy
        cx = frame.cx * sx
        cy = frame.cy * sy

        v_idx, u_idx = np.nonzero(mask)
        z = depth_m[v_idx, u_idx]
        x = (u_idx.astype(np.float64) - cx) * z / fx
        y = (v_idx.astype(np.float64) - cy) * z / fy

        cam_pts = np.stack([x, y, z], axis=1)
        rot = pose_depth[:3, :3]
        trans = pose_depth[:3, 3]
        world_pts = cam_pts @ rot.T + trans

        if rgb_arr is not None:
            if poses_differ:
                # Reprojeta pontos 3D no instante exato da foto RGB para casar cor sem arrasto.
                r_rgb = frame.t_wc[:3, :3]
                t_rgb = frame.t_wc[:3, 3]
                cam_rgb = (world_pts - t_rgb) @ r_rgb
                z_rgb = cam_rgb[:, 2]
                valid_rgb = z_rgb > 0.1
                u_rgb = np.rint(fx * (cam_rgb[:, 0] / np.where(valid_rgb, z_rgb, 1.0)) + cx).astype(np.int64)
                v_rgb = np.rint(fy * (cam_rgb[:, 1] / np.where(valid_rgb, z_rgb, 1.0)) + cy).astype(np.int64)
                in_bounds = valid_rgb & (u_rgb >= 0) & (u_rgb < d_w) & (v_rgb >= 0) & (v_rgb < d_h)
                frame_colors = np.full((len(world_pts), 3), 0.65, dtype=np.float64)
                frame_colors[in_bounds] = rgb_arr[v_rgb[in_bounds], u_rgb[in_bounds]]
            else:
                frame_colors = rgb_arr[v_idx, u_idx]
        else:
            frame_colors = np.full((len(world_pts), 3), 0.65, dtype=np.float64)

        if refine_poses and len(world_pts) >= 150:
            cur_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world_pts))
            cur_pcd.colors = o3d.utility.Vector3dVector(frame_colors)
            if icp_map is None:
                icp_map = cur_pcd.voxel_down_sample(icp_voxel)
                icp_map.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=20))
            else:
                cur_ds = cur_pcd.voxel_down_sample(max(icp_voxel, 0.03))
                if len(cur_ds.points) >= 80:
                    cur_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=20))
                    reg = o3d.pipelines.registration.registration_icp(
                        cur_ds,
                        icp_map,
                        0.22,
                        np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    )
                    if reg.fitness >= 0.35:
                        cur_pcd.transform(reg.transformation)
                        world_pts = np.asarray(cur_pcd.points)
                icp_map = (icp_map + cur_pcd).voxel_down_sample(icp_voxel)
                icp_map.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=20))

        all_points.append(world_pts)
        all_colors.append(frame_colors)

    pcd = o3d.geometry.PointCloud()
    if not all_points:
        return pcd

    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    if voxel_size_m > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size_m)

    if len(pcd.points) >= 50:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd
