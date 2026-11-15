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
    img_w: int | None = None
    img_h: int | None = None


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
                    img_w=int(_opt_float(row.get("img_w"))) if _opt_float(row.get("img_w")) else None,
                    img_h=int(_opt_float(row.get("img_h"))) if _opt_float(row.get("img_h")) else None,
                )
            )

    return LidupBundle(root=root, manifest=manifest, frames=frames, gcps=gcps)


@dataclass
class _ExtractedFrame:
    frame: FrameRecord
    cam_pts: np.ndarray  # (N, 3) em coordenadas da câmera
    weights: np.ndarray  # (N,) peso físico w = cos(theta) / z^2
    pose_depth: np.ndarray  # (4, 4) pose da câmera de profundidade
    rgb_arr: np.ndarray | None
    depth_filtered_m: np.ndarray  # (d_h, d_w) float32, zeros fora da máscara válida
    v_idx: np.ndarray
    u_idx: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    d_w: int
    d_h: int
    img_w: int
    img_h: int
    poses_differ: bool


def _compute_incidence_cos(pts_cam_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula cosseno do ângulo de incidência entre raio da câmera e normal local."""
    d_pu = np.gradient(pts_cam_grid, axis=1)
    d_pv = np.gradient(pts_cam_grid, axis=0)
    normals = np.cross(d_pu, d_pv)
    n_norm = np.linalg.norm(normals, axis=-1)
    normals_unit = normals / np.where(n_norm > 1e-8, n_norm, 1.0)[..., None]

    ray_norm = np.linalg.norm(pts_cam_grid, axis=-1, keepdims=True)
    view_dir = -pts_cam_grid / np.maximum(ray_norm, 1e-8)
    cos_inc = np.abs(np.sum(normals_unit * view_dir, axis=-1))
    return cos_inc, n_norm


def _upsample_depth_guided_by_rgb(
    depth_m: np.ndarray,
    valid_lo: np.ndarray,
    conf: np.ndarray | None,
    rgb_lo: np.ndarray,
    rgb_hi: np.ndarray,
    scale: int,
    edge_jump_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Super-resolução bilateral guiada pela foto RGB + separação bimodal de bordas (unmixing).
    Sobe a malha de profundidade (ex.: 256x192 -> 512x384) travando descontinuidades
    nas bordas visuais nítidas da câmera 12 MP e descartando flying pixels ambíguos.
    """
    d_h, d_w = depth_m.shape[:2]
    u_h, u_w = rgb_hi.shape[:2]

    if edge_jump_m > 0 and np.any(valid_lo):
        d_hi_nb = maximum_filter(np.where(valid_lo, depth_m, 0.0), size=3)
        d_lo_nb = minimum_filter(np.where(valid_lo, depth_m, 1e6), size=3)
        is_edge_lo = valid_lo & ((d_hi_nb - d_lo_nb) > edge_jump_m)
    else:
        is_edge_lo = np.zeros_like(valid_lo, dtype=bool)

    conf_w_lo = np.where(valid_lo, 1.0, 0.0)
    if conf is not None and conf.shape == depth_m.shape:
        conf_w_lo = np.where(valid_lo, np.where(conf >= 2, 1.5, 1.0), 0.0)

    y_f = (np.arange(u_h, dtype=np.float64) + 0.5) / float(scale) - 0.5
    x_f = (np.arange(u_w, dtype=np.float64) + 0.5) / float(scale) - 0.5
    y0 = np.clip(np.rint(y_f).astype(np.int64), 0, d_h - 1)
    x0 = np.clip(np.rint(x_f).astype(np.int64), 0, d_w - 1)

    edge_hi = is_edge_lo[y0[:, None], x0[None, :]]
    inv_2sr2 = np.where(edge_hi, 1.0 / (2.0 * 0.065 * 0.065), 1.0 / (2.0 * 0.16 * 0.16))
    inv_2ss2 = 1.0 / (2.0 * 0.75 * 0.75)

    num = np.zeros((u_h, u_w), dtype=np.float64)
    num2 = np.zeros((u_h, u_w), dtype=np.float64)
    den = np.zeros((u_h, u_w), dtype=np.float64)
    max_wr = np.zeros((u_h, u_w), dtype=np.float64)

    for dy in (-1, 0, 1):
        yr = y0 + dy
        in_y = (yr >= 0) & (yr < d_h)
        yi = np.clip(yr, 0, d_h - 1)
        dy2 = (y_f - yi) ** 2
        for dx in (-1, 0, 1):
            xr = x0 + dx
            in_b = in_y[:, None] & (xr >= 0)[None, :] & (xr < d_w)[None, :]
            xi = np.clip(xr, 0, d_w - 1)

            v_nb = valid_lo[yi[:, None], xi[None, :]] & in_b
            if not np.any(v_nb):
                continue

            ds2 = dy2[:, None] + ((x_f - xi) ** 2)[None, :]
            ws = np.exp(-ds2 * inv_2ss2)

            dr2 = np.sum((rgb_hi - rgb_lo[yi[:, None], xi[None, :]]) ** 2, axis=-1)
            wr = np.exp(-dr2 * inv_2sr2)
            max_wr = np.maximum(max_wr, np.where(v_nb, wr, 0.0))

            w = np.where(v_nb, ws * wr * conf_w_lo[yi[:, None], xi[None, :]], 0.0)
            d_nb = depth_m[yi[:, None], xi[None, :]]
            num += w * d_nb
            num2 += w * (d_nb * d_nb)
            den += w

    safe_den = np.maximum(den, 1e-12)
    up_depth = np.where(den > 1e-4, num / safe_den, 0.0)
    var_d = np.maximum(0.0, np.where(den > 1e-4, num2 / safe_den - up_depth * up_depth, 0.0))
    std_d = np.sqrt(var_d)

    # Em zonas de salto bimodal, exige casamento fotométrico claro e baixa variância de profundidade
    max_std = max(0.04, edge_jump_m * 0.45) if edge_jump_m > 0 else 0.08
    unmixed_ok = (~edge_hi) | ((max_wr >= 0.20) & (std_d <= max_std))
    up_mask = (den > 1e-4) & unmixed_ok

    if edge_jump_m > 0 and np.any(up_mask):
        hi_clean = maximum_filter(np.where(up_mask, up_depth, 0.0), size=3)
        lo_clean = minimum_filter(np.where(up_mask, up_depth, 1e6), size=3)
        up_mask &= (hi_clean - lo_clean) <= (edge_jump_m * 1.25)

    return up_depth, up_mask, rgb_hi


def _estimate_orbit_focus_point(poses: list[np.ndarray]) -> np.ndarray | None:
    """Estima o ponto 3D de convergência dos eixos ópticos das câmeras (centro do objeto em scan 360°)."""
    if len(poses) < 2:
        return None
    a_mat = np.eye(3, dtype=np.float64) * 1e-3
    b_vec = np.zeros(3, dtype=np.float64)
    for pose in poses:
        c = pose[:3, 3]
        d = pose[:3, 2]
        norm = float(np.linalg.norm(d))
        if norm < 1e-8:
            continue
        d = d / norm
        proj = np.eye(3, dtype=np.float64) - np.outer(d, d)
        a_mat += proj
        b_vec += proj @ c
    try:
        focus = np.linalg.solve(a_mat, b_vec)
    except np.linalg.LinAlgError:
        return None
    return focus if np.all(np.isfinite(focus)) else None


def _extract_frame_points(
    bundle: LidupBundle,
    frame: FrameRecord,
    *,
    min_confidence: int,
    min_depth_m: float,
    max_depth_m: float,
    edge_jump_m: float,
    max_incidence_deg: float | None,
    ref_img_w: int,
    ref_img_h: int,
    depth_upsample: int = 1,
) -> _ExtractedFrame | None:
    if not frame.depth_file:
        return None
    depth_path = bundle.root / frame.depth_file
    if not depth_path.exists():
        return None

    depth_mm = np.asarray(Image.open(depth_path), dtype=np.uint16)
    depth_m = depth_mm.astype(np.float64) / 1000.0
    d_h_lo, d_w_lo = depth_m.shape[:2]

    base_mask = (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    conf: np.ndarray | None = None
    if frame.confidence_file:
        conf_path = bundle.root / frame.confidence_file
        if conf_path.exists():
            conf_raw = np.asarray(Image.open(conf_path), dtype=np.uint8)
            if conf_raw.shape == depth_m.shape:
                conf = conf_raw
                base_mask &= conf >= min_confidence

    if not np.any(base_mask):
        return None

    pose_depth = frame.depth_t_wc if frame.depth_t_wc is not None else frame.t_wc
    poses_differ = frame.depth_t_wc is not None and not np.allclose(frame.depth_t_wc, frame.t_wc, atol=1e-4)
    pose_sync_ok = True
    if poses_differ:
        rel_tf = np.linalg.inv(pose_depth) @ frame.t_wc
        trans_diff = float(np.linalg.norm(rel_tf[:3, 3]))
        rot_diff_deg = float(np.degrees(np.arccos(np.clip((np.trace(rel_tf[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
        pose_sync_ok = trans_diff <= 0.02 and rot_diff_deg <= 0.5

    rgb_arr: np.ndarray | None = None
    img_w = frame.img_w if (frame.img_w and frame.img_w > 0) else ref_img_w
    img_h = frame.img_h if (frame.img_h and frame.img_h > 0) else ref_img_h
    eff_scale = max(1, int(depth_upsample))
    d_h, d_w = d_h_lo, d_w_lo
    mask = base_mask

    if frame.image_file:
        img_path = bundle.root / frame.image_file
        if img_path.exists():
            with Image.open(img_path) as im:
                rgb_im = im.convert("RGB")
                img_w, img_h = rgb_im.size
                rgb_lo_im = rgb_im.resize((d_w_lo, d_h_lo), Image.Resampling.BILINEAR) if (img_w, img_h) != (d_w_lo, d_h_lo) else rgb_im
                rgb_lo_arr = np.asarray(rgb_lo_im, dtype=np.float64) / 255.0

                if eff_scale > 1 and pose_sync_ok:
                    d_w_hi = d_w_lo * eff_scale
                    d_h_hi = d_h_lo * eff_scale
                    rgb_hi_im = rgb_im.resize((d_w_hi, d_h_hi), Image.Resampling.BILINEAR) if (img_w, img_h) != (d_w_hi, d_h_hi) else rgb_im
                    rgb_hi_arr = np.asarray(rgb_hi_im, dtype=np.float64) / 255.0
                    depth_m, mask, rgb_arr = _upsample_depth_guided_by_rgb(
                        depth_m,
                        base_mask,
                        conf,
                        rgb_lo_arr,
                        rgb_hi_arr,
                        scale=eff_scale,
                        edge_jump_m=edge_jump_m,
                    )
                    d_h, d_w = d_h_hi, d_w_hi
                else:
                    rgb_arr = rgb_lo_arr

    if d_h == d_h_lo and d_w == d_w_lo and edge_jump_m > 0 and np.any(mask):
        d_hi = maximum_filter(np.where(mask, depth_m, 0.0), size=3)
        d_lo = minimum_filter(np.where(mask, depth_m, 1e6), size=3)
        mask = mask & ((d_hi - d_lo) <= edge_jump_m)

    if not np.any(mask):
        return None

    if img_w <= 0 or img_h <= 0:
        img_w = max(d_w, int(round(frame.cx * 2.0)))
        img_h = max(d_h, int(round(frame.cy * 2.0)))

    sx = d_w / float(img_w)
    sy = d_h / float(img_h)
    fx = frame.fx * sx
    fy = frame.fy * sy
    cx = frame.cx * sx
    cy = frame.cy * sy

    uu, vv = np.meshgrid(np.arange(d_w, dtype=np.float64), np.arange(d_h, dtype=np.float64))
    pts_grid = np.stack([(uu - cx) * depth_m / fx, (vv - cy) * depth_m / fy, depth_m], axis=-1)
    cos_inc, n_norm = _compute_incidence_cos(pts_grid)

    if max_incidence_deg is not None and 0.0 < max_incidence_deg < 89.9:
        cos_thresh = float(np.cos(np.radians(max_incidence_deg)))
        phys_mask = mask & (cos_inc >= cos_thresh) & (n_norm > 1e-6)
        if np.count_nonzero(phys_mask) >= max(20, int(0.15 * np.count_nonzero(mask))):
            mask = phys_mask

    if not np.any(mask):
        return None

    v_idx, u_idx = np.nonzero(mask)
    cam_pts = pts_grid[v_idx, u_idx]
    z = cam_pts[:, 2]
    cos_vals = np.clip(cos_inc[v_idx, u_idx], 0.1, 1.0)
    weights = cos_vals / np.maximum(z, 0.30) ** 2

    depth_filtered_m = np.where(mask, depth_m, 0.0).astype(np.float32)

    return _ExtractedFrame(
        frame=frame,
        cam_pts=cam_pts,
        weights=weights,
        pose_depth=pose_depth,
        rgb_arr=rgb_arr,
        depth_filtered_m=depth_filtered_m,
        v_idx=v_idx,
        u_idx=u_idx,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        d_w=d_w,
        d_h=d_h,
        img_w=img_w,
        img_h=img_h,
        poses_differ=poses_differ,
    )


def _sample_frame_colors(ext: _ExtractedFrame, raw_world_pts: np.ndarray) -> np.ndarray:
    if ext.rgb_arr is None:
        return np.full((len(raw_world_pts), 3), 0.65, dtype=np.float64)
    if not ext.poses_differ:
        return ext.rgb_arr[ext.v_idx, ext.u_idx]

    r_rgb = ext.frame.t_wc[:3, :3]
    t_rgb = ext.frame.t_wc[:3, 3]
    cam_rgb = (raw_world_pts - t_rgb) @ r_rgb
    z_rgb = cam_rgb[:, 2]
    valid_rgb = z_rgb > 0.1
    u_rgb = np.rint(ext.fx * (cam_rgb[:, 0] / np.where(valid_rgb, z_rgb, 1.0)) + ext.cx).astype(np.int64)
    v_rgb = np.rint(ext.fy * (cam_rgb[:, 1] / np.where(valid_rgb, z_rgb, 1.0)) + ext.cy).astype(np.int64)
    in_bounds = valid_rgb & (u_rgb >= 0) & (u_rgb < ext.d_w) & (v_rgb >= 0) & (v_rgb < ext.d_h)
    colors = np.full((len(raw_world_pts), 3), 0.65, dtype=np.float64)
    colors[in_bounds] = ext.rgb_arr[v_rgb[in_bounds], u_rgb[in_bounds]]
    return colors


def _weighted_voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    weights: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Funde pontos por voxel ponderando por w = cos(theta) / z^2."""
    if len(points) == 0 or voxel_size_m <= 0:
        return points, colors
    keys = np.floor(points / float(voxel_size_m)).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n_vox = int(inv.max()) + 1
    w = np.maximum(weights, 1e-6)
    w_sum = np.bincount(inv, weights=w, minlength=n_vox)
    safe_w = np.maximum(w_sum, 1e-12)[:, None]

    out_pts = np.empty((n_vox, 3), dtype=np.float64)
    out_cols = np.empty((n_vox, 3), dtype=np.float64)
    for k in range(3):
        out_pts[:, k] = np.bincount(inv, weights=points[:, k] * w, minlength=n_vox)
        out_cols[:, k] = np.bincount(inv, weights=colors[:, k] * w, minlength=n_vox)
    out_pts /= safe_w
    out_cols = np.clip(out_cols / safe_w, 0.0, 1.0)
    return out_pts, out_cols


def _register_pair(
    src_pcd: o3d.geometry.PointCloud,
    dst_pcd: o3d.geometry.PointCloud,
    init_tf: np.ndarray,
    *,
    max_corr_m: float,
    use_color: bool,
) -> o3d.pipelines.registration.RegistrationResult:
    if use_color and src_pcd.has_colors() and dst_pcd.has_colors():
        try:
            return o3d.pipelines.registration.registration_colored_icp(
                src_pcd,
                dst_pcd,
                max_corr_m,
                init_tf,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=35),
            )
        except RuntimeError:
            pass
    return o3d.pipelines.registration.registration_icp(
        src_pcd,
        dst_pcd,
        max_corr_m,
        init_tf,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )


def _refine_poses_posegraph(
    extracted: list[_ExtractedFrame],
    *,
    use_color: bool,
    voxel_size_m: float,
) -> list[np.ndarray]:
    """Refina poses globalmente via PoseGraph (odometria + loop closure com limite físico)."""
    n = len(extracted)
    poses_init = [ext.pose_depth.copy() for ext in extracted]
    if n < 2:
        return poses_init

    ds_voxel = max(0.025, float(voxel_size_m) if voxel_size_m > 0 else 0.03)
    frame_pcds: list[o3d.geometry.PointCloud] = []
    cam_centers = np.array([p[:3, 3] for p in poses_init], dtype=np.float64)

    for ext in extracted:
        raw_world = ext.cam_pts @ ext.pose_depth[:3, :3].T + ext.pose_depth[:3, 3]
        cols = _sample_frame_colors(ext, raw_world)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ext.cam_pts))
        pcd.colors = o3d.utility.Vector3dVector(cols)
        ds = pcd.voxel_down_sample(ds_voxel)
        if len(ds.points) >= 20:
            ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30))
        frame_pcds.append(ds)

    pg = o3d.pipelines.registration.PoseGraph()
    for p in poses_init:
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(p))

    max_corr = 0.08
    edge_count = 0
    seq_connected: set[int] = set()
    for i in range(n):
        if len(frame_pcds[i].points) < 40:
            continue
        for j in range(i + 1, n):
            if len(frame_pcds[j].points) < 40:
                continue
            is_odom = (j - i) <= 2
            dist_cams = float(np.linalg.norm(cam_centers[j] - cam_centers[i]))
            if not is_odom and dist_cams > 3.5:
                continue

            t_init = np.linalg.inv(poses_init[j]) @ poses_init[i]
            if not is_odom:
                ev = o3d.pipelines.registration.evaluate_registration(frame_pcds[i], frame_pcds[j], 0.12, t_init)
                if ev.fitness < 0.25:
                    continue

            corr = max_corr if is_odom else 0.10
            reg = _register_pair(frame_pcds[i], frame_pcds[j], t_init, max_corr_m=corr, use_color=use_color)
            delta = reg.transformation @ np.linalg.inv(t_init)
            shift = float(np.linalg.norm(delta[:3, 3]))
            ang_deg = float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
            if reg.fitness >= 0.35 and reg.inlier_rmse <= 0.045 and shift <= 0.35 and ang_deg <= 12.0:
                info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    frame_pcds[i], frame_pcds[j], corr, reg.transformation
                )
                pg.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        i, j, reg.transformation, info, uncertain=not is_odom
                    )
                )
                edge_count += 1
                if j == i + 1:
                    seq_connected.add(i)

    if edge_count == 0:
        return poses_init

    # Garante grafo conexo ligando pares sequenciais sem ICP via odometria bruta do ARKit
    weak_info = np.eye(6, dtype=np.float64) * 5.0
    for i in range(n - 1):
        if i not in seq_connected:
            t_prior = np.linalg.inv(poses_init[i + 1]) @ poses_init[i]
            pg.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i, i + 1, t_prior, weak_info, uncertain=False
                )
            )

    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_corr,
            edge_prune_threshold=0.25,
            reference_node=0,
        ),
    )
    return [np.array(pg.nodes[i].pose, dtype=np.float64) for i in range(n)]


def _prepare_extracted_and_poses(
    bundle: LidupBundle,
    *,
    min_confidence: int,
    min_depth_m: float,
    max_depth_m: float,
    voxel_size_m: float,
    frame_stride: int,
    edge_jump_m: float,
    max_incidence_deg: float | None,
    mode: str,
    depth_upsample: int = 1,
) -> tuple[list[_ExtractedFrame], list[np.ndarray]]:
    cfg = bundle.manifest.get("capture_config", {})
    ref_img_w = int(cfg.get("image_width") or 0)
    ref_img_h = int(cfg.get("image_height") or 0)
    if ref_img_w <= 0 or ref_img_h <= 0:
        for fr in bundle.frames:
            if fr.img_w and fr.img_h:
                ref_img_w, ref_img_h = fr.img_w, fr.img_h
                break
            if fr.image_file and (bundle.root / fr.image_file).exists():
                with Image.open(bundle.root / fr.image_file) as im:
                    ref_img_w, ref_img_h = im.size
                break

    extracted: list[_ExtractedFrame] = []
    for idx, frame in enumerate(bundle.frames):
        if idx % max(1, frame_stride) != 0:
            continue
        ext = _extract_frame_points(
            bundle,
            frame,
            min_confidence=min_confidence,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            edge_jump_m=edge_jump_m,
            max_incidence_deg=max_incidence_deg,
            ref_img_w=ref_img_w,
            ref_img_h=ref_img_h,
            depth_upsample=depth_upsample,
        )
        if ext is not None:
            extracted.append(ext)

    if not extracted:
        return [], []

    if mode in {"posegraph", "colored_icp"}:
        refined_poses = _refine_poses_posegraph(
            extracted,
            use_color=(mode == "colored_icp"),
            voxel_size_m=voxel_size_m,
        )
    else:
        refined_poses = [ext.pose_depth for ext in extracted]

    return extracted, refined_poses


def _colorize_from_hires_rgb(
    bundle: LidupBundle,
    points_arkit: np.ndarray,
    normals_arkit: np.ndarray | None,
    extracted: list[_ExtractedFrame],
    refined_poses: list[np.ndarray],
    fallback_colors: np.ndarray,
    *,
    max_depth_m: float,
) -> np.ndarray:
    """
    Projeta vértices/pontos finais diretamente nas fotos 12 MP (uma por vez na RAM),
    escolhendo o frame mais próximo/frontal com teste de oclusão contra o mapa LiDAR.
    """
    if len(points_arkit) == 0:
        return fallback_colors

    best_score = np.full(len(points_arkit), -1.0, dtype=np.float64)
    out_colors = fallback_colors.copy()

    for ext, pose in zip(extracted, refined_poses):
        if not ext.frame.image_file:
            continue
        img_path = bundle.root / ext.frame.image_file
        if not img_path.exists():
            continue

        # Mantém offset relativo RGB<->Depth quando PoseGraph ajustou a pose do frame
        rel_rgb = np.linalg.inv(ext.pose_depth) @ ext.frame.t_wc
        pose_rgb = pose @ rel_rgb

        cam = (points_arkit - pose_rgb[:3, 3]) @ pose_rgb[:3, :3]
        z = cam[:, 2]
        in_range = (z > 0.20) & (z <= max_depth_m)
        if not np.any(in_range):
            continue

        with Image.open(img_path) as im:
            hi_w, hi_h = im.size
            sx = hi_w / float(max(ext.img_w, 1))
            sy = hi_h / float(max(ext.img_h, 1))
            fx_hi = ext.frame.fx * sx
            fy_hi = ext.frame.fy * sy
            cx_hi = ext.frame.cx * sx
            cy_hi = ext.frame.cy * sy

            safe_z = np.where(in_range, z, 1.0)
            u_hi = np.rint(fx_hi * (cam[:, 0] / safe_z) + cx_hi).astype(np.int64)
            v_hi = np.rint(fy_hi * (cam[:, 1] / safe_z) + cy_hi).astype(np.int64)
            inside = in_range & (u_hi >= 0) & (u_hi < hi_w) & (v_hi >= 0) & (v_hi < hi_h)
            if not np.any(inside):
                continue

            idx_cand = np.nonzero(inside)[0]
            u_d = np.clip(np.rint(u_hi[idx_cand] * (ext.d_w / float(hi_w))).astype(np.int64), 0, ext.d_w - 1)
            v_d = np.clip(np.rint(v_hi[idx_cand] * (ext.d_h / float(hi_h))).astype(np.int64), 0, ext.d_h - 1)
            d_obs = ext.depth_filtered_m[v_d, u_d]
            not_occluded = (d_obs > 0.0) & (np.abs(z[idx_cand] - d_obs) <= 0.05)
            idx_vis = idx_cand[not_occluded]
            if len(idx_vis) == 0:
                continue

            view_dir = -cam[idx_vis] / np.maximum(np.linalg.norm(cam[idx_vis], axis=1, keepdims=True), 1e-8)
            if normals_arkit is not None and len(normals_arkit) == len(points_arkit):
                norms_cam = normals_arkit[idx_vis] @ pose_rgb[:3, :3]
                cos_v = np.sum(norms_cam * view_dir, axis=1)
                facing = cos_v > 0.20
                idx_vis = idx_vis[facing]
                cos_v = cos_v[facing]
            else:
                cos_v = np.clip(z[idx_vis] / np.maximum(np.linalg.norm(cam[idx_vis], axis=1), 1e-8), 0.1, 1.0)

            if len(idx_vis) == 0:
                continue

            scores = cos_v / np.maximum(z[idx_vis], 0.30) ** 2
            better = scores > best_score[idx_vis]
            idx_win = idx_vis[better]
            if len(idx_win) == 0:
                continue

            hi_arr = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
            out_colors[idx_win] = hi_arr[v_hi[idx_win], u_hi[idx_win]]
            best_score[idx_win] = scores[better]

    return np.clip(out_colors, 0.0, 1.0)


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
    pose_refine: str | None = None,
    max_incidence_deg: float | None = 75.0,
    physics_weight: bool = True,
    hires_color: bool = False,
    depth_upsample: int = 1,
) -> o3d.geometry.PointCloud:
    """
    Reconstrói a nuvem de pontos 3D no referencial local do ARKit a partir dos mapas
    de profundidade LiDAR filtrados por confiança, salto de borda e ângulo de incidência,
    com ponderação física w = cos(theta) / z^2 e refinamento de poses opcional.
    """
    mode = pose_refine if pose_refine is not None else ("posegraph" if refine_poses else "none")
    if mode not in {"none", "posegraph", "colored_icp", "legacy_icp"}:
        raise ValueError(f"Modo pose_refine inválido: {mode}")

    extracted, refined_poses = _prepare_extracted_and_poses(
        bundle,
        min_confidence=min_confidence,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        voxel_size_m=voxel_size_m,
        frame_stride=frame_stride,
        edge_jump_m=edge_jump_m,
        max_incidence_deg=max_incidence_deg,
        mode=mode,
        depth_upsample=depth_upsample,
    )
    if not extracted:
        return o3d.geometry.PointCloud()

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []

    icp_map: o3d.geometry.PointCloud | None = None
    icp_voxel = max(0.02, float(voxel_size_m) if voxel_size_m > 0 else 0.02)

    for ext, pose in zip(extracted, refined_poses):
        world_pts = ext.cam_pts @ pose[:3, :3].T + pose[:3, 3]
        raw_world_pts = (
            world_pts
            if mode in {"none", "legacy_icp"}
            else ext.cam_pts @ ext.pose_depth[:3, :3].T + ext.pose_depth[:3, 3]
        )
        frame_colors = _sample_frame_colors(ext, raw_world_pts)

        if mode == "legacy_icp" and len(world_pts) >= 150:
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
        all_weights.append(ext.weights if physics_weight else np.ones(len(world_pts), dtype=np.float64))

    merged_pts = np.vstack(all_points)
    merged_cols = np.vstack(all_colors)
    merged_w = np.concatenate(all_weights)

    if voxel_size_m > 0:
        merged_pts, merged_cols = _weighted_voxel_downsample(merged_pts, merged_cols, merged_w, voxel_size_m)

    if hires_color and len(merged_pts) > 0:
        merged_cols = _colorize_from_hires_rgb(
            bundle,
            merged_pts,
            None,
            extracted,
            refined_poses,
            merged_cols,
            max_depth_m=max_depth_m,
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_pts)
    pcd.colors = o3d.utility.Vector3dVector(merged_cols)

    if len(pcd.points) >= 50:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd


def _integrate_uniform_tsdf(
    extracted: list[_ExtractedFrame],
    refined_poses: list[np.ndarray],
    lo: np.ndarray,
    length: float,
    res: int,
    vox: float,
    max_depth_m: float,
) -> o3d.geometry.TriangleMesh:
    actual_vox = length / float(res)
    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=length,
        resolution=res,
        sdf_trunc=max(0.03, actual_vox * 4.5),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for ext, pose in zip(extracted, refined_poses):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(ext.d_w, ext.d_h, ext.fx, ext.fy, ext.cx, ext.cy)
        if ext.rgb_arr is not None:
            color_u8 = np.clip(ext.rgb_arr * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            color_u8 = np.full((ext.d_h, ext.d_w, 3), 165, dtype=np.uint8)

        depth_mm = np.clip(np.rint(ext.depth_filtered_m * 1000.0), 0.0, 65535.0).astype(np.uint16)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color_u8)),
            o3d.geometry.Image(np.ascontiguousarray(depth_mm)),
            depth_scale=1000.0,
            depth_trunc=float(max_depth_m),
            convert_rgb_to_intensity=False,
        )
        pose_shifted = pose.copy()
        pose_shifted[:3, 3] -= lo
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose_shifted))

    mesh = volume.extract_triangle_mesh()
    mesh.translate(lo, relative=True)
    return mesh


def reconstruct_tsdf_mesh(
    bundle: LidupBundle,
    *,
    min_confidence: int = 2,
    min_depth_m: float = 0.30,
    max_depth_m: float = 4.50,
    tsdf_voxel_m: float = 0.008,
    frame_stride: int = 1,
    edge_jump_m: float = 0.12,
    refine_poses: bool = True,
    pose_refine: str | None = None,
    max_incidence_deg: float | None = 75.0,
    hires_color: bool = True,
    depth_upsample: int = 2,
    focus_radius_m: float | None = None,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """
    Funde múltiplos quadros em uma malha 3D contínua via UniformTSDFVolume com super-resolução
    de profundidade guiada pelo RGB (depth_upsample) e foco automático opcional em torno
    do centro de convergência dos raios de visão 360°, pintando os vértices pelas fotos 12 MP.
    """
    mode = pose_refine if pose_refine is not None else ("posegraph" if refine_poses else "none")
    if mode not in {"none", "posegraph", "colored_icp"}:
        mode = "posegraph" if refine_poses else "none"

    extracted, refined_poses = _prepare_extracted_and_poses(
        bundle,
        min_confidence=min_confidence,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        voxel_size_m=max(tsdf_voxel_m * 3.0, 0.02),
        frame_stride=frame_stride,
        edge_jump_m=edge_jump_m,
        max_incidence_deg=max_incidence_deg,
        mode=mode,
        depth_upsample=depth_upsample,
    )
    if not extracted:
        return o3d.geometry.PointCloud(), o3d.geometry.TriangleMesh()

    vox = max(0.004, float(tsdf_voxel_m))
    max_res = 400
    mesh = o3d.geometry.TriangleMesh()

    if focus_radius_m is not None and focus_radius_m > 0.25:
        focus = _estimate_orbit_focus_point(refined_poses)
        if focus is not None:
            r = float(focus_radius_m)
            lo_focus = focus - r
            length_focus = max(0.8, 2.0 * r)
            res_focus = min(max_res, max(64, int(np.ceil(length_focus / vox))))
            mesh = _integrate_uniform_tsdf(extracted, refined_poses, lo_focus, length_focus, res_focus, vox, max_depth_m)

    if len(mesh.vertices) == 0:
        world_samples = np.vstack(
            [
                ext.cam_pts[:: max(1, len(ext.cam_pts) // 250)] @ pose[:3, :3].T + pose[:3, 3]
                for ext, pose in zip(extracted, refined_poses)
            ]
        )
        lo = np.percentile(world_samples, 0.5, axis=0) - 0.30
        hi = np.percentile(world_samples, 99.5, axis=0) + 0.30
        length = max(1.0, float(np.max(hi - lo)))
        res = min(max_res, max(64, int(np.ceil(length / vox))))
        mesh = _integrate_uniform_tsdf(extracted, refined_poses, lo, length, res, vox, max_depth_m)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    norms = np.asarray(mesh.vertex_normals, dtype=np.float64)
    cols = np.asarray(mesh.vertex_colors, dtype=np.float64)
    if len(cols) != len(verts):
        cols = np.full((len(verts), 3), 0.65, dtype=np.float64)

    if hires_color and len(verts) > 0:
        cols = _colorize_from_hires_rgb(
            bundle,
            verts,
            norms,
            extracted,
            refined_poses,
            cols,
            max_depth_m=max_depth_m,
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(cols)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd, mesh
