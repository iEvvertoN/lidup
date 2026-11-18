from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, minimum_filter, uniform_filter, zoom
from scipy.spatial.transform import Rotation

from ._rust_bridge import (
    rust_colorize_hires_frame,
    rust_extract_depth_points,
    rust_mvs_sweep,
    rust_resize_rgb_u8_to_f64,
    rust_upsample_depth_guided,
    rust_warp_rgb_to_depth,
)


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
    gyro_deg_s: float | None = None
    rgb_hi_u8: np.ndarray | None = None


def _load_imu_gyro_map(bundle: LidupBundle) -> dict[str, float]:
    """Lê `imu.csv` (100 Hz) e associa a velocidade angular (deg/s) mais próxima a cada frame."""
    imu_path = bundle.root / "imu.csv"
    if not imu_path.exists():
        return {}
    ts_list: list[float] = []
    gyro_list: list[float] = []
    with imu_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row["timestamp_s"])
                gx = float(row["gx"])
                gy = float(row["gy"])
                gz = float(row["gz"])
            except (KeyError, TypeError, ValueError):
                continue
            ts_list.append(ts)
            gyro_list.append(float(np.degrees(np.sqrt(gx * gx + gy * gy + gz * gz))))
    if not ts_list:
        return {}
    ts_arr = np.asarray(ts_list, dtype=np.float64)
    g_arr = np.asarray(gyro_list, dtype=np.float64)
    order = np.argsort(ts_arr)
    ts_arr = ts_arr[order]
    g_arr = g_arr[order]

    out: dict[str, float] = {}
    for fr in bundle.frames:
        lo_i = int(np.searchsorted(ts_arr, fr.timestamp_s - 0.025, side="left"))
        hi_i = int(np.searchsorted(ts_arr, fr.timestamp_s + 0.025, side="right"))
        if hi_i > lo_i:
            out[fr.frame_id] = float(np.median(g_arr[lo_i:hi_i]))
        else:
            idx = min(max(int(np.searchsorted(ts_arr, fr.timestamp_s)), 0), len(g_arr) - 1)
            if abs(ts_arr[idx] - fr.timestamp_s) <= 0.25:
                out[fr.frame_id] = float(g_arr[idx])
    return out


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

    rust_res = rust_upsample_depth_guided(depth_m, valid_lo, conf, rgb_lo, rgb_hi, int(scale), float(edge_jump_m))
    if rust_res is not None:
        return rust_res[0], rust_res[1], rgb_hi

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


def _filter_floor_mirror_outliers(
    extracted: list[_ExtractedFrame],
    poses: list[np.ndarray],
) -> tuple[list[_ExtractedFrame], list[np.ndarray], float | None]:
    """
    Detecta piso plano dominante (ex.: cerâmica/mesa) e remove quadros ou raios espelhados
    que aparecem abaixo do piso por reflexão especular do feixe 940 nm.
    """
    if len(extracted) < 5:
        return extracted, poses, None

    all_y = np.concatenate(
        [(ext.cam_pts[::4] @ p[:3, :3].T + p[:3, 3])[:, 1] for ext, p in zip(extracted, poses) if len(ext.cam_pts) > 0]
    )
    if len(all_y) < 80:
        return extracted, poses, None

    y_lo, y_hi = float(np.percentile(all_y, 1.0)), float(np.percentile(all_y, 99.0))
    span = y_hi - y_lo
    if span < 0.04:
        return extracted, poses, None

    n_bins = max(20, min(120, int(np.ceil(span / 0.015))))
    hist, edges = np.histogram(all_y, bins=n_bins, range=(y_lo, y_hi))
    pk = int(np.argmax(hist))
    if hist[pk] < max(100, int(0.20 * len(all_y))):
        return extracted, poses, None

    floor_y = float(0.5 * (edges[pk] + edges[pk + 1]))
    clean_ext: list[_ExtractedFrame] = []
    clean_poses: list[np.ndarray] = []

    for ext, pose in zip(extracted, poses):
        w = ext.cam_pts @ pose[:3, :3].T + pose[:3, 3]
        if len(w) == 0:
            continue
        if float(np.mean(w[:, 1] >= floor_y - 0.04)) < 0.50:
            continue
        keep = w[:, 1] >= floor_y - 0.03
        if np.any(~keep) and np.count_nonzero(keep) >= 50:
            ext.depth_filtered_m[ext.v_idx[~keep], ext.u_idx[~keep]] = 0.0
            ext.cam_pts = ext.cam_pts[keep]
            ext.weights = ext.weights[keep]
            ext.v_idx = ext.v_idx[keep]
            ext.u_idx = ext.u_idx[keep]
        clean_ext.append(ext)
        clean_poses.append(pose)

    if not clean_ext:
        return extracted, poses, None
    return clean_ext, clean_poses, floor_y


def _refine_object_focus_point(
    extracted: list[_ExtractedFrame],
    poses: list[np.ndarray],
    ray_focus: np.ndarray,
    floor_y: float | None,
) -> np.ndarray:
    """Centraliza a caixa de foco exatamente no objeto acima do piso próximo ao cruzamento dos raios."""
    if floor_y is None:
        return ray_focus
    pts = np.vstack([ext.cam_pts[::2] @ p[:3, :3].T + p[:3, 3] for ext, p in zip(extracted, poses)])
    above = pts[(np.linalg.norm(pts - ray_focus, axis=1) <= 0.45) & (pts[:, 1] > floor_y + 0.015)]
    if len(above) < 100:
        return ray_focus
    return 0.5 * (above.min(axis=0) + above.max(axis=0))


def _warp_rgb_to_depth_grid(
    rgb_hi_arr: np.ndarray,
    depth_m_lo: np.ndarray,
    valid_lo: np.ndarray,
    pose_depth: np.ndarray,
    pose_rgb: np.ndarray,
    fx_d_hi: float,
    fy_d_hi: float,
    cx_d_hi: float,
    cy_d_hi: float,
    scale: int,
) -> np.ndarray:
    """Reprojeta a imagem RGB para a pose exata da câmera LiDAR quando houve rotação entre capturas."""
    d_h_hi, d_w_hi = rgb_hi_arr.shape[:2]
    med_d = float(np.median(depth_m_lo[valid_lo])) if np.any(valid_lo) else 1.0
    d_filled = np.where(valid_lo, depth_m_lo, med_d)
    d_hi = zoom(d_filled, float(scale), order=1)
    if d_hi.shape != (d_h_hi, d_w_hi):
        d_hi = d_hi[:d_h_hi, :d_w_hi]

    rel_rgb_from_d = np.linalg.inv(pose_rgb) @ pose_depth
    warped_rust = rust_warp_rgb_to_depth(
        rgb_hi_arr,
        d_hi,
        rel_rgb_from_d,
        fx_d_hi,
        fy_d_hi,
        cx_d_hi,
        cy_d_hi,
    )
    if warped_rust is not None:
        return warped_rust

    uu, vv = np.meshgrid(np.arange(d_w_hi, dtype=np.float64), np.arange(d_h_hi, dtype=np.float64))
    cam_d = np.stack([(uu - cx_d_hi) * d_hi / fx_d_hi, (vv - cy_d_hi) * d_hi / fy_d_hi, d_hi], axis=-1)

    pts_flat = cam_d.reshape(-1, 3) @ rel_rgb_from_d[:3, :3].T + rel_rgb_from_d[:3, 3]
    z_r = np.maximum(pts_flat[:, 2], 0.05)
    u_r = (fx_d_hi * (pts_flat[:, 0] / z_r) + cx_d_hi).reshape(d_h_hi, d_w_hi)
    v_r = (fy_d_hi * (pts_flat[:, 1] / z_r) + cy_d_hi).reshape(d_h_hi, d_w_hi)

    warped = np.empty_like(rgb_hi_arr)
    for ch in range(3):
        warped[:, :, ch] = map_coordinates(rgb_hi_arr[:, :, ch], [v_r, u_r], order=1, mode="nearest")
    return np.clip(warped, 0.0, 1.0)


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
    cache_hires_rgb: bool = False,
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
    needs_rgb_warp = False
    if poses_differ:
        rel_tf = np.linalg.inv(pose_depth) @ frame.t_wc
        trans_diff = float(np.linalg.norm(rel_tf[:3, 3]))
        rot_diff_deg = float(np.degrees(np.arccos(np.clip((np.trace(rel_tf[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
        if trans_diff <= 0.008 and rot_diff_deg <= 0.15:
            pose_sync_ok = True
        elif trans_diff <= 0.15 and rot_diff_deg <= 8.0:
            needs_rgb_warp = True
            pose_sync_ok = True
        else:
            pose_sync_ok = False

    rgb_arr: np.ndarray | None = None
    img_w = frame.img_w if (frame.img_w and frame.img_w > 0) else ref_img_w
    img_h = frame.img_h if (frame.img_h and frame.img_h > 0) else ref_img_h
    eff_scale = max(1, int(depth_upsample))
    d_h, d_w = d_h_lo, d_w_lo
    mask = base_mask

    rgb_hi_u8_cached: np.ndarray | None = None
    if frame.image_file:
        img_path = bundle.root / frame.image_file
        if img_path.exists():
            with Image.open(img_path) as im:
                rgb_im = im.convert("RGB")
                img_w, img_h = rgb_im.size
                hi_u8 = np.asarray(rgb_im, dtype=np.uint8)
                if cache_hires_rgb:
                    rgb_hi_u8_cached = hi_u8
                if eff_scale > 1 and pose_sync_ok:
                    d_w_hi = d_w_lo * eff_scale
                    d_h_hi = d_h_lo * eff_scale
                    rgb_hi_arr = (
                        rust_resize_rgb_u8_to_f64(hi_u8, d_w_hi, d_h_hi)
                        if (img_w, img_h) != (d_w_hi, d_h_hi)
                        else hi_u8.astype(np.float64) / 255.0
                    )
                    if rgb_hi_arr is None:
                        rgb_hi_im = rgb_im.resize((d_w_hi, d_h_hi), Image.Resampling.BILINEAR)
                        rgb_hi_arr = np.asarray(rgb_hi_im, dtype=np.float64) / 255.0
                    if needs_rgb_warp:
                        sx_w = d_w_hi / float(max(img_w, 1))
                        sy_w = d_h_hi / float(max(img_h, 1))
                        rgb_hi_arr = _warp_rgb_to_depth_grid(
                            rgb_hi_arr,
                            depth_m,
                            base_mask,
                            pose_depth,
                            frame.t_wc,
                            frame.fx * sx_w,
                            frame.fy * sy_w,
                            frame.cx * sx_w,
                            frame.cy * sy_w,
                            eff_scale,
                        )
                        rgb_lo_arr = rust_resize_rgb_u8_to_f64(
                            np.clip(rgb_hi_arr * 255.0, 0.0, 255.0).astype(np.uint8),
                            d_w_lo,
                            d_h_lo,
                        )
                        if rgb_lo_arr is None:
                            rgb_lo_im = Image.fromarray(np.clip(rgb_hi_arr * 255.0, 0.0, 255.0).astype(np.uint8)).resize(
                                (d_w_lo, d_h_lo), Image.Resampling.BILINEAR
                            )
                            rgb_lo_arr = np.asarray(rgb_lo_im, dtype=np.float64) / 255.0
                        poses_differ = False
                    else:
                        rgb_lo_arr = (
                            rust_resize_rgb_u8_to_f64(hi_u8, d_w_lo, d_h_lo)
                            if (img_w, img_h) != (d_w_lo, d_h_lo)
                            else hi_u8.astype(np.float64) / 255.0
                        )
                        if rgb_lo_arr is None:
                            rgb_lo_im = rgb_im.resize((d_w_lo, d_h_lo), Image.Resampling.BILINEAR)
                            rgb_lo_arr = np.asarray(rgb_lo_im, dtype=np.float64) / 255.0

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
                    rgb_arr = (
                        rust_resize_rgb_u8_to_f64(hi_u8, d_w_lo, d_h_lo)
                        if (img_w, img_h) != (d_w_lo, d_h_lo)
                        else hi_u8.astype(np.float64) / 255.0
                    )
                    if rgb_arr is None:
                        rgb_lo_im = rgb_im.resize((d_w_lo, d_h_lo), Image.Resampling.BILINEAR)
                        rgb_arr = np.asarray(rgb_lo_im, dtype=np.float64) / 255.0

    if img_w <= 0 or img_h <= 0:
        img_w = max(d_w, int(round(frame.cx * 2.0)))
        img_h = max(d_h, int(round(frame.cy * 2.0)))

    sx = d_w / float(img_w)
    sy = d_h / float(img_h)
    fx = frame.fx * sx
    fy = frame.fy * sy
    cx = frame.cx * sx
    cy = frame.cy * sy

    edge_pass = edge_jump_m if (d_h == d_h_lo and d_w == d_w_lo and edge_jump_m > 0) else 0.0
    cos_thresh = float(np.cos(np.radians(max_incidence_deg))) if (max_incidence_deg is not None and 0.0 < max_incidence_deg < 89.9) else -1.0

    rust_ext = rust_extract_depth_points(depth_m, mask, fx, fy, cx, cy, edge_pass, cos_thresh)
    if rust_ext is not None:
        mask, cam_pts, weights, v_idx, u_idx = rust_ext
        if len(cam_pts) == 0:
            return None
    else:
        if edge_pass > 0 and np.any(mask):
            d_hi = maximum_filter(np.where(mask, depth_m, 0.0), size=3)
            d_lo = minimum_filter(np.where(mask, depth_m, 1e6), size=3)
            mask = mask & ((d_hi - d_lo) <= edge_pass)

        if not np.any(mask):
            return None

        uu, vv = np.meshgrid(np.arange(d_w, dtype=np.float64), np.arange(d_h, dtype=np.float64))
        pts_grid = np.stack([(uu - cx) * depth_m / fx, (vv - cy) * depth_m / fy, depth_m], axis=-1)
        cos_inc, n_norm = _compute_incidence_cos(pts_grid)

        if cos_thresh > 0.0:
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
        rgb_hi_u8=rgb_hi_u8_cached,
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

    fine_mode = voxel_size_m > 0 and voxel_size_m <= 0.015
    ds_voxel = max(0.008, float(voxel_size_m)) if fine_mode else max(0.025, float(voxel_size_m) if voxel_size_m > 0 else 0.03)
    norm_radius = 0.04 if fine_mode else 0.12
    frame_pcds: list[o3d.geometry.PointCloud] = []
    cam_centers = np.array([p[:3, 3] for p in poses_init], dtype=np.float64)

    for ext in extracted:
        raw_world = ext.cam_pts @ ext.pose_depth[:3, :3].T + ext.pose_depth[:3, 3]
        cols = _sample_frame_colors(ext, raw_world)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ext.cam_pts))
        pcd.colors = o3d.utility.Vector3dVector(cols)
        ds = pcd.voxel_down_sample(ds_voxel)
        if len(ds.points) >= 20:
            ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=norm_radius, max_nn=30))
        frame_pcds.append(ds)

    pg = o3d.pipelines.registration.PoseGraph()
    for p in poses_init:
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(p))

    max_corr = 0.025 if fine_mode else 0.08
    max_rmse = 0.015 if fine_mode else 0.045
    max_shift = 0.08 if fine_mode else 0.35
    max_ang = 3.5 if fine_mode else 12.0
    max_dist_loop = 1.6 if fine_mode else 3.5
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
            if not is_odom and dist_cams > max_dist_loop:
                continue

            t_init = np.linalg.inv(poses_init[j]) @ poses_init[i]
            corr = max_corr if is_odom else (max_corr * 1.25)
            ev = o3d.pipelines.registration.evaluate_registration(frame_pcds[i], frame_pcds[j], corr * 1.4, t_init)
            if not is_odom and ev.fitness < (0.35 if fine_mode else 0.25):
                continue

            reg = _register_pair(frame_pcds[i], frame_pcds[j], t_init, max_corr_m=corr, use_color=use_color)
            delta = reg.transformation @ np.linalg.inv(t_init)
            shift = float(np.linalg.norm(delta[:3, 3]))
            ang_deg = float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
            improves_vio = (not fine_mode) or (ev.fitness < 0.2) or (reg.inlier_rmse <= max(ev.inlier_rmse, 0.004) * 1.02)
            if reg.fitness >= (0.45 if fine_mode else 0.35) and reg.inlier_rmse <= max_rmse and shift <= max_shift and ang_deg <= max_ang and improves_vio:
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
    poses_pg = [np.array(pg.nodes[i].pose, dtype=np.float64) for i in range(n)]

    # Auto-verificação física: se o ICP deslizou sobre piso plano/liso e piorou o alinhamento do ARKit, descarta
    if fine_mode:
        step = max(1, (n - 1) // 20)
        eval_d = max(0.015, max_corr)
        rms_before, fit_before, rms_after, fit_after = [], [], [], []
        for i in range(0, n - 1, step):
            if len(frame_pcds[i].points) < 30 or len(frame_pcds[i + 1].points) < 30:
                continue
            tf0 = np.linalg.inv(poses_init[i + 1]) @ poses_init[i]
            tf1 = np.linalg.inv(poses_pg[i + 1]) @ poses_pg[i]
            ev0 = o3d.pipelines.registration.evaluate_registration(frame_pcds[i], frame_pcds[i + 1], eval_d, tf0)
            ev1 = o3d.pipelines.registration.evaluate_registration(frame_pcds[i], frame_pcds[i + 1], eval_d, tf1)
            if ev0.fitness > 0.1 and ev1.fitness > 0.1:
                rms_before.append(ev0.inlier_rmse)
                fit_before.append(ev0.fitness)
                rms_after.append(ev1.inlier_rmse)
                fit_after.append(ev1.fitness)
        if rms_before and (np.mean(rms_after) > np.mean(rms_before) * 1.01 or np.mean(fit_after) < np.mean(fit_before) * 0.96):
            return poses_init

    return poses_pg


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
    hires_only: bool = False,
    filter_mirror_floor: bool = False,
    cache_hires_rgb: bool = False,
) -> tuple[list[_ExtractedFrame], list[np.ndarray], float | None]:
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

    frames_src = bundle.frames
    if hires_only:
        hi_frames = [f for f in bundle.frames if f.image_file and (bundle.root / f.image_file).exists()]
        if len(hi_frames) >= 6:
            frames_src = hi_frames

    selected_frames = [frame for idx, frame in enumerate(frames_src) if idx % max(1, frame_stride) == 0]
    allow_ram_cache = cache_hires_rgb and len(selected_frames) <= 96
    imu_gyro_map = _load_imu_gyro_map(bundle)

    def _worker(fr: FrameRecord) -> _ExtractedFrame | None:
        return _extract_frame_points(
            bundle,
            fr,
            min_confidence=min_confidence,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            edge_jump_m=edge_jump_m,
            max_incidence_deg=max_incidence_deg,
            ref_img_w=ref_img_w,
            ref_img_h=ref_img_h,
            depth_upsample=depth_upsample,
            cache_hires_rgb=allow_ram_cache,
        )

    extracted: list[_ExtractedFrame] = []
    max_workers = min(8, os.cpu_count() or 4)
    if len(selected_frames) >= 4 and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_worker, selected_frames))
    else:
        results = [_worker(fr) for fr in selected_frames]

    for fr, ext in zip(selected_frames, results):
        if ext is not None:
            ext.gyro_deg_s = imu_gyro_map.get(fr.frame_id)
            extracted.append(ext)

    if not extracted:
        return [], [], None

    floor_y: float | None = None
    if filter_mirror_floor:
        extracted, _, floor_y = _filter_floor_mirror_outliers(extracted, [ext.pose_depth for ext in extracted])

    if mode in {"posegraph", "colored_icp"}:
        refined_poses = _refine_poses_posegraph(
            extracted,
            use_color=(mode == "colored_icp"),
            voxel_size_m=voxel_size_m,
        )
    else:
        refined_poses = [ext.pose_depth for ext in extracted]

    return extracted, refined_poses, floor_y


def _colorize_from_hires_rgb(
    bundle: LidupBundle,
    points_arkit: np.ndarray,
    normals_arkit: np.ndarray | None,
    extracted: list[_ExtractedFrame],
    refined_poses: list[np.ndarray],
    fallback_colors: np.ndarray,
    *,
    max_depth_m: float,
    occ_tol_m: float = 0.05,
) -> np.ndarray:
    """
    Projeta vértices/pontos finais diretamente nas fotos 12 MP (uma por vez na RAM)
    com interpolação bilinear subpixel, escolhendo o frame mais próximo/frontal/nítido
    com teste de oclusão contra o mapa LiDAR.
    """
    if len(points_arkit) == 0:
        return fallback_colors

    best_score = np.full(len(points_arkit), -1.0, dtype=np.float64)
    out_colors = fallback_colors.copy()
    sharp_vals = [ext.frame.sharpness for ext in extracted if ext.frame.sharpness > 0]
    med_sharp = float(np.median(sharp_vals)) if sharp_vals else 0.0

    for ext, pose in zip(extracted, refined_poses):
        # Mantém offset relativo RGB<->Depth quando PoseGraph ajustou a pose do frame
        rel_rgb = np.linalg.inv(ext.pose_depth) @ ext.frame.t_wc
        pose_rgb = pose @ rel_rgb

        sharp_w = 1.0
        if med_sharp > 0 and ext.frame.sharpness > 0:
            sharp_w = float(np.clip(np.sqrt(ext.frame.sharpness / med_sharp), 0.6, 1.4))
        imu_w = 1.0
        if ext.gyro_deg_s is not None:
            imu_w = float(np.clip(20.0 / max(ext.gyro_deg_s, 12.0), 0.55, 1.30))

        hi_u8 = ext.rgb_hi_u8
        if hi_u8 is None:
            if not ext.frame.image_file:
                continue
            img_path = bundle.root / ext.frame.image_file
            if not img_path.exists():
                continue

            cam_z_pre = (points_arkit[:, 0] - pose_rgb[0, 3]) * pose_rgb[0, 2] + (
                points_arkit[:, 1] - pose_rgb[1, 3]
            ) * pose_rgb[1, 2] + (points_arkit[:, 2] - pose_rgb[2, 3]) * pose_rgb[2, 2]
            if not np.any((cam_z_pre > 0.20) & (cam_z_pre <= max_depth_m)):
                continue

            with Image.open(img_path) as im:
                hi_u8 = np.asarray(im.convert("RGB"), dtype=np.uint8)

        hi_h, hi_w = hi_u8.shape[:2]
        sx = hi_w / float(max(ext.img_w, 1))
        sy = hi_h / float(max(ext.img_h, 1))
        fx_hi = ext.frame.fx * sx
        fy_hi = ext.frame.fy * sy
        cx_hi = ext.frame.cx * sx
        cy_hi = ext.frame.cy * sy

        if rust_colorize_hires_frame(
            points_arkit,
            normals_arkit,
            pose_rgb,
            hi_u8,
            ext.depth_filtered_m,
            fx_hi,
            fy_hi,
            cx_hi,
            cy_hi,
            max_depth_m,
            occ_tol_m,
            sharp_w * imu_w,
            best_score,
            out_colors,
        ):
            continue

        cam = (points_arkit - pose_rgb[:3, 3]) @ pose_rgb[:3, :3]
        z = cam[:, 2]
        in_range = (z > 0.20) & (z <= max_depth_m)
        if not np.any(in_range):
            continue

        safe_z = np.where(in_range, z, 1.0)
        u_f = fx_hi * (cam[:, 0] / safe_z) + cx_hi
        v_f = fy_hi * (cam[:, 1] / safe_z) + cy_hi
        inside = in_range & (u_f >= 0.0) & (u_f <= hi_w - 1.001) & (v_f >= 0.0) & (v_f <= hi_h - 1.001)
        if not np.any(inside):
            continue

        idx_cand = np.nonzero(inside)[0]
        u_d = np.clip(np.rint(u_f[idx_cand] * (ext.d_w / float(hi_w))).astype(np.int64), 0, ext.d_w - 1)
        v_d = np.clip(np.rint(v_f[idx_cand] * (ext.d_h / float(hi_h))).astype(np.int64), 0, ext.d_h - 1)
        d_obs = ext.depth_filtered_m[v_d, u_d]
        not_occluded = (d_obs > 0.0) & (np.abs(z[idx_cand] - d_obs) <= occ_tol_m)
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

        scores = (cos_v * sharp_w * imu_w) / np.maximum(z[idx_vis], 0.25) ** 2
        better = scores > best_score[idx_vis]
        idx_win = idx_vis[better]
        if len(idx_win) == 0:
            continue

        hi_arr = hi_u8.astype(np.float64) / 255.0
        uw = u_f[idx_win]
        vw = v_f[idx_win]
        u0 = np.floor(uw).astype(np.int64)
        v0 = np.floor(vw).astype(np.int64)
        du = (uw - u0)[:, None]
        dv = (vw - v0)[:, None]
        c00 = hi_arr[v0, u0]
        c10 = hi_arr[v0, u0 + 1]
        c01 = hi_arr[v0 + 1, u0]
        c11 = hi_arr[v0 + 1, u0 + 1]
        interp = c00 * (1.0 - du) * (1.0 - dv) + c10 * du * (1.0 - dv) + c01 * (1.0 - du) * dv + c11 * du * dv
        out_colors[idx_win] = interp
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

    extracted, refined_poses, _ = _prepare_extracted_and_poses(
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
        cache_hires_rgb=hires_color,
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
    for ext in extracted:
        ext.rgb_hi_u8 = None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_pts)
    pcd.colors = o3d.utility.Vector3dVector(merged_cols)

    if len(pcd.points) >= 50:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd


def _refine_depth_mvs(
    extracted: list[_ExtractedFrame],
    poses: list[np.ndarray],
    *,
    max_shift_m: float = 0.012,
    n_steps: int = 11,
) -> None:
    """
    Estéreo Multi-Visão guiado pelo LiDAR (LiDAR-Seeded MVS):
    refina cada mapa de profundidade buscando pequenos deslocamentos dz em [-12, +12] mm
    que maximizam ZNCC fotométrico contra quadros vizinhos, modulando pela textura local
    para recuperar micro-relevos reais sem criar ruído em áreas lisas.
    """
    rgb_indices = [i for i, e in enumerate(extracted) if e.rgb_arr is not None and np.any(e.depth_filtered_m > 0.0)]
    if len(rgb_indices) < 3:
        return

    norm_hp: dict[int, np.ndarray] = {}
    local_std: dict[int, np.ndarray] = {}
    for i in rgb_indices:
        rgb = extracted[i].rgb_arr
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        hp = gray - uniform_filter(gray, size=7)
        var = uniform_filter(hp * hp, size=7)
        local_std[i] = np.sqrt(np.maximum(var, 0.0))
        norm_hp[i] = hp / np.sqrt(var + 1e-4)

    cam_centers = {i: poses[i][:3, 3] for i in rgb_indices}
    cam_fwd = {i: poses[i][:3, 2] / max(float(np.linalg.norm(poses[i][:3, 2])), 1e-8) for i in rgb_indices}
    dz_grid = np.linspace(-float(max_shift_m), float(max_shift_m), max(5, int(n_steps)))
    step_m = float(dz_grid[1] - dz_grid[0])

    new_depths: dict[int, np.ndarray] = {}
    for i in rgb_indices:
        ext = extracted[i]
        d0 = ext.depth_filtered_m.astype(np.float64)
        valid = d0 > 0.0
        if np.count_nonzero(valid) < 80:
            continue

        # Escolhe até 4 quadros vizinhos com bom paralaxe (baseline 0.08..0.48 m) e vista concordante
        cands: list[tuple[float, int]] = []
        for j in rgb_indices:
            if j == i:
                continue
            dist = float(np.linalg.norm(cam_centers[j] - cam_centers[i]))
            cos_v = float(np.dot(cam_fwd[j], cam_fwd[i]))
            if dist < 0.05 or cos_v < 0.60:
                continue
            # Prefere baseline perto de 0.22 m e quadro com menor rotação na IMU
            gyro_pen = 1.0 + 0.02 * (extracted[j].gyro_deg_s or 15.0)
            score_nb = (abs(dist - 0.22) + (0.4 if dist < 0.08 or dist > 0.50 else 0.0)) * gyro_pen
            cands.append((score_nb, j))
        if not cands:
            continue
        cands.sort(key=lambda x: x[0])
        neighbors = [j for _, j in cands[:4]]

        d_h, d_w = d0.shape
        pose_i = poses[i]
        t_nb_list = [(j, np.linalg.inv(poses[j]) @ pose_i) for j in neighbors]
        ref_nhp = norm_hp[i]

        rust_mvs = rust_mvs_sweep(
            d0,
            valid,
            ref_nhp,
            float(ext.fx),
            float(ext.fy),
            float(ext.cx),
            float(ext.cy),
            [norm_hp[j] for j, _ in t_nb_list],
            [extracted[j].depth_filtered_m for j, _ in t_nb_list],
            [tf for _, tf in t_nb_list],
            [(extracted[j].d_w, extracted[j].d_h) for j, _ in t_nb_list],
            [(float(extracted[j].fx), float(extracted[j].fy), float(extracted[j].cx), float(extracted[j].cy)) for j, _ in t_nb_list],
            float(max_shift_m),
            len(dz_grid),
        )

        if rust_mvs is not None:
            dz_raw, best_score, best_vis = rust_mvs
        else:
            uu, vv = np.meshgrid(np.arange(d_w, dtype=np.float64), np.arange(d_h, dtype=np.float64))
            ray_x = (uu - ext.cx) / ext.fx
            ray_y = (vv - ext.cy) / ext.fy

            score_vol = np.zeros((len(dz_grid), d_h, d_w), dtype=np.float32)
            vis_vol = np.zeros((len(dz_grid), d_h, d_w), dtype=np.float32)

            for k_idx, dz in enumerate(dz_grid):
                d_cand = np.where(valid, np.maximum(0.15, d0 + dz), 0.0)
                pts_i = np.stack([ray_x * d_cand, ray_y * d_cand, d_cand], axis=-1)
                pts_flat = pts_i[valid]

                sum_corr = np.zeros((d_h, d_w), dtype=np.float64)
                cnt_vis = np.zeros((d_h, d_w), dtype=np.float64)

                for j, tf in t_nb_list:
                    ext_j = extracted[j]
                    p_j = pts_flat @ tf[:3, :3].T + tf[:3, 3]
                    z_j = p_j[:, 2]
                    in_z = z_j > 0.15
                    safe_z = np.where(in_z, z_j, 1.0)
                    u_j = ext_j.fx * (p_j[:, 0] / safe_z) + ext_j.cx
                    v_j = ext_j.fy * (p_j[:, 1] / safe_z) + ext_j.cy
                    in_b = in_z & (u_j >= 0.0) & (u_j <= ext_j.d_w - 1.001) & (v_j >= 0.0) & (v_j <= ext_j.d_h - 1.001)
                    if not np.any(in_b):
                        continue

                    uj_i = np.clip(np.rint(u_j[in_b]).astype(np.int64), 0, ext_j.d_w - 1)
                    vj_i = np.clip(np.rint(v_j[in_b]).astype(np.int64), 0, ext_j.d_h - 1)
                    dj_obs = ext_j.depth_filtered_m[vj_i, uj_i]
                    occ_ok = (dj_obs <= 0.0) | (np.abs(z_j[in_b] - dj_obs) <= 0.04)

                    warped_nb = np.zeros((d_h, d_w), dtype=np.float64)
                    vis_mask = np.zeros((d_h, d_w), dtype=bool)

                    valid_indices = np.nonzero(valid)
                    in_b_idx = np.nonzero(in_b)[0][occ_ok]
                    if len(in_b_idx) == 0:
                        continue

                    vr_sel = valid_indices[0][in_b_idx]
                    ur_sel = valid_indices[1][in_b_idx]
                    sampled = map_coordinates(norm_hp[j], [v_j[in_b_idx], u_j[in_b_idx]], order=1, mode="nearest")
                    warped_nb[vr_sel, ur_sel] = sampled
                    vis_mask[vr_sel, ur_sel] = True

                    corr = uniform_filter(ref_nhp * warped_nb, size=5)
                    sum_corr += np.where(vis_mask, np.maximum(corr, 0.0), 0.0)
                    cnt_vis += vis_mask

                score_vol[k_idx] = (sum_corr / np.maximum(cnt_vis, 1.0)).astype(np.float32)
                vis_vol[k_idx] = cnt_vis.astype(np.float32)

            best_k = np.argmax(score_vol, axis=0)
            best_score = np.take_along_axis(score_vol, best_k[None, :, :], axis=0)[0].astype(np.float64)
            best_vis = np.take_along_axis(vis_vol, best_k[None, :, :], axis=0)[0].astype(np.float64)

            k_m = np.clip(best_k - 1, 0, len(dz_grid) - 1)
            k_p = np.clip(best_k + 1, 0, len(dz_grid) - 1)
            s_m = np.take_along_axis(score_vol, k_m[None, :, :], axis=0)[0].astype(np.float64)
            s_p = np.take_along_axis(score_vol, k_p[None, :, :], axis=0)[0].astype(np.float64)
            denom = 2.0 * (s_m - 2.0 * best_score + s_p)
            interior = (best_k > 0) & (best_k < len(dz_grid) - 1) & (denom < -1e-4)
            sub_off = np.where(interior, np.clip((s_m - s_p) / np.where(interior, denom, -1.0), -0.5, 0.5), 0.0)
            dz_raw = dz_grid[best_k] + sub_off * step_m

        tex_w = np.clip((local_std[i] - 0.015) / 0.035, 0.0, 1.0)
        corr_w = np.clip((best_score - 0.30) / 0.40, 0.0, 1.0)
        alpha = np.where(valid & (best_vis >= 1.0), tex_w * corr_w, 0.0)
        dz_smooth = gaussian_filter(dz_raw * alpha, sigma=0.8) / np.maximum(gaussian_filter(alpha, sigma=0.8), 1e-4)
        new_depths[i] = np.where(valid, d0 + alpha * np.clip(dz_smooth, -max_shift_m, max_shift_m), 0.0)

    for i, d_new in new_depths.items():
        ext = extracted[i]
        mask = d_new > 0.0
        if not np.any(mask):
            continue
        rust_re = rust_extract_depth_points(d_new, mask, ext.fx, ext.fy, ext.cx, ext.cy, 0.0, -1.0)
        if rust_re is not None:
            _, cam_pts, weights, v_idx, u_idx = rust_re
            if len(cam_pts) == 0:
                continue
        else:
            d_h, d_w = d_new.shape
            uu, vv = np.meshgrid(np.arange(d_w, dtype=np.float64), np.arange(d_h, dtype=np.float64))
            pts_grid = np.stack([(uu - ext.cx) * d_new / ext.fx, (vv - ext.cy) * d_new / ext.fy, d_new], axis=-1)
            cos_inc, _ = _compute_incidence_cos(pts_grid)
            v_idx, u_idx = np.nonzero(mask)
            cam_pts = pts_grid[v_idx, u_idx]
            weights = np.clip(cos_inc[v_idx, u_idx], 0.1, 1.0) / np.maximum(cam_pts[:, 2], 0.30) ** 2

        ext.cam_pts = cam_pts
        ext.v_idx = v_idx
        ext.u_idx = u_idx
        ext.weights = weights
        ext.depth_filtered_m = d_new.astype(np.float32)


def _optimize_photometric_colormap(
    mesh: o3d.geometry.TriangleMesh,
    extracted: list[_ExtractedFrame],
    poses: list[np.ndarray],
    *,
    max_depth_m: float,
    max_iter: int = 80,
) -> tuple[o3d.geometry.TriangleMesh, list[np.ndarray]]:
    """
    Executa ajuste fotométrico rígido (Open3D ColorMapOptimizer) refinando as poses
    das câmeras diretamente sobre as bordas visuais da malha 3D.
    """
    rgb_indices = [i for i, e in enumerate(extracted) if e.rgb_arr is not None and len(e.cam_pts) > 50]
    if len(rgb_indices) < 2 or len(mesh.triangles) == 0:
        return mesh, poses

    rgbd_images: list[o3d.geometry.RGBDImage] = []
    traj = o3d.camera.PinholeCameraTrajectory()
    params: list[o3d.camera.PinholeCameraParameters] = []
    for i in rgb_indices:
        ext = extracted[i]
        color_u8 = np.clip(ext.rgb_arr * 255.0, 0.0, 255.0).astype(np.uint8)
        depth_mm = np.clip(np.rint(ext.depth_filtered_m * 1000.0), 0.0, 65535.0).astype(np.uint16)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color_u8)),
            o3d.geometry.Image(np.ascontiguousarray(depth_mm)),
            depth_scale=1000.0,
            depth_trunc=float(max_depth_m),
            convert_rgb_to_intensity=False,
        )
        rgbd_images.append(rgbd)

        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(ext.d_w, ext.d_h, ext.fx, ext.fy, ext.cx, ext.cy)
        param.extrinsic = np.linalg.inv(poses[i])
        params.append(param)
    traj.parameters = params

    option = o3d.pipelines.color_map.RigidOptimizerOption(
        maximum_iteration=int(max_iter),
        maximum_allowable_depth=float(max_depth_m),
        depth_threshold_for_visibility_check=0.025,
        depth_threshold_for_discontinuity_check=0.08,
        half_dilation_kernel_size_for_discontinuity_map=2,
        image_boundary_margin=6,
        invisible_vertex_color_knn=3,
    )
    try:
        mesh_opt, traj_opt = o3d.pipelines.color_map.run_rigid_optimizer(
            o3d.geometry.TriangleMesh(mesh), rgbd_images, traj, option
        )
    except RuntimeError:
        return mesh, poses

    out_poses = [p.copy() for p in poses]
    for slot, i in enumerate(rgb_indices):
        ext_mat = np.asarray(traj_opt.parameters[slot].extrinsic, dtype=np.float64)
        pose_opt = np.linalg.inv(ext_mat)
        delta = pose_opt @ np.linalg.inv(poses[i])
        shift = float(np.linalg.norm(delta[:3, 3]))
        ang_deg = float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
        if np.all(np.isfinite(pose_opt)) and shift <= 0.03 and ang_deg <= 2.5:
            out_poses[i] = pose_opt
    return mesh_opt, out_poses


def _integrate_uniform_tsdf(
    extracted: list[_ExtractedFrame],
    refined_poses: list[np.ndarray],
    lo: np.ndarray,
    length: float,
    res: int,
    vox: float,
    max_depth_m: float,
) -> tuple[o3d.geometry.TriangleMesh, float]:
    actual_vox = length / float(res)
    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=length,
        resolution=res,
        sdf_trunc=max(0.008, min(0.035, actual_vox * 6.0)),
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
    return mesh, actual_vox


def reconstruct_tsdf_details(
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
    hires_only: bool = True,
    use_mvs: bool = False,
    photometric_opt: bool = False,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, list[_ExtractedFrame], list[np.ndarray]]:
    """
    Funde múltiplos quadros em uma malha 3D contínua via UniformTSDFVolume com super-resolução
    de profundidade guiada pelo RGB (depth_upsample), refinamento estéreo multi-visão (MVS)
    opcional e otimização fotométrica sobre os JPEGs 12 MP, retornando também os quadros
    e poses refinadas para texturização UV.
    """
    mode = pose_refine if pose_refine is not None else ("posegraph" if refine_poses else "none")
    if mode not in {"none", "posegraph", "colored_icp"}:
        mode = "posegraph" if refine_poses else "none"

    pose_voxel = max(tsdf_voxel_m * 2.5, 0.008) if (focus_radius_m is not None or tsdf_voxel_m <= 0.006) else max(tsdf_voxel_m * 3.0, 0.02)
    extracted, refined_poses, floor_y = _prepare_extracted_and_poses(
        bundle,
        min_confidence=min_confidence,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        voxel_size_m=pose_voxel,
        frame_stride=frame_stride,
        edge_jump_m=edge_jump_m,
        max_incidence_deg=max_incidence_deg,
        mode=mode,
        depth_upsample=depth_upsample,
        hires_only=hires_only,
        filter_mirror_floor=(focus_radius_m is not None),
        cache_hires_rgb=hires_color,
    )
    if not extracted:
        return o3d.geometry.PointCloud(), o3d.geometry.TriangleMesh(), [], []

    if use_mvs:
        _refine_depth_mvs(extracted, refined_poses)

    vox = max(0.0008, float(tsdf_voxel_m))
    max_res = 480
    mesh = o3d.geometry.TriangleMesh()
    actual_vox = vox

    if focus_radius_m is not None and focus_radius_m > 0.15:
        ray_focus = _estimate_orbit_focus_point(refined_poses)
        if ray_focus is not None:
            focus = _refine_object_focus_point(extracted, refined_poses, ray_focus, floor_y)
            r = float(focus_radius_m)
            lo_focus = focus - r
            length_focus = max(0.36, 2.0 * r)
            res_focus = min(max_res, max(64, int(np.ceil(length_focus / vox))))
            mesh, actual_vox = _integrate_uniform_tsdf(extracted, refined_poses, lo_focus, length_focus, res_focus, vox, max_depth_m)

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
        mesh, actual_vox = _integrate_uniform_tsdf(extracted, refined_poses, lo, length, res, vox, max_depth_m)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if len(mesh.vertices) > 0 and actual_vox <= 0.0045 and not use_mvs:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=2)
    mesh.compute_vertex_normals()

    if photometric_opt and len(mesh.triangles) > 0:
        mesh, refined_poses = _optimize_photometric_colormap(
            mesh, extracted, refined_poses, max_depth_m=max_depth_m
        )
        mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    norms = np.asarray(mesh.vertex_normals, dtype=np.float64)
    cols = np.asarray(mesh.vertex_colors, dtype=np.float64)
    if len(cols) != len(verts):
        cols = np.full((len(verts), 3), 0.65, dtype=np.float64)

    occ_tol = 0.025 if (focus_radius_m is not None and focus_radius_m <= 1.0) else 0.045
    if hires_color and len(verts) > 0:
        cols = _colorize_from_hires_rgb(
            bundle,
            verts,
            norms,
            extracted,
            refined_poses,
            cols,
            max_depth_m=max_depth_m,
            occ_tol_m=occ_tol,
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(cols)

    for ext in extracted:
        ext.rgb_hi_u8 = None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd, mesh, extracted, refined_poses


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
    hires_only: bool = True,
    use_mvs: bool = False,
    photometric_opt: bool = False,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """Compat wrapper que retorna apenas `(PointCloud, TriangleMesh)`."""
    pcd, mesh, _, _ = reconstruct_tsdf_details(
        bundle,
        min_confidence=min_confidence,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        tsdf_voxel_m=tsdf_voxel_m,
        frame_stride=frame_stride,
        edge_jump_m=edge_jump_m,
        refine_poses=refine_poses,
        pose_refine=pose_refine,
        max_incidence_deg=max_incidence_deg,
        hires_color=hires_color,
        depth_upsample=depth_upsample,
        focus_radius_m=focus_radius_m,
        hires_only=hires_only,
        use_mvs=use_mvs,
        photometric_opt=photometric_opt,
    )
    return pcd, mesh
