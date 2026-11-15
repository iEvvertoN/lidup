from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.spatial.transform import Rotation

from .georef import sirgas2000_utm_epsg


def _ground_height_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Superfície inclinada suave simulando terreno rural (eixo X comprido, Y estreito)."""
    return 0.06 * x + 0.015 * y + 0.25 * np.sin(0.25 * x) * np.cos(0.3 * y)


def generate_synthetic_bundle(
    out_dir: str | Path,
    *,
    length_x_m: float = 24.0,
    width_y_m: float = 12.0,
    seed: int = 42,
) -> Path:
    """
    Gera um pacote sintético completo no formato `lidup_bundle_v1` para testar
    todo o pipeline topográfico no Windows antes da captura real em campo.
    """
    rng = np.random.default_rng(seed)
    root = Path(out_dir)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    (root / "confidence").mkdir(parents=True, exist_ok=True)

    # Origem geográfica de referência (região rural no estado de SP -> SIRGAS 2000 UTM 23S)
    base_lat = -22.7250
    base_lon = -47.6490
    base_alt = 540.0
    epsg, _ = sirgas2000_utm_epsg(base_lat, base_lon)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    base_e, base_n = to_utm.transform(base_lon, base_lat)

    # Coordenadas ARKit: +X = Leste, +Y = Cima (elevação), -Z = Norte (y_eng = -z_arkit)
    # Câmera caminha a ~1.9m acima do solo olhando ~40 graus para baixo
    img_w, img_h = 640, 480
    depth_w, depth_h = 64, 48
    fx_img, fy_img = 520.0, 520.0
    cx_img, cy_img = img_w / 2.0, img_h / 2.0

    # Grade de pontos de câmera em 2 faixas longitudinais (ida e volta fechando loop)
    xs_path = np.concatenate([np.linspace(1.0, length_x_m - 1.0, 14), np.linspace(length_x_m - 1.0, 1.0, 14)])
    ys_path = np.concatenate([np.full(14, 3.0), np.full(14, width_y_m - 3.0)])

    # Obstáculos sintéticos (2 moitas/arbustos + 1 mourão) acima do terreno
    obstacles = [
        (8.0, 5.5, 1.1, 1.4),
        (16.5, 7.2, 1.3, 1.8),
    ]

    # Pré-calcula raios normalizados de profundidade
    u_d, v_d = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
    sx, sy = depth_w / float(img_w), depth_h / float(img_h)
    fx_d, fy_d = fx_img * sx, fy_img * sy
    cx_d, cy_d = cx_img * sx, cy_img * sy
    ray_x = (u_d - cx_d) / fx_d
    ray_y = (v_d - cy_d) / fy_d
    ray_z = np.ones_like(ray_x)
    rays_cam = np.stack([ray_x, ray_y, ray_z], axis=-1).reshape(-1, 3)

    frames_rows: list[list[str]] = []
    gnss_rows: list[list[str]] = []

    for idx, (ex, ey) in enumerate(zip(xs_path, ys_path)):
        frame_id = f"{idx:06d}"
        ez = float(_ground_height_fn(np.array([ex]), np.array([ey]))[0]) + 1.9
        # Converte posição topográfica (ex, ey, ez) para ARKit (X=ex, Y=ez, Z=-ey)
        pos_arkit = np.array([ex, ez, -ey], dtype=np.float64)

        # Câmera olhando para frente no sentido da faixa e 42 graus para baixo
        yaw = 0.0 if idx < 14 else np.pi
        # Base câmera OpenCV (+X dir, +Y baixo, +Z frente) -> ARKit
        # Quando yaw=0: frente é +X_eng (+X_arkit), baixo-frente é -Y_arkit/+X_arkit
        forward_eng = np.array([np.cos(yaw) * 0.74, np.sin(yaw) * 0.74, -0.67], dtype=np.float64)
        forward_eng /= np.linalg.norm(forward_eng)
        right_eng = np.cross(forward_eng, np.array([0.0, 0.0, 1.0]))
        right_eng /= np.linalg.norm(right_eng)
        down_eng = np.cross(forward_eng, right_eng)

        # Converte vetores de engenharia para ARKit (x, z, -y)
        def eng_to_arkit(v: np.ndarray) -> np.ndarray:
            return np.array([v[0], v[2], -v[1]], dtype=np.float64)

        r_wc = np.column_stack(
            [
                eng_to_arkit(right_eng),
                eng_to_arkit(down_eng),
                eng_to_arkit(forward_eng),
            ]
        )
        quat = Rotation.from_matrix(r_wc).as_quat()  # x, y, z, w

        # Raycast analítico rápido ao solo/obstáculos para gerar mapa de profundidade 16-bit realista
        rays_world_eng = (rays_cam @ r_wc.T)
        # Converte rays de ARKit para eng: (X, -Z, Y)
        rays_eng = np.column_stack([rays_world_eng[:, 0], -rays_world_eng[:, 2], rays_world_eng[:, 1]])

        t_vals = np.full(len(rays_eng), 0.0, dtype=np.float64)
        down_mask = rays_eng[:, 2] < -0.05
        # Interseção iterativa em 3 passos contra o solo ondulado
        t_est = np.zeros(len(rays_eng), dtype=np.float64)
        t_est[down_mask] = (ez - float(_ground_height_fn(np.array([ex]), np.array([ey]))[0])) / (-rays_eng[down_mask, 2])
        for _ in range(3):
            px = ex + t_est[down_mask] * rays_eng[down_mask, 0]
            py = ey + t_est[down_mask] * rays_eng[down_mask, 1]
            gz = _ground_height_fn(px, py)
            t_est[down_mask] = np.clip((ez - gz) / (-rays_eng[down_mask, 2]), 0.3, 6.0)

        # Sobrepõe obstáculos cilíndricos/esféricos mais altos que o terreno
        hit_x = ex + t_est * rays_eng[:, 0]
        hit_y = ey + t_est * rays_eng[:, 1]
        hit_z = ez + t_est * rays_eng[:, 2]
        for ox, oy, orad, oh in obstacles:
            dist_obs = np.hypot(hit_x - ox, hit_y - oy)
            obs_mask = down_mask & (dist_obs <= orad)
            if np.any(obs_mask):
                t_est[obs_mask] = np.maximum(0.4, t_est[obs_mask] - oh * 0.7)

        t_vals[down_mask] = t_est[down_mask]
        # Z no frame da câmera = profundidade planar (Z_cam), já que rays_cam[:, 2] == 1.0
        depth_m = t_vals.reshape((depth_h, depth_w))
        valid_d = (depth_m >= 0.35) & (depth_m <= 4.3)
        depth_mm = np.where(valid_d, np.clip(np.round(depth_m * 1000.0), 0, 65535), 0).astype(np.uint16)

        conf = np.where(valid_d, 2, 0).astype(np.uint8)
        # Borda inferior recebe confiança média (1) para testar filtro de confiança
        conf[0:2, :] = np.where(valid_d[0:2, :], 1, 0)

        Image.fromarray(depth_mm).save(root / "depth" / f"{frame_id}.png")
        Image.fromarray(conf).save(root / "confidence" / f"{frame_id}.png")

        # Imagem RGB sintética colorida por solo verde/terra
        rgb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        rgb[:, :, 0] = 115
        rgb[:, :, 1] = 135
        rgb[:, :, 2] = 75
        Image.fromarray(rgb, mode="RGB").save(root / "images" / f"{frame_id}.jpg", quality=92)

        lon, lat = to_wgs.transform(base_e + ex, base_n + ey)
        alt = base_alt + ez
        ts = 1763217000.0 + idx * 1.5

        frames_rows.append(
            [
                frame_id,
                f"{ts:.6f}",
                f"images/{frame_id}.jpg",
                f"depth/{frame_id}.png",
                f"confidence/{frame_id}.png",
                f"{pos_arkit[0]:.6f}",
                f"{pos_arkit[1]:.6f}",
                f"{pos_arkit[2]:.6f}",
                f"{quat[0]:.6f}",
                f"{quat[1]:.6f}",
                f"{quat[2]:.6f}",
                f"{quat[3]:.6f}",
                f"{fx_img:.2f}",
                f"{fy_img:.2f}",
                f"{cx_img:.2f}",
                f"{cy_img:.2f}",
                "420.0",
                "normal",
                f"{lat:.8f}",
                f"{lon:.8f}",
                f"{alt:.3f}",
                "2.2",
                "3.1",
            ]
        )
        gnss_rows.append(
            [f"{ts:.6f}", f"{lat:.8f}", f"{lon:.8f}", f"{alt:.3f}", "2.2", "3.1", "0.8", "90.0", "1", "synthetic"]
        )

    with (root / "frames.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "frame_id", "timestamp_s", "image_file", "depth_file", "confidence_file",
                "tx", "ty", "tz", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy",
                "sharpness", "tracking_state", "lat", "lon", "alt_m", "h_acc_m", "v_acc_m",
            ]
        )
        w.writerows(frames_rows)

    with (root / "gnss.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["timestamp_s", "lat", "lon", "alt_ellipsoidal_m", "h_acc_m", "v_acc_m", "speed_mps", "course_deg", "fix_quality", "source"]
        )
        w.writerows(gnss_rows)

    # 4 estacas estacionárias nos cantos da área útil + 1 medida de trena
    stake_coords = [
        ("ESTACA-01", 2.0, 2.0),
        ("ESTACA-02", length_x_m - 2.0, 2.0),
        ("ESTACA-03", length_x_m - 2.0, width_y_m - 2.0),
        ("ESTACA-04", 2.0, width_y_m - 2.0),
    ]
    gcp_points = []
    for i, (sid, sx_m, sy_m) in enumerate(stake_coords):
        sz_m = float(_ground_height_fn(np.array([sx_m]), np.array([sy_m]))[0])
        # Simula pequeno ruído realista de GPS (~0.15m no cenário sintético controlado)
        lon, lat = to_wgs.transform(
            base_e + sx_m + rng.normal(0, 0.08),
            base_n + sy_m + rng.normal(0, 0.08),
        )
        gcp_points.append(
            {
                "id": sid,
                "kind": "stationary_gnss_avg",
                "timestamp_s": 1763217000.0 + i * 20.0,
                "sample_count": 30,
                "lat": float(lat),
                "lon": float(lon),
                "alt_ellipsoidal_m": float(base_alt + sz_m + rng.normal(0, 0.10)),
                "h_acc_mean_m": 1.8,
                "v_acc_mean_m": 2.4,
                "arkit_xyz": [float(sx_m), float(sz_m), float(-sy_m)],
                "frame_id": f"{i * 5:06d}",
                "notes": "Estaca sintética de referência",
            }
        )

    gcps = {
        "points": gcp_points,
        "scale_bars": [
            {
                "id": "TRENA-01",
                "point_a_id": "ESTACA-01",
                "point_b_id": "ESTACA-02",
                "measured_distance_m": float(
                    np.linalg.norm(
                        np.array([length_x_m - 4.0, 0.0, _ground_height_fn(np.array([length_x_m - 2.0]), np.array([2.0]))[0] - _ground_height_fn(np.array([2.0]), np.array([2.0]))[0]])
                    )
                ),
            }
        ],
    }
    (root / "gcps.json").write_text(json.dumps(gcps, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": "lidup_bundle_v1",
        "session_id": root.name,
        "sector_name": "galpao_norte_demo",
        "created_at_iso": "2026-11-15T15:30:00Z",
        "device": {"model": "iPhone16,2", "system_version": "18.1", "has_lidar": True},
        "capture_config": {
            "min_translation_m": 0.8,
            "min_rotation_rad": 0.21,
            "video_stabilization_disabled": True,
            "focus_locked": True,
            "image_width": img_w,
            "image_height": img_h,
            "depth_width": depth_w,
            "depth_height": depth_h,
            "depth_unit": "millimeters_uint16",
        },
        "gnss_source": "synthetic",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root
