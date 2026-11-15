from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from pyproj import Transformer

from .bundle import LidupBundle


@dataclass
class GeorefResult:
    epsg: int
    crs_name: str
    mode: str
    scale: float
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    origin_utm: np.ndarray  # (3,) [E0, N0, Z0] usado para exportação local sem jitter
    rmse_xyz_m: tuple[float, float, float] | None
    rmse_total_m: float | None
    control_points_used: int
    notes: str


def sirgas2000_utm_epsg(lat: float, lon: float) -> tuple[int, str]:
    """
    Seleciona o código EPSG SIRGAS 2000 / UTM adequado para a latitude/longitude.
    No Brasil (zonas 18S a 25S): EPSG:31978 a EPSG:31985.
    """
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    if -60.0 <= lat <= 14.0 and -90.0 <= lon <= -30.0:
        if lat < 0:
            epsg = 31960 + zone
            return epsg, f"SIRGAS 2000 / UTM zone {zone}S"
        epsg = 31954 + zone  # SIRGAS 2000 / UTM Norte (zonas 11N-22N: 31965..31976)
        if 11 <= zone <= 22:
            epsg = 31954 + zone
            return epsg, f"SIRGAS 2000 / UTM zone {zone}N"

    base = 32700 if lat < 0 else 32600
    hemi = "S" if lat < 0 else "N"
    return base + zone, f"WGS 84 / UTM zone {zone}{hemi}"


def umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    allow_scale: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calcula transformação de similaridade 3D (Umeyama/Kabsch): dst ≈ scale * (R @ src) + t.
    Por padrão mantém `allow_scale=False` para não deixar ruído de GPS comum deformar a escala métrica do LiDAR.
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src e dst devem ter shape (N, 3)")
    n = src.shape[0]
    if n < 3:
        raise ValueError("São necessários pelo menos 3 pontos para alinhamento 3D")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d

    cov = (dst_c.T @ src_c) / float(n)
    u, singular_vals, vt = np.linalg.svd(cov)

    d_mat = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d_mat[2, 2] = -1.0

    rot = u @ d_mat @ vt

    if allow_scale:
        var_s = float(np.sum(src_c ** 2) / float(n))
        scale = float(np.sum(singular_vals * np.diag(d_mat)) / max(var_s, 1e-12))
    else:
        scale = 1.0

    trans = mu_d - scale * (rot @ mu_s)
    return scale, rot, trans


def compute_scale_from_bars(gcps: dict[str, Any]) -> float | None:
    """Estima fator de escala métrica a partir de medidas de trena entre estacas registradas no ARKit."""
    points_by_id = {p["id"]: np.array(p["arkit_xyz"], dtype=np.float64) for p in gcps.get("points", [])}
    ratios: list[float] = []
    for bar in gcps.get("scale_bars", []):
        pa = points_by_id.get(bar.get("point_a_id"))
        pb = points_by_id.get(bar.get("point_b_id"))
        measured = float(bar.get("measured_distance_m", 0))
        if pa is None or pb is None or measured <= 0:
            continue
        arkit_dist = float(np.linalg.norm(pb - pa))
        if arkit_dist > 1e-3:
            ratios.append(measured / arkit_dist)
    if not ratios:
        return None
    return float(np.median(ratios))


def arkit_to_engineering_local(points_arkit: np.ndarray) -> np.ndarray:
    """
    Converte coordenadas do mundo ARKit (+X dir, +Y cima, +Z trás) para referencial local
    topográfico (+X leste/dir, +Y frente/-Z_arkit, +Z cima).
    """
    out = np.empty_like(points_arkit, dtype=np.float64)
    out[:, 0] = points_arkit[:, 0]
    out[:, 1] = -points_arkit[:, 2]
    out[:, 2] = points_arkit[:, 1]
    return out


def _arkit_to_eng_matrix() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def georeference_points(
    points_arkit: np.ndarray,
    bundle: LidupBundle,
    *,
    allow_gnss_scale: bool = False,
) -> tuple[np.ndarray, GeorefResult]:
    """
    Transforma pontos do mundo ARKit para coordenadas projetadas (SIRGAS 2000 / UTM) ou
    local topográfico (Z para cima) quando não há GNSS suficiente.
    """
    bar_scale = compute_scale_from_bars(bundle.gcps)

    # 1. Tenta usar estacas/GCPs estacionários de `gcps.json`
    gcp_candidates = [
        p
        for p in bundle.gcps.get("points", [])
        if p.get("lat") is not None and p.get("lon") is not None and p.get("alt_ellipsoidal_m") is not None
    ]

    if len(gcp_candidates) >= 3:
        ref_lat = float(np.mean([p["lat"] for p in gcp_candidates]))
        ref_lon = float(np.mean([p["lon"] for p in gcp_candidates]))
        epsg, crs_name = sirgas2000_utm_epsg(ref_lat, ref_lon)
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

        src_pts = np.array([p["arkit_xyz"] for p in gcp_candidates], dtype=np.float64)
        dst_pts = np.array(
            [
                (*transformer.transform(float(p["lon"]), float(p["lat"])), float(p["alt_ellipsoidal_m"]))
                for p in gcp_candidates
            ],
            dtype=np.float64,
        )

        # Se houver trena, pré-escala pontos ARKit antes de ajustar rotação+translação rígida
        pre_scale = bar_scale if bar_scale is not None else 1.0
        scale_u, rot, trans = umeyama_alignment(
            src_pts * pre_scale,
            dst_pts,
            allow_scale=(allow_gnss_scale and bar_scale is None),
        )
        total_scale = pre_scale * scale_u

        aligned_gcp = (src_pts * total_scale) @ rot.T + trans
        err = aligned_gcp - dst_pts
        rmse_xyz = tuple(float(np.sqrt(np.mean(err[:, i] ** 2))) for i in range(3))
        rmse_total = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))

        utm_pts = (points_arkit * total_scale) @ rot.T + trans
        origin = utm_pts.mean(axis=0)
        return utm_pts, GeorefResult(
            epsg=epsg,
            crs_name=crs_name,
            mode="stationary_gcps",
            scale=total_scale,
            rotation=rot,
            translation=trans,
            origin_utm=origin,
            rmse_xyz_m=rmse_xyz,
            rmse_total_m=rmse_total,
            control_points_used=len(gcp_candidates),
            notes="Alinhamento 3D via estacas estacionárias (gcps.json)"
            + (" com escala calibrada por trena." if bar_scale else "."),
        )

    # 2. Fallback: usa leituras GNSS associadas aos frames quando houver deslocamento suficiente
    gnss_frames = [f for f in bundle.frames if f.lat is not None and f.lon is not None and f.alt_m is not None]
    if len(gnss_frames) >= 3:
        ref_lat = float(np.mean([f.lat for f in gnss_frames]))  # type: ignore[arg-type]
        ref_lon = float(np.mean([f.lon for f in gnss_frames]))  # type: ignore[arg-type]
        epsg, crs_name = sirgas2000_utm_epsg(ref_lat, ref_lon)
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

        src_pts = np.array([f.t_wc[:3, 3] for f in gnss_frames], dtype=np.float64)
        dst_pts = np.array(
            [
                (*transformer.transform(float(f.lon), float(f.lat)), float(f.alt_m))  # type: ignore[arg-type]
                for f in gnss_frames
            ],
            dtype=np.float64,
        )

        # Só faz alinhamento 3D se o trajeto teve espalhamento mínimo (>= 3 m)
        span = float(np.linalg.norm(src_pts.max(axis=0) - src_pts.min(axis=0)))
        if span >= 3.0:
            pre_scale = bar_scale if bar_scale is not None else 1.0
            scale_u, rot, trans = umeyama_alignment(
                src_pts * pre_scale,
                dst_pts,
                allow_scale=(allow_gnss_scale and bar_scale is None),
            )
            total_scale = pre_scale * scale_u
            aligned = (src_pts * total_scale) @ rot.T + trans
            err = aligned - dst_pts
            rmse_xyz = tuple(float(np.sqrt(np.mean(err[:, i] ** 2))) for i in range(3))
            rmse_total = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))

            utm_pts = (points_arkit * total_scale) @ rot.T + trans
            origin = utm_pts.mean(axis=0)
            return utm_pts, GeorefResult(
                epsg=epsg,
                crs_name=crs_name,
                mode="frame_gnss",
                scale=total_scale,
                rotation=rot,
                translation=trans,
                origin_utm=origin,
                rmse_xyz_m=rmse_xyz,
                rmse_total_m=rmse_total,
                control_points_used=len(gnss_frames),
                notes="Alinhamento 3D via GNSS dos keyframes (sem estacas estacionárias suficientes).",
            )

    # 3. Fallback local: apenas corrige eixos (Z para cima) + escala de trena se houver
    scale = bar_scale if bar_scale is not None else 1.0
    rot = _arkit_to_eng_matrix()
    eng_pts = (points_arkit * scale) @ rot.T
    origin = eng_pts.mean(axis=0)
    return eng_pts, GeorefResult(
        epsg=0,
        crs_name="LOCAL_ENGINEERING_METERS",
        mode="local_only",
        scale=scale,
        rotation=rot,
        translation=np.zeros(3, dtype=np.float64),
        origin_utm=origin,
        rmse_xyz_m=None,
        rmse_total_m=None,
        control_points_used=0,
        notes="Sem GNSS suficiente; mantido em coordenadas locais métricas com eixo +Z para cima.",
    )
