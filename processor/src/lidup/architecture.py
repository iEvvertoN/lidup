from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import open3d as o3d


@dataclass
class WallSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    length_m: float
    inlier_count: int


@dataclass
class RoomReport:
    source: str
    up_axis: str
    point_count: int
    width_m: float
    length_m: float
    ceiling_height_m: float
    ceiling_source: str
    floor_area_m2: float
    perimeter_m: float
    wall_area_m2: float
    detected_wall_count: int
    detected_walls_total_m: float
    floor_flatness_rms_mm: float
    floor_flatness_p95_mm: float
    floor_flatness_max_mm: float
    floor_normal: list[float]
    floor_offset_d: float
    slice_band_m: list[float]
    walls: list[dict[str, Any]]
    outputs: dict[str, str]


def _extract_points_from_input(target: str | Path | np.ndarray | o3d.geometry.Geometry) -> tuple[np.ndarray, str]:
    """Carrega matriz Nx3 de pontos a partir de array, geometria Open3D, arquivo ou pasta."""
    if isinstance(target, np.ndarray):
        pts = np.asarray(target, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("Array de pontos deve ter shape (N, 3).")
        return pts, "numpy_array"

    if isinstance(target, o3d.geometry.PointCloud):
        return np.asarray(target.points, dtype=np.float64), "open3d_point_cloud"

    if isinstance(target, o3d.geometry.TriangleMesh):
        return np.asarray(target.vertices, dtype=np.float64), "open3d_triangle_mesh"

    path = Path(target)
    if path.is_dir():
        candidates = [
            path / "cloud_local.ply",
            path / "object_mesh_local.ply",
            path / "object_textured.obj",
            path / "cloud_georef.ply",
            path / "cloud_georef.laz",
            path / "cloud_georef.las",
        ]
        chosen = next((c for c in candidates if c.exists()), None)
        if chosen is None:
            raise FileNotFoundError(f"Nenhuma nuvem/malha encontrada em {path}")
        path = chosen

    if not path.exists():
        raise FileNotFoundError(f"Alvo não encontrado: {path}")

    ext = path.suffix.lower()
    if ext in {".las", ".laz"}:
        las = laspy.read(path)
        pts = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
        return pts, str(path)

    if ext == ".obj":
        mesh = o3d.io.read_triangle_mesh(str(path))
        pts = np.asarray(mesh.vertices, dtype=np.float64)
        return pts, str(path)

    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) == 0:
        mesh = o3d.io.read_triangle_mesh(str(path))
        pts = np.asarray(mesh.vertices, dtype=np.float64)
    return pts, str(path)


def _convex_hull_2d(pts_2d: np.ndarray) -> np.ndarray:
    """Casco convexo 2D (Andrew monotone chain) sem dependências externas."""
    if len(pts_2d) <= 2:
        return pts_2d
    uniq = np.unique(np.round(pts_2d, 4), axis=0)
    if len(uniq) <= 2:
        return uniq
    order = np.lexsort((uniq[:, 1], uniq[:, 0]))
    sorted_pts = uniq[order]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _polygon_area_and_perimeter(poly: np.ndarray) -> tuple[float, float]:
    if len(poly) < 3:
        return 0.0, 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    diffs = np.roll(poly, -1, axis=0) - poly
    perim = float(np.sum(np.linalg.norm(diffs, axis=1)))
    return area, perim


def _evaluate_up_candidate(pts_ds: np.ndarray, axis_vec: np.ndarray) -> tuple[float, np.ndarray | None]:
    """Avalia se um eixo candidato (+Y ou +Z) possui um piso horizontal extenso na base."""
    proj = pts_ds @ axis_vec
    thr = float(np.percentile(proj, 35.0))
    low_pts = pts_ds[proj <= thr]
    if len(low_pts) < 30:
        return 0.0, None
    pcd_low = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(low_pts))
    model, inliers = pcd_low.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=400)
    if len(inliers) < 20:
        return 0.0, None
    normal = np.asarray(model[:3], dtype=np.float64)
    normal /= max(1e-9, float(np.linalg.norm(normal)))
    align = abs(float(np.dot(normal, axis_vec)))
    if align < 0.78:
        return 0.0, None
    inl_pts = low_pts[inliers]
    # Estima área da projeção ortogonal a `axis_vec`
    mask_other = [i for i in range(3) if abs(axis_vec[i]) < 0.5]
    hull = _convex_hull_2d(inl_pts[:, mask_other])
    area, _ = _polygon_area_and_perimeter(hull)
    score = area * align * (len(inliers) / max(1, len(low_pts)))
    return score, model


def _choose_up_axis(pts: np.ndarray, source_label: str, up_axis: str = "auto") -> tuple[str, np.ndarray]:
    axis_clean = up_axis.strip().lower()
    if axis_clean == "y":
        return "Y", np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if axis_clean == "z":
        return "Z", np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if axis_clean != "auto":
        raise ValueError(f"up_axis inválido: {up_axis}")

    lower_src = source_label.lower()
    if any(tok in lower_src for tok in ("georef", ".las", ".laz", "dtm")):
        return "Z", np.array([0.0, 0.0, 1.0], dtype=np.float64)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd_ds = pcd.voxel_down_sample(0.04) if len(pts) > 5000 else pcd
    pts_ds = np.asarray(pcd_ds.points, dtype=np.float64)

    score_y, _ = _evaluate_up_candidate(pts_ds, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    score_z, _ = _evaluate_up_candidate(pts_ds, np.array([0.0, 0.0, 1.0], dtype=np.float64))

    # Em nuvens locais ARKit (+Y up), só troca para Z se Z tiver área/suporte claramente superior.
    if score_z > score_y * 1.25:
        return "Z", np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return "Y", np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _fit_floor_and_align(pts: np.ndarray, nominal_up: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Detecta o plano do piso por RANSAC + refino SVD e projeta os pontos para (u, v, h)."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd_ds = pcd.voxel_down_sample(0.03) if len(pts) > 12000 else pcd
    pts_ds = np.asarray(pcd_ds.points, dtype=np.float64)

    proj = pts_ds @ nominal_up
    low_mask = proj <= float(np.percentile(proj, 38.0))
    work_pts = pts_ds[low_mask]
    work_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(work_pts))

    floor_normal = nominal_up.copy()
    floor_d = -float(np.percentile(pts @ nominal_up, 2.0))

    for _ in range(4):
        if len(work_pcd.points) < 20:
            break
        model, inliers = work_pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=800)
        n = np.asarray(model[:3], dtype=np.float64)
        norm_len = float(np.linalg.norm(n))
        if norm_len < 1e-9:
            break
        n /= norm_len
        d = float(model[3]) / norm_len
        if float(np.dot(n, nominal_up)) < 0.0:
            n = -n
            d = -d
        if float(np.dot(n, nominal_up)) >= 0.78 and len(inliers) >= 20:
            floor_normal = n
            floor_d = d
            break
        work_pcd = work_pcd.select_by_index(inliers, invert=True)

    # Refino SVD sobre todos os pontos dentro de ±0.055 m do plano RANSAC
    signed_init = pts @ floor_normal + floor_d
    inlier_mask = np.abs(signed_init) <= 0.055
    if int(np.sum(inlier_mask)) >= 50:
        floor_pts = pts[inlier_mask]
        centroid = floor_pts.mean(axis=0)
        centered = floor_pts - centroid
        if len(centered) > 50000:
            step = max(1, len(centered) // 50000)
            centered = centered[::step]
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        refined_n = vt[-1]
        if float(np.dot(refined_n, floor_normal)) < 0.0:
            refined_n = -refined_n
        if float(np.dot(refined_n, nominal_up)) >= 0.78:
            floor_normal = refined_n / max(1e-9, float(np.linalg.norm(refined_n)))
            floor_d = -float(np.dot(floor_normal, centroid))

    # Base ortonormal (u, v, floor_normal)
    ref_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref_x, floor_normal))) > 0.9:
        ref_x = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u_axis = ref_x - float(np.dot(ref_x, floor_normal)) * floor_normal
    u_axis /= max(1e-9, float(np.linalg.norm(u_axis)))
    v_axis = np.cross(floor_normal, u_axis)
    v_axis /= max(1e-9, float(np.linalg.norm(v_axis)))

    h = pts @ floor_normal + floor_d
    # Garante que a maior parte do ambiente esteja em h >= 0
    if float(np.percentile(h, 50.0)) < -0.1:
        floor_normal = -floor_normal
        floor_d = -floor_d
        h = -h
        v_axis = -v_axis

    aligned = np.column_stack([pts @ u_axis, pts @ v_axis, h])
    return aligned, floor_normal, floor_d, inlier_mask


def _estimate_ceiling_height(aligned: np.ndarray) -> tuple[float, str]:
    h = aligned[:, 2]
    p99 = float(np.percentile(h, 99.0))
    if p99 < 1.3:
        return max(0.0, p99), "p99_height"

    top_thr = max(1.3, p99 - 0.35)
    top_pts = aligned[h >= top_thr]
    if len(top_pts) < 120:
        return p99, "p99_height"

    pcd_top = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(top_pts))
    if len(top_pts) > 12000:
        pcd_top = pcd_top.voxel_down_sample(0.03)

    work = pcd_top
    total_top = len(work.points)
    for _ in range(3):
        if len(work.points) < 60:
            break
        model, inliers = work.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=600)
        nz = abs(float(model[2])) / max(1e-9, float(np.linalg.norm(model[:3])))
        if nz >= 0.88 and len(inliers) >= max(80, int(0.10 * total_top)):
            ceil_h = float(np.median(np.asarray(work.points)[inliers, 2]))
            return ceil_h, "ceiling_plane"
        work = work.select_by_index(inliers, invert=True)

    return p99, "p99_height"


def _extract_walls_and_align_2d(
    aligned: np.ndarray,
    ceiling_height_m: float,
) -> tuple[np.ndarray, np.ndarray, list[WallSegment], tuple[float, float]]:
    """Extrai paredes verticais na faixa de corte 2D e gira o plano XY pela parede dominante."""
    h_max = max(0.9, ceiling_height_m)
    band_hi = min(1.80, max(0.75, h_max - 0.15))
    band_lo = min(0.60, band_hi * 0.45)

    h = aligned[:, 2]
    slice_pts = aligned[(h >= band_lo) & (h <= band_hi)]
    if len(slice_pts) < 60:
        slice_pts = aligned[(h >= 0.20) & (h <= h_max)]

    pcd_slice = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(slice_pts))
    if len(slice_pts) > 4000:
        pcd_slice = pcd_slice.voxel_down_sample(0.025)
    raw_slice_2d = np.asarray(pcd_slice.points)[:, :2]

    work = pcd_slice
    total_slice = max(1, len(work.points))
    raw_walls: list[tuple[np.ndarray, np.ndarray, int]] = []

    for _ in range(12):
        if len(work.points) < 50:
            break
        model, inliers = work.segment_plane(distance_threshold=0.035, ransac_n=3, num_iterations=600)
        n = np.asarray(model[:3], dtype=np.float64)
        n /= max(1e-9, float(np.linalg.norm(n)))
        nz = abs(float(n[2]))
        inl_pts = np.asarray(work.points)[inliers]
        work = work.select_by_index(inliers, invert=True)
        if nz > 0.25 or len(inliers) < max(45, int(0.025 * total_slice)):
            continue
        # Projeta para 2D e recorta pelos percentis [2, 98]
        dir_2d = np.array([-n[1], n[0]], dtype=np.float64)
        dir_2d /= max(1e-9, float(np.linalg.norm(dir_2d)))
        pts_2d = inl_pts[:, :2]
        origin = pts_2d.mean(axis=0)
        s = (pts_2d - origin) @ dir_2d
        s_lo, s_hi = float(np.percentile(s, 2.0)), float(np.percentile(s, 98.0))
        length = s_hi - s_lo
        if length >= 0.35:
            p1 = origin + s_lo * dir_2d
            p2 = origin + s_hi * dir_2d
            raw_walls.append((p1, p2, len(inliers)))

    # Alinha eixos 2D pela orientação da parede mais longa
    rot_angle = 0.0
    if raw_walls:
        best_p1, best_p2, _ = max(raw_walls, key=lambda item: float(np.linalg.norm(item[1] - item[0])))
        delta = best_p2 - best_p1
        rot_angle = -math.atan2(float(delta[1]), float(delta[0]))

    c, s = math.cos(rot_angle), math.sin(rot_angle)
    rot_2d = np.array([[c, -s], [s, c]], dtype=np.float64)

    aligned_out = aligned.copy()
    aligned_out[:, :2] = aligned[:, :2] @ rot_2d.T
    slice_2d = raw_slice_2d @ rot_2d.T

    walls: list[WallSegment] = []
    for p1, p2, cnt in raw_walls:
        q1 = rot_2d @ p1
        q2 = rot_2d @ p2
        seg_len = float(np.linalg.norm(q2 - q1))
        walls.append(
            WallSegment(
                x1=round(float(q1[0]), 4),
                y1=round(float(q1[1]), 4),
                x2=round(float(q2[0]), 4),
                y2=round(float(q2[1]), 4),
                length_m=round(seg_len, 4),
                inlier_count=int(cnt),
            )
        )

    return aligned_out, slice_2d, walls, (round(band_lo, 2), round(band_hi, 2))


def _export_floorplan_svg(
    out_path: Path,
    slice_2d: np.ndarray,
    walls: list[WallSegment],
    hull_2d: np.ndarray,
    report: RoomReport,
) -> None:
    pts_all = slice_2d if len(slice_2d) > 0 else hull_2d
    if len(pts_all) == 0:
        out_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='800' height='600'></svg>", encoding="utf-8")
        return

    min_x = float(min(np.min(pts_all[:, 0]), *(min(w.x1, w.x2) for w in walls))) if walls else float(np.min(pts_all[:, 0]))
    max_x = float(max(np.max(pts_all[:, 0]), *(max(w.x1, w.x2) for w in walls))) if walls else float(np.max(pts_all[:, 0]))
    min_y = float(min(np.min(pts_all[:, 1]), *(min(w.y1, w.y2) for w in walls))) if walls else float(np.min(pts_all[:, 1]))
    max_y = float(max(np.max(pts_all[:, 1]), *(max(w.y1, w.y2) for w in walls))) if walls else float(np.max(pts_all[:, 1]))

    pad_m = 0.5
    min_x -= pad_m
    max_x += pad_m
    min_y -= pad_m
    max_y += pad_m

    span_x = max(0.5, max_x - min_x)
    span_y = max(0.5, max_y - min_y)

    canvas_w, canvas_h = 1000.0, 760.0
    margin = 70.0
    draw_w = canvas_w - 2.0 * margin
    draw_h = canvas_h - 2.0 * margin - 60.0
    scale = min(draw_w / span_x, draw_h / span_y)

    def to_px(x: float, y: float) -> tuple[float, float]:
        px = margin + (x - min_x) * scale
        py = margin + 40.0 + (max_y - y) * scale
        return px, py

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(canvas_w)}" height="{int(canvas_h)}" viewBox="0 0 {int(canvas_w)} {int(canvas_h)}" style="background:#0f172a;font-family:Segoe UI,Arial,sans-serif;">',
        f'<text x="{margin}" y="38" fill="#f8fafc" font-size="20" font-weight="600">LidUp Planta Métrica 2D (Corte {report.slice_band_m[0]:.2f}–{report.slice_band_m[1]:.2f} m)</text>',
        (
            f'<text x="{margin}" y="62" fill="#94a3b8" font-size="13">'
            f'Dimensões: {report.width_m:.2f} × {report.length_m:.2f} m | '
            f'Área Piso: {report.floor_area_m2:.2f} m² | '
            f'Pé-direito: {report.ceiling_height_m:.2f} m | '
            f'Paredes: {report.wall_area_m2:.2f} m² | '
            f'Planeza Piso RMS: {report.floor_flatness_rms_mm:.1f} mm</text>'
        ),
    ]

    # Grade métrica de 1.0 m
    gx0 = math.floor(min_x)
    gx1 = math.ceil(max_x)
    gy0 = math.floor(min_y)
    gy1 = math.ceil(max_y)
    for gx in range(gx0, gx1 + 1):
        x_px, y_top = to_px(float(gx), max_y)
        _, y_bot = to_px(float(gx), min_y)
        lines.append(f'<line x1="{x_px:.1f}" y1="{y_top:.1f}" x2="{x_px:.1f}" y2="{y_bot:.1f}" stroke="#1e293b" stroke-width="1"/>')
    for gy in range(gy0, gy1 + 1):
        x_left, y_px = to_px(min_x, float(gy))
        x_right, _ = to_px(max_x, float(gy))
        lines.append(f'<line x1="{x_left:.1f}" y1="{y_px:.1f}" x2="{x_right:.1f}" y2="{y_px:.1f}" stroke="#1e293b" stroke-width="1"/>')

    # Contorno envolvente do cômodo
    if len(hull_2d) >= 3:
        pts_str = " ".join(f"{to_px(float(p[0]), float(p[1]))[0]:.1f},{to_px(float(p[0]), float(p[1]))[1]:.1f}" for p in hull_2d)
        lines.append(f'<polygon points="{pts_str}" fill="#1e293b" fill-opacity="0.35" stroke="#475569" stroke-width="1.5" stroke-dasharray="6,4"/>')

    # Pontos da fatia horizontal (amostrados para manter SVG leve)
    if len(slice_2d) > 0:
        step = max(1, len(slice_2d) // 2500)
        for p in slice_2d[::step]:
            px, py = to_px(float(p[0]), float(p[1]))
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.4" fill="#38bdf8" fill-opacity="0.55"/>')

    # Segmentos de parede ajustados + cotas em metros
    for w in walls:
        p1x, p1y = to_px(w.x1, w.y1)
        p2x, p2y = to_px(w.x2, w.y2)
        lines.append(f'<line x1="{p1x:.1f}" y1="{p1y:.1f}" x2="{p2x:.1f}" y2="{p2y:.1f}" stroke="#f8fafc" stroke-width="4" stroke-linecap="round"/>')
        mx, my = 0.5 * (p1x + p2x), 0.5 * (p1y + p2y)
        lines.append(f'<rect x="{mx - 24:.1f}" y="{my - 11:.1f}" width="48" height="18" rx="4" fill="#0f172a" fill-opacity="0.85"/>')
        lines.append(f'<text x="{mx:.1f}" y="{my + 3:.1f}" fill="#fde047" font-size="11" font-weight="600" text-anchor="middle">{w.length_m:.2f}m</text>')

    # Barra de escala de 1.0 m
    sb_x1, sb_y = margin, canvas_h - 26.0
    sb_x2 = sb_x1 + 1.0 * scale
    lines.append(f'<line x1="{sb_x1:.1f}" y1="{sb_y:.1f}" x2="{sb_x2:.1f}" y2="{sb_y:.1f}" stroke="#f8fafc" stroke-width="3"/>')
    lines.append(f'<text x="{0.5 * (sb_x1 + sb_x2):.1f}" y="{sb_y - 7:.1f}" fill="#cbd5e1" font-size="12" text-anchor="middle">1.00 m</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _export_floorplan_dxf(out_path: Path, walls: list[WallSegment], hull_2d: np.ndarray) -> None:
    """Grava arquivo DXF R12 (unidades em metros) compatível com AutoCAD/LibreCAD/QGIS."""
    ent: list[str] = ["0", "SECTION", "2", "HEADER", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]

    for w in walls:
        ent.extend(
            [
                "0", "LINE", "8", "WALLS",
                "10", f"{w.x1:.4f}", "20", f"{w.y1:.4f}", "30", "0.0",
                "11", f"{w.x2:.4f}", "21", f"{w.y2:.4f}", "31", "0.0",
            ]
        )

    if len(hull_2d) >= 3:
        for i in range(len(hull_2d)):
            p1 = hull_2d[i]
            p2 = hull_2d[(i + 1) % len(hull_2d)]
            ent.extend(
                [
                    "0", "LINE", "8", "ROOM_FOOTPRINT",
                    "10", f"{float(p1[0]):.4f}", "20", f"{float(p1[1]):.4f}", "30", "0.0",
                    "11", f"{float(p2[0]):.4f}", "21", f"{float(p2[1]):.4f}", "31", "0.0",
                ]
            )

    ent.extend(["0", "ENDSEC", "0", "EOF"])
    out_path.write_text("\n".join(ent) + "\n", encoding="utf-8")


def _export_floor_flatness_ply(out_path: Path, floor_pts: np.ndarray, h_mm: np.ndarray) -> None:
    """Exporta mapa de calor de planeza do piso: verde <=3mm, amarelo 10mm, vermelho >=25mm."""
    pcd = o3d.geometry.PointCloud()
    if len(floor_pts) == 0:
        o3d.io.write_point_cloud(str(out_path), pcd)
        return
    pcd.points = o3d.utility.Vector3dVector(floor_pts)
    abs_mm = np.abs(h_mm)
    t = np.clip(abs_mm / 25.0, 0.0, 1.0)
    # 0 -> verde (0.15, 0.85, 0.30); 0.4 -> amarelo (0.95, 0.85, 0.15); 1 -> vermelho (0.92, 0.18, 0.18)
    r = np.where(t < 0.4, 0.15 + (t / 0.4) * 0.80, 0.95 - ((t - 0.4) / 0.6) * 0.03)
    g = np.where(t < 0.4, 0.85, 0.85 - ((t - 0.4) / 0.6) * 0.67)
    b = np.where(t < 0.4, 0.30 - (t / 0.4) * 0.15, 0.15 + ((t - 0.4) / 0.6) * 0.03)
    pcd.colors = o3d.utility.Vector3dVector(np.column_stack([r, g, b]))
    o3d.io.write_point_cloud(str(out_path), pcd)


def analyze_room_geometry(
    target: str | Path | np.ndarray | o3d.geometry.Geometry,
    out_dir: str | Path,
    *,
    up_axis: str = "auto",
) -> RoomReport:
    """Extrai métricas arquitetônicas (piso, pé-direito, planeza mm, área m², planta SVG/DXF)."""
    pts, source_label = _extract_points_from_input(target)
    if len(pts) < 50:
        raise ValueError(f"Poucos pontos para análise arquitetônica: {len(pts)}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    axis_name, nominal_up = _choose_up_axis(pts, source_label, up_axis=up_axis)
    aligned, floor_normal, floor_d, floor_mask = _fit_floor_and_align(pts, nominal_up)

    h_floor_mm = aligned[floor_mask, 2] * 1000.0 if int(np.sum(floor_mask)) > 0 else np.zeros(1, dtype=np.float64)
    flat_rms = float(np.sqrt(np.mean(h_floor_mm**2)))
    flat_p95 = float(np.percentile(np.abs(h_floor_mm), 95.0))
    flat_max = float(np.max(np.abs(h_floor_mm)))

    ceiling_h, ceiling_src = _estimate_ceiling_height(aligned)
    aligned_2d, slice_2d, walls, slice_band = _extract_walls_and_align_2d(aligned, ceiling_h)

    hull_src = slice_2d if len(slice_2d) >= 20 else aligned_2d[:, :2]
    hull_2d = _convex_hull_2d(hull_src)
    floor_area, hull_perim = _polygon_area_and_perimeter(hull_2d)

    if len(hull_2d) >= 3:
        width_m = float(np.max(hull_2d[:, 0]) - np.min(hull_2d[:, 0]))
        length_m = float(np.max(hull_2d[:, 1]) - np.min(hull_2d[:, 1]))
    else:
        width_m = float(np.percentile(aligned_2d[:, 0], 99) - np.percentile(aligned_2d[:, 0], 1))
        length_m = float(np.percentile(aligned_2d[:, 1], 99) - np.percentile(aligned_2d[:, 1], 1))

    walls_total_m = float(sum(w.length_m for w in walls))
    wall_area_m2 = float(hull_perim * ceiling_h)

    svg_path = out_path / "floorplan.svg"
    dxf_path = out_path / "floorplan.dxf"
    flat_ply_path = out_path / "floor_flatness.ply"
    report_path = out_path / "room_report.json"

    report = RoomReport(
        source=source_label,
        up_axis=axis_name,
        point_count=int(len(pts)),
        width_m=round(width_m, 3),
        length_m=round(length_m, 3),
        ceiling_height_m=round(ceiling_h, 3),
        ceiling_source=ceiling_src,
        floor_area_m2=round(floor_area, 3),
        perimeter_m=round(hull_perim, 3),
        wall_area_m2=round(wall_area_m2, 3),
        detected_wall_count=len(walls),
        detected_walls_total_m=round(walls_total_m, 3),
        floor_flatness_rms_mm=round(flat_rms, 2),
        floor_flatness_p95_mm=round(flat_p95, 2),
        floor_flatness_max_mm=round(flat_max, 2),
        floor_normal=[round(float(v), 5) for v in floor_normal],
        floor_offset_d=round(float(floor_d), 5),
        slice_band_m=[slice_band[0], slice_band[1]],
        walls=[asdict(w) for w in walls],
        outputs={
            "floorplan_svg": str(svg_path),
            "floorplan_dxf": str(dxf_path),
            "floor_flatness_ply": str(flat_ply_path),
            "room_report_json": str(report_path),
        },
    )

    _export_floorplan_svg(svg_path, slice_2d, walls, hull_2d, report)
    _export_floorplan_dxf(dxf_path, walls, hull_2d)
    _export_floor_flatness_ply(flat_ply_path, pts[floor_mask], h_floor_mm)
    report_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report
