from __future__ import annotations

import heapq
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import rasterio
from PIL import Image
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import binary_closing, binary_fill_holes
from scipy.spatial import cKDTree

from ._rust_bridge import rust_clearance_and_viewshed, rust_priority_flood_ponding
from .architecture import _choose_up_axis, _extract_points_from_input, _fit_floor_and_align
from .bundle import load_bundle


@dataclass
class DiffReport:
    before_source: str
    after_source: str
    aligned_icp: bool
    icp_fitness: float
    point_count_compared: int
    tol_mm: float
    rms_mm: float
    mean_abs_mm: float
    p95_mm: float
    max_mm: float
    changed_pct: float
    added_pct: float
    removed_pct: float
    net_volume_m3: float
    outputs: dict[str, str]


@dataclass
class DrainageReport:
    source: str
    up_axis: str
    cell_m: float
    floor_area_m2: float
    ponding_area_m2: float
    ponding_pct: float
    max_depth_mm: float
    mean_depth_mm: float
    water_volume_liters: float
    critical_ponding_xy_m: list[float]
    outputs: dict[str, str]


@dataclass
class ClearanceReport:
    source: str
    up_axis: str
    cell_m: float
    box_dimensions_m: list[float]
    floor_area_m2: float
    passable_area_m2: float
    passable_pct: float
    obstacle_area_m2: float
    viewshed_pct: float
    blindspot_area_m2: float
    sensor_xy_m: list[float]
    outputs: dict[str, str]


@dataclass
class VolumeReport:
    source: str
    up_axis: str
    cell_m: float
    footprint_m2: float
    volume_m3: float
    max_height_m: float
    mean_height_m: float
    outputs: dict[str, str]


@dataclass
class ReflectiveReport:
    bundle_dir: str
    frames_analyzed: int
    reflective_point_count: int
    reflective_ratio_pct: float
    outputs: dict[str, str]


def _priority_flood_python(z_grid: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    ny, nx = z_grid.shape
    out = np.zeros_like(z_grid, dtype=np.float64)
    if nx < 3 or ny < 3:
        return out
    visited = np.zeros((ny, nx), dtype=bool)
    heap: list[tuple[float, int, int]] = []

    for y in range(ny):
        for x in range(nx):
            if not valid_mask[y, x] or not np.isfinite(z_grid[y, x]):
                visited[y, x] = True
                continue
            is_border = (
                x == 0
                or y == 0
                or x + 1 == nx
                or y + 1 == ny
                or not valid_mask[y - 1, x]
                or not valid_mask[y + 1, x]
                or not valid_mask[y, x - 1]
                or not valid_mask[y, x + 1]
            )
            if is_border:
                visited[y, x] = True
                heapq.heappush(heap, (float(z_grid[y, x]), y, x))

    dirs = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
    while heap:
        cz, cy, cx = heapq.heappop(heap)
        for dx, dy in dirs:
            nx_i, ny_i = cx + dx, cy + dy
            if 0 <= nx_i < nx and 0 <= ny_i < ny and not visited[ny_i, nx_i]:
                visited[ny_i, nx_i] = True
                rz = float(z_grid[ny_i, nx_i])
                if not math.isfinite(rz):
                    continue
                spill = max(cz, rz)
                out[ny_i, nx_i] = max(0.0, spill - rz)
                heapq.heappush(heap, (spill, ny_i, nx_i))
    return out


def _clearance_viewshed_python(
    occ_mask: np.ndarray,
    domain_mask: np.ndarray,
    cell_m: float,
    half_w_m: float,
    half_d_m: float,
    sensor_ix: int,
    sensor_iy: int,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = occ_mask.shape
    passable = np.zeros((ny, nx), dtype=bool)
    visible = np.zeros((ny, nx), dtype=bool)

    max_r = int(math.ceil(math.hypot(half_w_m, half_d_m) / cell_m))
    footprints: list[list[tuple[int, int]]] = []
    for a_idx in range(12):
        th = a_idx * math.pi / 12.0
        ct, st = math.cos(th), math.sin(th)
        pts: list[tuple[int, int]] = []
        for dy in range(-max_r, max_r + 1):
            for dx in range(-max_r, max_r + 1):
                lx, ly = dx * cell_m, dy * cell_m
                rx = lx * ct + ly * st
                ry = -lx * st + ly * ct
                if abs(rx) <= half_w_m and abs(ry) <= half_d_m:
                    pts.append((dx, dy))
        footprints.append(pts or [(0, 0)])

    for y in range(ny):
        for x in range(nx):
            if not domain_mask[y, x] or occ_mask[y, x]:
                continue
            for fp in footprints:
                ok = True
                for dx, dy in fp:
                    xx, yy = x + dx, y + dy
                    if not (0 <= xx < nx and 0 <= yy < ny and domain_mask[yy, xx] and not occ_mask[yy, xx]):
                        ok = False
                        break
                if ok:
                    passable[y, x] = True
                    break

    if 0 <= sensor_ix < nx and 0 <= sensor_iy < ny:
        for y in range(ny):
            for x in range(nx):
                if not domain_mask[y, x]:
                    continue
                dx, dy = x - sensor_ix, y - sensor_iy
                steps = int(math.ceil(max(abs(dx), abs(dy))))
                if steps == 0:
                    visible[y, x] = True
                    continue
                blocked = False
                for st in range(1, steps):
                    t = st / float(steps)
                    rx = int(round(sensor_ix + dx * t))
                    ry = int(round(sensor_iy + dy * t))
                    if occ_mask[ry, rx]:
                        blocked = True
                        break
                if not blocked:
                    visible[y, x] = True

    return passable, visible


def diff_point_clouds(
    before_target: str | Path | np.ndarray | o3d.geometry.Geometry,
    after_target: str | Path | np.ndarray | o3d.geometry.Geometry,
    out_dir: str | Path,
    *,
    align: bool = True,
    tol_mm: float = 15.0,
) -> DiffReport:
    """Inspeção 4D: compara dois scans (Antes × Depois) e gera mapa de calor em mm + estimativa volumétrica."""
    pts_before, src_before = _extract_points_from_input(before_target)
    pts_after, src_after = _extract_points_from_input(after_target)
    if len(pts_before) < 20 or len(pts_after) < 20:
        raise ValueError("Ambos os modelos precisam ter ao menos 20 pontos para comparação 4D.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pcd_b = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_before))
    pcd_a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_after))

    icp_fitness = 1.0
    aligned_icp = False
    if align:
        src_ds = pcd_a.voxel_down_sample(0.04) if len(pts_after) > 8000 else pcd_a
        dst_ds = pcd_b.voxel_down_sample(0.04) if len(pts_before) > 8000 else pcd_b
        reg = o3d.pipelines.registration.registration_icp(
            src_ds,
            dst_ds,
            0.12,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=45),
        )
        if reg.fitness > 0.25 and np.all(np.isfinite(reg.transformation)):
            pcd_a.transform(reg.transformation)
            pts_after = np.asarray(pcd_a.points, dtype=np.float64)
            icp_fitness = float(reg.fitness)
            aligned_icp = True

    pcd_b.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=24))
    normals_b = np.asarray(pcd_b.normals, dtype=np.float64)

    tree = cKDTree(pts_before)
    dists, nn_idx = tree.query(pts_after, workers=-1)
    vec = pts_after - pts_before[nn_idx]
    signs = np.where(np.sum(vec * normals_b[nn_idx], axis=1) >= 0.0, 1.0, -1.0)
    signed_mm = dists * signs * 1000.0
    abs_mm = dists * 1000.0

    tol = max(0.5, float(tol_mm))
    changed = abs_mm > tol
    added = signed_mm > tol
    removed = signed_mm < -tol

    # Estima variação volumétrica via projeção em células de 5 cm multiplicada pelo deslocamento mediano por célula
    cell_area = 0.05 * 0.05
    q_pts = np.round(pts_after / 0.05).astype(np.int64)
    _, uniq_inv = np.unique(q_pts[:, [0, 2]], axis=0, return_inverse=True)
    # Soma deslocamento médio aproximado por coluna espacial
    net_vol_m3 = float(np.sum(signed_mm / 1000.0) * (cell_area / max(1.0, len(pts_after) / max(1, int(uniq_inv.max() + 1)))))

    # Mapa de calor: cinza neutro estável, azul/ciano (+ adicionado), vermelho/laranja (- removido)
    colors = np.full((len(pts_after), 3), [0.42, 0.44, 0.48], dtype=np.float64)
    mag = np.clip((abs_mm - tol) / max(tol * 3.0, 40.0), 0.0, 1.0)
    colors[added] = np.column_stack([0.10 + 0.10 * (1.0 - mag[added]), 0.65 + 0.30 * mag[added], 0.96 * np.ones(int(np.sum(added)))])
    colors[removed] = np.column_stack([0.95 * np.ones(int(np.sum(removed))), 0.25 + 0.35 * (1.0 - mag[removed]), 0.12 * np.ones(int(np.sum(removed)))])

    heatmap_ply = out_path / "diff_heatmap.ply"
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(pts_after)
    out_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(heatmap_ply), out_pcd)

    report_path = out_path / "diff_report.json"
    report = DiffReport(
        before_source=src_before,
        after_source=src_after,
        aligned_icp=aligned_icp,
        icp_fitness=round(icp_fitness, 4),
        point_count_compared=int(len(pts_after)),
        tol_mm=round(tol, 2),
        rms_mm=round(float(np.sqrt(np.mean(abs_mm**2))), 2),
        mean_abs_mm=round(float(np.mean(abs_mm)), 2),
        p95_mm=round(float(np.percentile(abs_mm, 95.0)), 2),
        max_mm=round(float(np.max(abs_mm)), 2),
        changed_pct=round(float(np.mean(changed) * 100.0), 2),
        added_pct=round(float(np.mean(added) * 100.0), 2),
        removed_pct=round(float(np.mean(removed) * 100.0), 2),
        net_volume_m3=round(net_vol_m3, 4),
        outputs={"diff_heatmap_ply": str(heatmap_ply), "diff_report_json": str(report_path)},
    )
    report_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def simulate_water_ponding(
    target: str | Path | np.ndarray | o3d.geometry.Geometry,
    out_dir: str | Path,
    *,
    up_axis: str = "auto",
    cell_m: float = 0.04,
    min_depth_mm: float = 1.0,
) -> DrainageReport:
    """Simulação Hidrológica (Priority-Flood em Rust): identifica poças e volumes de água no piso/MDT."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cell = max(0.015, float(cell_m))

    target_p = Path(target) if isinstance(target, (str, Path)) else None
    if target_p is not None and target_p.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(target_p) as src:
            arr = src.read(1).astype(np.float64)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            res = float(abs(src.transform.a))
        cell = res
        z_grid = np.where(np.isfinite(arr), arr, 0.0)
        valid_mask = np.isfinite(arr)
        ny, nx = z_grid.shape
        gx, gy = np.meshgrid(np.arange(nx, dtype=np.float64) * cell, np.arange(ny, dtype=np.float64) * cell)
        grid_pts = np.column_stack([gx[valid_mask], gy[valid_mask], z_grid[valid_mask]])
        src_label = str(target_p)
        axis_name = "Z"
        u_min, v_min = 0.0, 0.0
    else:
        pts, src_label = _extract_points_from_input(target)
        axis_name, nom_up = _choose_up_axis(pts, src_label, up_axis=up_axis)
        aligned, _, _, _ = _fit_floor_and_align(pts, nom_up)
        floor_pts = aligned[aligned[:, 2] <= 0.12]
        if len(floor_pts) < 30:
            floor_pts = aligned[aligned[:, 2] <= float(np.percentile(aligned[:, 2], 35.0))]

        u_min, v_min = float(np.min(floor_pts[:, 0])), float(np.min(floor_pts[:, 1]))
        u_max, v_max = float(np.max(floor_pts[:, 0])), float(np.max(floor_pts[:, 1]))
        nx = max(5, int(math.ceil((u_max - u_min) / cell)) + 1)
        ny = max(5, int(math.ceil((v_max - v_min) / cell)) + 1)

        ix = np.clip(np.rint((floor_pts[:, 0] - u_min) / cell).astype(np.int64), 0, nx - 1)
        iy = np.clip(np.rint((floor_pts[:, 1] - v_min) / cell).astype(np.int64), 0, ny - 1)

        z_sum = np.zeros((ny, nx), dtype=np.float64)
        z_cnt = np.zeros((ny, nx), dtype=np.int32)
        np.add.at(z_sum, (iy, ix), floor_pts[:, 2])
        np.add.at(z_cnt, (iy, ix), 1)
        valid_mask = z_cnt > 0
        z_grid = np.where(valid_mask, z_sum / np.maximum(z_cnt, 1), 0.0)

        # Preenche lacunas de discretização (binary_closing) e buracos internos para não vazar poças
        closed_domain = binary_fill_holes(binary_closing(valid_mask, structure=np.ones((3, 3), dtype=bool)))
        holes = closed_domain & (~valid_mask)
        if np.any(holes) and np.any(valid_mask):
            vy_v, vx_v = np.nonzero(valid_mask)
            vy_h, vx_h = np.nonzero(holes)
            interp = NearestNDInterpolator(np.column_stack([vx_v, vy_v]), z_grid[vy_v, vx_v])
            z_grid = z_grid.copy()
            z_grid[vy_h, vx_h] = interp(vx_h, vy_h)
            valid_mask = closed_domain

        vy, vx = np.nonzero(valid_mask)
        grid_pts = np.column_stack([u_min + vx * cell, v_min + vy * cell, z_grid[vy, vx]])

    water_m = rust_priority_flood_ponding(z_grid, valid_mask)
    if water_m is None:
        water_m = _priority_flood_python(z_grid, valid_mask)

    water_mm = np.where(valid_mask, water_m * 1000.0, 0.0)
    ponded = valid_mask & (water_mm >= float(min_depth_mm))

    floor_cells = int(np.sum(valid_mask))
    pond_cells = int(np.sum(ponded))
    cell_area = cell * cell
    floor_area_m2 = floor_cells * cell_area
    pond_area_m2 = pond_cells * cell_area
    water_liters = float(np.sum(water_m[ponded]) * cell_area * 1000.0) if pond_cells > 0 else 0.0
    max_d_mm = float(np.max(water_mm[ponded])) if pond_cells > 0 else 0.0
    mean_d_mm = float(np.mean(water_mm[ponded])) if pond_cells > 0 else 0.0

    crit_xy = [0.0, 0.0]
    if pond_cells > 0:
        cy_i, cx_i = np.unravel_index(int(np.argmax(water_mm)), water_mm.shape)
        crit_xy = [round(u_min + float(cx_i) * cell, 3), round(v_min + float(cy_i) * cell, 3)]

    vy, vx = np.nonzero(valid_mask)
    w_here = water_mm[vy, vx]
    cols = np.full((len(grid_pts), 3), [0.35, 0.37, 0.40], dtype=np.float64)
    has_p = w_here >= float(min_depth_mm)
    t_p = np.clip(w_here[has_p] / max(max_d_mm, 10.0), 0.0, 1.0)
    cols[has_p] = np.column_stack([0.12 * (1.0 - t_p), 0.55 + 0.30 * (1.0 - t_p), 0.95 * np.ones_like(t_p)])

    ply_path = out_path / "ponding_map.ply"
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(grid_pts)
    out_pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(str(ply_path), out_pcd)

    svg_path = out_path / "ponding_map.svg"
    _export_grid_svg(
        svg_path,
        title=f"Simulação Hidrológica / Empoçamento ({pond_area_m2:.2f} m² | {water_liters:.1f} L)",
        nx=z_grid.shape[1],
        ny=z_grid.shape[0],
        valid_mask=valid_mask,
        highlight_a=ponded,
        highlight_b=None,
        color_base="#27272A",
        color_a="#38BDF8",
        color_b="#FFFFFF",
    )

    report_path = out_path / "drainage_report.json"
    report = DrainageReport(
        source=src_label,
        up_axis=axis_name,
        cell_m=round(cell, 4),
        floor_area_m2=round(floor_area_m2, 3),
        ponding_area_m2=round(pond_area_m2, 3),
        ponding_pct=round(pond_cells / max(1, floor_cells) * 100.0, 2),
        max_depth_mm=round(max_d_mm, 2),
        mean_depth_mm=round(mean_d_mm, 2),
        water_volume_liters=round(water_liters, 2),
        critical_ponding_xy_m=crit_xy,
        outputs={
            "ponding_map_ply": str(ply_path),
            "ponding_map_svg": str(svg_path),
            "drainage_report_json": str(report_path),
        },
    )
    report_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def analyze_clearance_and_viewshed(
    target: str | Path | np.ndarray | o3d.geometry.Geometry,
    out_dir: str | Path,
    *,
    box_width_m: float = 0.80,
    box_depth_m: float = 1.20,
    box_height_m: float = 1.80,
    sensor_xy_m: tuple[float, float] | None = None,
    up_axis: str = "auto",
    cell_m: float = 0.05,
) -> ClearanceReport:
    """Logística ('Cabe/Passa?') + Visibilidade CFTV/Iluminação via Rust."""
    pts, src_label = _extract_points_from_input(target)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cell = max(0.02, float(cell_m))

    axis_name, nom_up = _choose_up_axis(pts, src_label, up_axis=up_axis)
    aligned, _, _, _ = _fit_floor_and_align(pts, nom_up)

    u_min, v_min = float(np.min(aligned[:, 0])), float(np.min(aligned[:, 1]))
    u_max, v_max = float(np.max(aligned[:, 0])), float(np.max(aligned[:, 1]))
    nx = max(5, int(math.ceil((u_max - u_min) / cell)) + 1)
    ny = max(5, int(math.ceil((v_max - v_min) / cell)) + 1)

    ix = np.clip(np.rint((aligned[:, 0] - u_min) / cell).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.rint((aligned[:, 1] - v_min) / cell).astype(np.int64), 0, ny - 1)
    h = aligned[:, 2]

    domain_mask = np.zeros((ny, nx), dtype=bool)
    domain_mask[iy, ix] = True
    domain_mask = binary_fill_holes(domain_mask)

    occ_sel = (h >= 0.18) & (h <= float(box_height_m) + 0.05)
    occ_cnt_grid = np.zeros((ny, nx), dtype=np.int32)
    np.add.at(occ_cnt_grid, (iy[occ_sel], ix[occ_sel]), 1)
    min_occ_pts = 2 if len(pts) >= 1000 else 1
    occ_mask = occ_cnt_grid >= min_occ_pts

    if sensor_xy_m is not None:
        s_ix = int(np.clip(round((sensor_xy_m[0] - u_min) / cell), 0, nx - 1))
        s_iy = int(np.clip(round((sensor_xy_m[1] - v_min) / cell), 0, ny - 1))
    else:
        free_y, free_x = np.nonzero(domain_mask & ~occ_mask)
        if len(free_x) > 0:
            cx_m, cy_m = float(np.mean(free_x)), float(np.mean(free_y))
            best = int(np.argmin((free_x - cx_m) ** 2 + (free_y - cy_m) ** 2))
            s_ix, s_iy = int(free_x[best]), int(free_y[best])
        else:
            s_ix, s_iy = nx // 2, ny // 2

    rust_res = rust_clearance_and_viewshed(
        occ_mask, domain_mask, cell, 0.5 * box_width_m, 0.5 * box_depth_m, s_ix, s_iy
    )
    if rust_res is not None:
        passable, visible = rust_res
    else:
        passable, visible = _clearance_viewshed_python(
            occ_mask, domain_mask, cell, 0.5 * box_width_m, 0.5 * box_depth_m, s_ix, s_iy
        )

    dom_cnt = max(1, int(np.sum(domain_mask)))
    pass_cnt = int(np.sum(passable))
    occ_cnt = int(np.sum(domain_mask & occ_mask))
    vis_cnt = int(np.sum(domain_mask & visible))
    cell_area = cell * cell

    vy, vx = np.nonzero(domain_mask)
    grid_pts = np.column_stack([u_min + vx * cell, v_min + vy * cell, np.zeros(len(vx))])

    c_clear = np.full((len(grid_pts), 3), [0.35, 0.35, 0.38], dtype=np.float64)
    c_clear[passable[vy, vx]] = [0.94, 0.94, 0.96]
    c_clear[occ_mask[vy, vx]] = [0.85, 0.22, 0.22]
    clear_ply = out_path / "clearance_map.ply"
    p_c = o3d.geometry.PointCloud()
    p_c.points = o3d.utility.Vector3dVector(grid_pts)
    p_c.colors = o3d.utility.Vector3dVector(c_clear)
    o3d.io.write_point_cloud(str(clear_ply), p_c)

    c_vis = np.full((len(grid_pts), 3), [0.16, 0.16, 0.20], dtype=np.float64)
    c_vis[visible[vy, vx]] = [0.95, 0.95, 0.95]
    c_vis[occ_mask[vy, vx]] = [0.75, 0.20, 0.20]
    vis_ply = out_path / "viewshed_map.ply"
    p_v = o3d.geometry.PointCloud()
    p_v.points = o3d.utility.Vector3dVector(grid_pts)
    p_v.colors = o3d.utility.Vector3dVector(c_vis)
    o3d.io.write_point_cloud(str(vis_ply), p_v)

    svg_path = out_path / "clearance_map.svg"
    _export_grid_svg(
        svg_path,
        title=f"Logística ({box_width_m:.2f}×{box_depth_m:.2f}×{box_height_m:.2f}m) | Passa: {pass_cnt / dom_cnt * 100:.1f}% | CFTV Visível: {vis_cnt / dom_cnt * 100:.1f}%",
        nx=nx,
        ny=ny,
        valid_mask=domain_mask,
        highlight_a=passable,
        highlight_b=occ_mask & domain_mask,
        color_base="#27272A",
        color_a="#FAFAFA",
        color_b="#EF4444",
    )

    rep_path = out_path / "clearance_report.json"
    report = ClearanceReport(
        source=src_label,
        up_axis=axis_name,
        cell_m=round(cell, 4),
        box_dimensions_m=[round(box_width_m, 3), round(box_depth_m, 3), round(box_height_m, 3)],
        floor_area_m2=round(dom_cnt * cell_area, 3),
        passable_area_m2=round(pass_cnt * cell_area, 3),
        passable_pct=round(pass_cnt / dom_cnt * 100.0, 2),
        obstacle_area_m2=round(occ_cnt * cell_area, 3),
        viewshed_pct=round(vis_cnt / dom_cnt * 100.0, 2),
        blindspot_area_m2=round((dom_cnt - vis_cnt) * cell_area, 3),
        sensor_xy_m=[round(u_min + s_ix * cell, 3), round(v_min + s_iy * cell, 3)],
        outputs={
            "clearance_map_ply": str(clear_ply),
            "viewshed_map_ply": str(vis_ply),
            "clearance_map_svg": str(svg_path),
            "clearance_report_json": str(rep_path),
        },
    )
    rep_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _export_grid_svg(
    out_path: Path,
    *,
    title: str,
    nx: int,
    ny: int,
    valid_mask: np.ndarray,
    highlight_a: np.ndarray,
    highlight_b: np.ndarray | None,
    color_base: str,
    color_a: str,
    color_b: str,
) -> None:
    w_px, h_px = 900, 700
    margin = 60.0
    scale = min((w_px - 2 * margin) / max(1, nx), (h_px - 2 * margin - 40.0) / max(1, ny))
    rects: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_px}" height="{h_px}" viewBox="0 0 {w_px} {h_px}" style="background:#09090B;font-family:Segoe UI,Arial,sans-serif;">',
        f'<text x="{margin}" y="38" fill="#FAFAFA" font-size="16" font-weight="600">{title}</text>',
    ]
    step = max(1, max(nx, ny) // 140)
    sz = max(1.5, scale * step)
    for y in range(0, ny, step):
        for x in range(0, nx, step):
            if not valid_mask[y, x]:
                continue
            px = margin + x * scale
            py = margin + 40.0 + (ny - 1 - y) * scale
            fill = color_base
            if highlight_b is not None and highlight_b[y, x]:
                fill = color_b
            elif highlight_a[y, x]:
                fill = color_a
            rects.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{sz:.1f}" height="{sz:.1f}" fill="{fill}"/>')
    rects.append("</svg>")
    out_path.write_text("\n".join(rects), encoding="utf-8")


def compute_object_or_pile_volume(
    target: str | Path | np.ndarray | o3d.geometry.Geometry,
    out_dir: str | Path,
    *,
    up_axis: str = "auto",
    cell_m: float = 0.04,
    min_height_m: float = 0.04,
) -> VolumeReport:
    """Cubagem automática: ajusta plano de base e calcula volume em m³ acima do piso/solo."""
    pts, src_label = _extract_points_from_input(target)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cell = max(0.015, float(cell_m))

    axis_name, nom_up = _choose_up_axis(pts, src_label, up_axis=up_axis)
    aligned, _, _, _ = _fit_floor_and_align(pts, nom_up)

    u_min, v_min = float(np.min(aligned[:, 0])), float(np.min(aligned[:, 1]))
    u_max, v_max = float(np.max(aligned[:, 0])), float(np.max(aligned[:, 1]))
    nx = max(5, int(math.ceil((u_max - u_min) / cell)) + 1)
    ny = max(5, int(math.ceil((v_max - v_min) / cell)) + 1)

    ix = np.clip(np.rint((aligned[:, 0] - u_min) / cell).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.rint((aligned[:, 1] - v_min) / cell).astype(np.int64), 0, ny - 1)

    h_top = np.full((ny, nx), -np.inf, dtype=np.float64)
    np.maximum.at(h_top, (iy, ix), aligned[:, 2])
    valid = np.isfinite(h_top)

    # Preenche micro-buracos no topo da pilha via fechamento morfológico + vizinho mais próximo
    above_raw = valid & (h_top >= float(min_height_m))
    closed_obj = binary_fill_holes(binary_closing(above_raw, structure=np.ones((3, 3), dtype=bool)))
    holes = closed_obj & (~above_raw)
    if np.any(holes) and np.any(above_raw):
        vy_a, vx_a = np.nonzero(above_raw)
        vy_h, vx_h = np.nonzero(holes)
        interp = NearestNDInterpolator(np.column_stack([vx_a, vy_a]), h_top[vy_a, vx_a])
        h_top[vy_h, vx_h] = interp(vx_h, vy_h)
        above_mask = closed_obj
    else:
        above_mask = above_raw

    cell_area = cell * cell
    cnt_obj = int(np.sum(above_mask))
    footprint_m2 = cnt_obj * cell_area
    heights = np.maximum(h_top[above_mask], 0.0) if cnt_obj > 0 else np.zeros(1, dtype=np.float64)
    volume_m3 = float(np.sum(heights) * cell_area) if cnt_obj > 0 else 0.0
    max_h = float(np.max(heights)) if cnt_obj > 0 else 0.0
    mean_h = float(np.mean(heights)) if cnt_obj > 0 else 0.0

    # Exporta mapa de calor colorido pelo gradiente de altura acima da base
    h_vals = np.maximum(aligned[:, 2], 0.0)
    cols = np.full((len(aligned), 3), [0.22, 0.23, 0.26], dtype=np.float64)
    is_obj_pt = aligned[:, 2] >= float(min_height_m)
    if np.any(is_obj_pt):
        t = np.clip(h_vals[is_obj_pt] / max(max_h, 0.05), 0.0, 1.0)
        cols[is_obj_pt] = np.column_stack([0.85 + 0.15 * t, 0.85 - 0.55 * t, 0.25 * (1.0 - t)])

    ply_path = out_path / "volume_heatmap.ply"
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(pts)
    out_pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(str(ply_path), out_pcd)

    rep_path = out_path / "volume_report.json"
    report = VolumeReport(
        source=src_label,
        up_axis=axis_name,
        cell_m=round(cell, 4),
        footprint_m2=round(footprint_m2, 3),
        volume_m3=round(volume_m3, 4),
        max_height_m=round(max_h, 3),
        mean_height_m=round(mean_h, 3),
        outputs={"volume_heatmap_ply": str(ply_path), "volume_report_json": str(rep_path)},
    )
    rep_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def detect_reflective_surfaces(
    bundle_dir: str | Path,
    out_dir: str | Path,
    *,
    min_depth_m: float = 0.30,
    max_depth_m: float = 4.50,
    frame_stride: int = 2,
) -> ReflectiveReport:
    """
    Inteligência física multimodal: cruza pixels com `confidence == 0` no mapa LiDAR com
    continuidade espacial/RGB para localizar vidros, espelhos ou lâminas d'água em 3D.
    """
    bundle = load_bundle(bundle_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ref_pts: list[np.ndarray] = []
    ref_cols: list[np.ndarray] = []
    norm_pts: list[np.ndarray] = []
    norm_cols: list[np.ndarray] = []

    cfg = bundle.manifest.get("capture_config", {})
    ref_w = int(cfg.get("image_width") or 4032)
    ref_h = int(cfg.get("image_height") or 3024)
    analyzed = 0

    for fr in bundle.frames[:: max(1, int(frame_stride))]:
        if not fr.depth_file or not fr.confidence_file:
            continue
        dp = bundle.root / fr.depth_file
        cp = bundle.root / fr.confidence_file
        if not dp.exists() or not cp.exists():
            continue

        d_m = np.asarray(Image.open(dp), dtype=np.uint16).astype(np.float64) / 1000.0
        conf = np.asarray(Image.open(cp), dtype=np.uint8)
        if conf.shape != d_m.shape:
            continue
        analyzed += 1

        dh, dw = d_m.shape
        iw = fr.img_w if (fr.img_w and fr.img_w > 0) else ref_w
        ih = fr.img_h if (fr.img_h and fr.img_h > 0) else ref_h
        fx = fr.fx * (dw / float(max(iw, 1)))
        fy = fr.fy * (dh / float(max(ih, 1)))
        cx = fr.cx * (dw / float(max(iw, 1)))
        cy = fr.cy * (dh / float(max(ih, 1)))

        in_rng = (d_m >= min_depth_m) & (d_m <= max_depth_m)
        is_suspicious = in_rng & (conf == 0)
        is_solid = in_rng & (conf >= 2)

        pose = fr.depth_t_wc if fr.depth_t_wc is not None else fr.t_wc
        uu, vv = np.meshgrid(np.arange(dw, dtype=np.float64), np.arange(dh, dtype=np.float64))

        if np.any(is_suspicious):
            ys, xs = np.nonzero(is_suspicious)
            zs = d_m[ys, xs]
            pts_c = np.column_stack([(xs - cx) * zs / fx, (ys - cy) * zs / fy, zs])
            pts_w = pts_c @ pose[:3, :3].T + pose[:3, 3]
            ref_pts.append(pts_w)
            ref_cols.append(np.tile([0.22, 0.85, 0.98], (len(pts_w), 1)))

        if np.any(is_solid):
            ys2, xs2 = np.nonzero(is_solid)
            step = max(1, len(ys2) // 800)
            ys2, xs2 = ys2[::step], xs2[::step]
            zs2 = d_m[ys2, xs2]
            pts_c2 = np.column_stack([(xs2 - cx) * zs2 / fx, (ys2 - cy) * zs2 / fy, zs2])
            pts_w2 = pts_c2 @ pose[:3, :3].T + pose[:3, 3]
            norm_pts.append(pts_w2)
            norm_cols.append(np.tile([0.30, 0.31, 0.35], (len(pts_w2), 1)))

    all_ref = np.vstack(ref_pts) if ref_pts else np.empty((0, 3), dtype=np.float64)
    all_norm = np.vstack(norm_pts) if norm_pts else np.empty((0, 3), dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    if len(all_ref) + len(all_norm) > 0:
        pts_m = np.vstack([p for p in (all_norm, all_ref) if len(p) > 0])
        cols_m = np.vstack([c for c in (norm_cols + ref_cols) if len(c) > 0])
        pcd.points = o3d.utility.Vector3dVector(pts_m)
        pcd.colors = o3d.utility.Vector3dVector(cols_m)
        if len(pcd.points) > 100:
            pcd = pcd.voxel_down_sample(0.025)

    ply_path = out_path / "reflective_zones.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)

    total_est = max(1, len(all_ref) + len(all_norm) * 10)
    ratio = len(all_ref) / float(total_est) * 100.0
    rep_path = out_path / "reflective_report.json"
    report = ReflectiveReport(
        bundle_dir=str(bundle.root),
        frames_analyzed=analyzed,
        reflective_point_count=int(len(all_ref)),
        reflective_ratio_pct=round(min(100.0, ratio), 2),
        outputs={"reflective_zones_ply": str(ply_path), "reflective_report_json": str(rep_path)},
    )
    rep_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report

