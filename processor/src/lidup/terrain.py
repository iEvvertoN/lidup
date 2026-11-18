from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import open3d as o3d
import rasterio
from pyproj import CRS
from rasterio.transform import Affine
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from shapely.geometry import LineString, mapping

from ._rust_bridge import rust_csf_cloth_filter


@dataclass
class TerrainRaster:
    grid: np.ndarray  # (rows, cols) float64, np.nan fora da área válida
    transform: Affine
    resolution_m: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float


def _bilinear_sample_grid(
    grid: np.ndarray,
    xyz: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    cell_size_m: float,
) -> np.ndarray:
    """Estima cota da superfície na posição exata de cada ponto por interpolação bilinear."""
    ny, nx = grid.shape
    fx = np.clip((xyz[:, 0] - min_x) / cell_size_m, 0.0, float(nx - 1))
    fy = np.clip((xyz[:, 1] - min_y) / cell_size_m, 0.0, float(ny - 1))
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    x1 = np.minimum(x0 + 1, nx - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    wx = fx - x0
    wy = fy - y0
    return (
        grid[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + grid[y0, x1] * wx * (1.0 - wy)
        + grid[y1, x0] * (1.0 - wx) * wy
        + grid[y1, x1] * wx * wy
    )


def _csf_ground_surface(
    xyz: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    nx: int,
    ny: int,
    cell_size_m: float,
    rigidness: int = 2,
    iterations: int = 300,
    gravity_step_m: float = 0.25,
) -> np.ndarray | None:
    """
    Cloth Simulation Filter (Zhang et al., 2016) em NumPy vetorizado:
    inverte a nuvem (-Z) e simula um tecido sob gravidade e rigidez de mola
    para separar solo contínuo de capim alto, árvores e estruturas.
    """
    ix = np.clip(np.rint((xyz[:, 0] - min_x) / cell_size_m).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.rint((xyz[:, 1] - min_y) / cell_size_m).astype(np.int64), 0, ny - 1)
    flat_idx = iy * nx + ix

    # Na nuvem invertida (-Z), o solo real corresponde aos máximos locais.
    z_inv = -xyz[:, 2]
    terrain_inv = np.full(nx * ny, -np.inf, dtype=np.float64)
    np.maximum.at(terrain_inv, flat_idx, z_inv)
    terrain_grid = terrain_inv.reshape((ny, nx))
    valid_mask = np.isfinite(terrain_grid)
    if not np.any(valid_mask):
        return None

    vy, vx = np.nonzero(valid_mask)
    interp_nn = NearestNDInterpolator(np.column_stack([vx, vy]), terrain_grid[vy, vx])
    gy, gx = np.mgrid[0:ny, 0:nx]
    terrain_filled = terrain_grid.copy()
    terrain_filled[~valid_mask] = interp_nn(gx[~valid_mask], gy[~valid_mask])

    rust_dtm = rust_csf_cloth_filter(
        terrain_filled,
        float(z_inv.max()),
        int(rigidness),
        int(iterations),
        float(gravity_step_m),
    )
    if rust_dtm is not None:
        return rust_dtm

    cloth = np.full((ny, nx), float(z_inv.max()) + 0.25, dtype=np.float64)
    passes = max(1, min(3, int(rigidness)))

    for _ in range(max(20, int(iterations))):
        prev = cloth.copy()
        movable = cloth > terrain_filled
        if np.any(movable):
            cloth[movable] -= gravity_step_m
        cloth = np.maximum(cloth, terrain_filled)

        for _ in range(passes):
            padded = np.pad(cloth, 1, mode="edge")
            avg_nbr = 0.25 * (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:])
            pinned = cloth <= (terrain_filled + 1e-6)
            cloth[~pinned] += 0.5 * (avg_nbr[~pinned] - cloth[~pinned])
            cloth = np.maximum(cloth, terrain_filled)

        if float(np.max(np.abs(cloth - prev))) < 1e-4:
            break

    return -cloth


def classify_ground_points(
    xyz: np.ndarray,
    *,
    cell_size_m: float = 1.0,
    max_height_above_ground_m: float = 0.25,
    method: str = "percentile",
    csf_rigidness: int = 2,
    csf_iterations: int = 300,
) -> np.ndarray:
    """
    Classifica pontos segundo padrão ASPRS:
      2 = Ground (solo)
      5 = High Vegetation / Non-ground (vegetação, cercas, construções)
    Métodos disponíveis:
      - `percentile`: grade de mínimos locais (percentil 5%) + filtro mediano
      - `csf`: Cloth Simulation Filter físico (Zhang et al., 2016)
    """
    if len(xyz) == 0:
        return np.empty((0,), dtype=np.uint8)
    if method not in {"percentile", "csf"}:
        raise ValueError(f"Método de classificação de solo inválido: {method}")

    min_x, min_y = float(xyz[:, 0].min()), float(xyz[:, 1].min())
    max_x, max_y = float(xyz[:, 0].max()), float(xyz[:, 1].max())

    nx = max(1, int(np.ceil((max_x - min_x) / cell_size_m)) + 1)
    ny = max(1, int(np.ceil((max_y - min_y) / cell_size_m)) + 1)

    if method == "csf":
        ground_surface = _csf_ground_surface(
            xyz,
            min_x=min_x,
            min_y=min_y,
            nx=nx,
            ny=ny,
            cell_size_m=cell_size_m,
            rigidness=csf_rigidness,
            iterations=csf_iterations,
        )
        if ground_surface is None:
            return np.full(len(xyz), 2, dtype=np.uint8)
    else:
        ix = np.clip(np.floor((xyz[:, 0] - min_x) / cell_size_m).astype(np.int64), 0, nx - 1)
        iy = np.clip(np.floor((xyz[:, 1] - min_y) / cell_size_m).astype(np.int64), 0, ny - 1)
        flat_idx = iy * nx + ix

        cell_z = np.full(nx * ny, np.nan, dtype=np.float64)
        order = np.argsort(flat_idx)
        sorted_cells = flat_idx[order]
        sorted_z = xyz[order, 2]
        unique_cells, starts = np.unique(sorted_cells, return_index=True)
        splits = np.split(sorted_z, starts[1:])
        for cid, zs in zip(unique_cells, splits):
            cell_z[cid] = float(np.percentile(zs, 5))

        grid_z = cell_z.reshape((ny, nx))
        valid_mask = ~np.isnan(grid_z)
        if not np.any(valid_mask):
            return np.full(len(xyz), 2, dtype=np.uint8)

        # Preenche células vazias por vizinho mais próximo antes do filtro mediano
        vy, vx = np.nonzero(valid_mask)
        interp_nn = NearestNDInterpolator(np.column_stack([vx, vy]), grid_z[vy, vx])
        gy, gx = np.mgrid[0:ny, 0:nx]
        filled = interp_nn(gx, gy)
        ground_surface = median_filter(filled, size=3)

    z_est = _bilinear_sample_grid(
        ground_surface,
        xyz,
        min_x=min_x,
        min_y=min_y,
        cell_size_m=cell_size_m,
    )

    is_ground = (xyz[:, 2] - z_est) <= max_height_above_ground_m
    classes = np.where(is_ground, 2, 5).astype(np.uint8)
    return classes


def build_dtm_raster(
    ground_xyz: np.ndarray,
    *,
    resolution_m: float = 0.25,
    max_support_distance_m: float = 2.0,
) -> TerrainRaster:
    """
    Constrói o Modelo Digital de Terreno (MDT) a partir dos pontos classificados como solo.
    Células além de `max_support_distance_m` de qualquer ponto observado recebem NaN.
    """
    if len(ground_xyz) < 3:
        raise ValueError("São necessários pelo menos 3 pontos de solo para gerar o MDT.")

    min_x = float(np.floor(ground_xyz[:, 0].min() / resolution_m) * resolution_m)
    max_x = float(np.ceil(ground_xyz[:, 0].max() / resolution_m) * resolution_m)
    min_y = float(np.floor(ground_xyz[:, 1].min() / resolution_m) * resolution_m)
    max_y = float(np.ceil(ground_xyz[:, 1].max() / resolution_m) * resolution_m)

    cols = max(2, int(round((max_x - min_x) / resolution_m)))
    rows = max(2, int(round((max_y - min_y) / resolution_m)))

    xs = min_x + (np.arange(cols) + 0.5) * resolution_m
    ys = max_y - (np.arange(rows) + 0.5) * resolution_m  # Norte/cima na linha 0 (padrão GeoTIFF)
    gx, gy = np.meshgrid(xs, ys)

    # Se houver muitos pontos, reduz por média em grade para acelerar triangulação Delaunay
    lin_interp = LinearNDInterpolator(ground_xyz[:, :2], ground_xyz[:, 2])
    dtm = lin_interp(gx, gy)

    # Preenche pequenas falhas próximas às bordas sem extrapolar longe do levantamento
    tree = cKDTree(ground_xyz[:, :2])
    query_pts = np.column_stack([gx.ravel(), gy.ravel()])
    dists, _ = tree.query(query_pts, k=1)
    near_mask = dists.reshape((rows, cols)) <= max_support_distance_m

    nan_inside = np.isnan(dtm) & near_mask
    if np.any(nan_inside):
        nn_interp = NearestNDInterpolator(ground_xyz[:, :2], ground_xyz[:, 2])
        dtm[nan_inside] = nn_interp(gx[nan_inside], gy[nan_inside])

    dtm[~near_mask] = np.nan
    transform = Affine.translation(min_x, max_y) @ Affine.scale(resolution_m, -resolution_m)

    return TerrainRaster(
        grid=dtm,
        transform=transform,
        resolution_m=resolution_m,
        min_x=min_x,
        max_x=min_x + cols * resolution_m,
        min_y=max_y - rows * resolution_m,
        max_y=max_y,
    )


def compute_slope_percent(dtm: TerrainRaster) -> np.ndarray:
    """Calcula declividade em porcentagem (%) preservando NaNs."""
    grid = dtm.grid
    valid = ~np.isnan(grid)
    if not np.any(valid):
        return np.full_like(grid, np.nan)

    # Preenche NaNs temporariamente para evitar artefatos numéricos nas bordas
    vy, vx = np.nonzero(valid)
    nn = NearestNDInterpolator(np.column_stack([vx, vy]), grid[vy, vx])
    gy, gx = np.mgrid[0 : grid.shape[0], 0 : grid.shape[1]]
    filled = grid.copy()
    filled[~valid] = nn(gx[~valid], gy[~valid])

    dy, dx = np.gradient(filled, -dtm.resolution_m, dtm.resolution_m)
    slope_pct = np.hypot(dx, dy) * 100.0
    slope_pct[~valid] = np.nan
    return slope_pct


def extract_contours_geojson(
    dtm: TerrainRaster,
    *,
    interval_m: float = 0.5,
    epsg: int = 0,
) -> dict[str, Any]:
    """
    Extrai curvas de nível (marching squares por segmento) diretamente para GeoJSON.
    """
    grid = dtm.grid
    valid_vals = grid[~np.isnan(grid)]
    if len(valid_vals) == 0:
        return {"type": "FeatureCollection", "features": []}

    z_min = math.floor(float(valid_vals.min()) / interval_m) * interval_m
    z_max = math.ceil(float(valid_vals.max()) / interval_m) * interval_m
    levels = np.arange(z_min, z_max + interval_m * 0.5, interval_m)

    rows, cols = grid.shape
    features: list[dict[str, Any]] = []

    # Tabela de arestas de marching squares (top=0, right=1, bottom=2, left=3)
    edge_table: dict[int, list[tuple[int, int]]] = {
        1: [(3, 2)],
        2: [(2, 1)],
        3: [(3, 1)],
        4: [(0, 1)],
        5: [(3, 0), (2, 1)],
        6: [(0, 2)],
        7: [(3, 0)],
        8: [(0, 3)],
        9: [(0, 2)],
        10: [(0, 1), (3, 2)],
        11: [(0, 1)],
        12: [(1, 3)],
        13: [(1, 2)],
        14: [(2, 3)],
    }

    def interp_edge(
        r: int, c: int, edge: int, level: float, v00: float, v10: float, v11: float, v01: float
    ) -> tuple[float, float]:
        # Coordenadas no espaço raster (col, row) dos cantos da célula:
        # tl=(c+0.5, r+0.5), tr=(c+1.5, r+0.5), br=(c+1.5, r+1.5), bl=(c+0.5, r+1.5)
        def frac(a: float, b: float) -> float:
            denom = b - a
            if abs(denom) < 1e-12:
                return 0.5
            return float(np.clip((level - a) / denom, 0.0, 1.0))

        if edge == 0:  # top: tl -> tr
            t = frac(v00, v10)
            px, py = c + 0.5 + t, r + 0.5
        elif edge == 1:  # right: tr -> br
            t = frac(v10, v11)
            px, py = c + 1.5, r + 0.5 + t
        elif edge == 2:  # bottom: bl -> br
            t = frac(v01, v11)
            px, py = c + 0.5 + t, r + 1.5
        else:  # left: tl -> bl
            t = frac(v00, v01)
            px, py = c + 0.5, r + 0.5 + t

        world_x, world_y = dtm.transform @ (px, py)
        return float(world_x), float(world_y)

    for level in levels:
        lvl = float(round(level, 4))
        for r in range(rows - 1):
            for c in range(cols - 1):
                v00 = grid[r, c]      # top-left
                v10 = grid[r, c + 1]  # top-right
                v11 = grid[r + 1, c + 1]  # bottom-right
                v01 = grid[r + 1, c]  # bottom-left
                if np.isnan(v00) or np.isnan(v10) or np.isnan(v11) or np.isnan(v01):
                    continue

                code = (
                    (8 if v00 >= lvl else 0)
                    | (4 if v10 >= lvl else 0)
                    | (2 if v11 >= lvl else 0)
                    | (1 if v01 >= lvl else 0)
                )
                pairs = edge_table.get(code)
                if not pairs:
                    continue

                for e1, e2 in pairs:
                    p1 = interp_edge(r, c, e1, lvl, v00, v10, v11, v01)
                    p2 = interp_edge(r, c, e2, lvl, v00, v10, v11, v01)
                    if p1 != p2:
                        features.append(
                            {
                                "type": "Feature",
                                "properties": {"elevation_m": lvl},
                                "geometry": mapping(LineString([p1, p2])),
                            }
                        )

    geojson: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if epsg > 0:
        geojson["crs"] = {"type": "name", "properties": {"name": f"EPSG:{epsg}"}}
    return geojson


def export_las_laz(
    xyz: np.ndarray,
    rgb: np.ndarray | None,
    classification: np.ndarray | None,
    *,
    epsg: int,
    out_path: str | Path,
) -> Path:
    """Exporta nuvem de pontos em ASPRS LAS/LAZ com CRS gravado no cabeçalho."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = laspy.LasHeader(point_format=2, version="1.4")
    header.offsets = np.min(xyz, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    if epsg > 0:
        header.add_crs(CRS.from_epsg(epsg))

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    if rgb is not None and len(rgb) == len(xyz):
        rgb_u16 = np.clip(rgb * 65535.0, 0, 65535).astype(np.uint16)
        las.red = rgb_u16[:, 0]
        las.green = rgb_u16[:, 1]
        las.blue = rgb_u16[:, 2]

    if classification is not None and len(classification) == len(xyz):
        las.classification = classification.astype(np.uint8)

    las.write(str(path))
    return path


def export_ply(
    xyz: np.ndarray,
    rgb: np.ndarray | None,
    *,
    out_path: str | Path,
    origin: np.ndarray | None = None,
) -> Path:
    """Exporta nuvem de pontos colorida em PLY (centralizada em `origin` se fornecida)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pts = np.asarray(xyz, dtype=np.float64)
    if origin is not None:
        pts = pts - np.asarray(origin, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None and len(rgb) == len(pts):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0))

    o3d.io.write_point_cloud(str(path), pcd)
    return path


def export_geotiff(
    grid: np.ndarray,
    transform: Affine,
    *,
    epsg: int,
    out_path: str | Path,
    nodata: float = -9999.0,
) -> Path:
    """Salva grade float32 como GeoTIFF para abertura direta no QGIS."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out_arr = np.where(np.isnan(grid), nodata, grid).astype(np.float32)
    crs = rasterio.CRS.from_epsg(epsg) if epsg > 0 else None

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=out_arr.shape[0],
        width=out_arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(out_arr, 1)

    return path


def export_terrain_obj(
    dtm: TerrainRaster,
    *,
    origin_utm: np.ndarray,
    out_path: str | Path,
) -> Path:
    """
    Exporta a malha triangular do terreno em Wavefront .OBJ centralizada na origem local
    (`origin_utm`) para abrir no Blender/FreeCAD sem tremor numérico de coordenadas UTM.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grid = dtm.grid
    rows, cols = grid.shape
    ox, oy, oz = float(origin_utm[0]), float(origin_utm[1]), float(origin_utm[2])

    vertex_index = np.full((rows, cols), -1, dtype=np.int64)
    vertices: list[str] = []
    idx = 1

    for r in range(rows):
        for c in range(cols):
            z = grid[r, c]
            if np.isnan(z):
                continue
            wx, wy = dtm.transform @ (c + 0.5, r + 0.5)
            vertices.append(f"v {wx - ox:.4f} {wy - oy:.4f} {z - oz:.4f}\n")
            vertex_index[r, c] = idx
            idx += 1

    faces: list[str] = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v00 = vertex_index[r, c]
            v10 = vertex_index[r, c + 1]
            v11 = vertex_index[r + 1, c + 1]
            v01 = vertex_index[r + 1, c]
            if v00 > 0 and v10 > 0 and v11 > 0:
                faces.append(f"f {v00} {v11} {v10}\n")
            if v00 > 0 and v11 > 0 and v01 > 0:
                faces.append(f"f {v00} {v01} {v11}\n")

    with path.open("w", encoding="utf-8") as f:
        f.write("# lidup MDT mesh (local origin meters, +Z up)\n")
        f.writelines(vertices)
        f.writelines(faces)

    return path


def save_geojson(geojson: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    return path
