from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .terrain import TerrainRaster


@dataclass
class EarthworkReport:
    target_z_m: float
    footprint_area_m2: float
    valid_cells: int
    cut_volume_m3: float
    fill_volume_m3: float
    net_volume_m3: float
    min_existing_z_m: float
    max_existing_z_m: float
    max_cut_depth_m: float
    max_fill_height_m: float


def simulate_plateau_earthwork(
    dtm: TerrainRaster,
    *,
    rect_utm: tuple[float, float, float, float],
    target_z_m: float | None = None,
) -> tuple[TerrainRaster, EarthworkReport]:
    """
    Simula o nivelamento de um platô retangular `(min_x, min_y, max_x, max_y)` em coordenadas
    UTM (ou locais) sobre o MDT. Se `target_z_m` for None, calcula automaticamente a cota de
    equilíbrio (corte == aterro) para minimizar movimentação de terra.
    """
    rx0, ry0, rx1, ry1 = rect_utm
    if rx1 <= rx0 or ry1 <= ry0:
        raise ValueError("Retângulo inválido: max_x > min_x e max_y > min_y são obrigatórios.")

    rows, cols = dtm.grid.shape
    c_idx = np.arange(cols) + 0.5
    r_idx = np.arange(rows) + 0.5
    xs, _ = dtm.transform @ (c_idx, np.zeros_like(c_idx))
    _, ys = dtm.transform @ (np.zeros_like(r_idx), r_idx)

    gx, gy = np.meshgrid(xs, ys)
    inside = (gx >= rx0) & (gx <= rx1) & (gy >= ry0) & (gy <= ry1)
    valid_inside = inside & ~np.isnan(dtm.grid)

    if not np.any(valid_inside):
        raise ValueError("Nenhuma célula válida do MDT dentro do retângulo do platô informado.")

    existing_vals = dtm.grid[valid_inside]
    cell_area = dtm.resolution_m * dtm.resolution_m

    chosen_z = float(np.mean(existing_vals)) if target_z_m is None else float(target_z_m)
    diff = existing_vals - chosen_z  # > 0 => corte; < 0 => aterro

    cut_m3 = float(np.sum(np.maximum(diff, 0.0)) * cell_area)
    fill_m3 = float(np.sum(np.maximum(-diff, 0.0)) * cell_area)

    graded_grid = dtm.grid.copy()
    graded_grid[inside] = chosen_z

    graded_dtm = TerrainRaster(
        grid=graded_grid,
        transform=dtm.transform,
        resolution_m=dtm.resolution_m,
        min_x=dtm.min_x,
        max_x=dtm.max_x,
        min_y=dtm.min_y,
        max_y=dtm.max_y,
    )

    report = EarthworkReport(
        target_z_m=chosen_z,
        footprint_area_m2=float(len(existing_vals) * cell_area),
        valid_cells=int(len(existing_vals)),
        cut_volume_m3=cut_m3,
        fill_volume_m3=fill_m3,
        net_volume_m3=cut_m3 - fill_m3,
        min_existing_z_m=float(existing_vals.min()),
        max_existing_z_m=float(existing_vals.max()),
        max_cut_depth_m=float(np.max(np.maximum(diff, 0.0))),
        max_fill_height_m=float(np.max(np.maximum(-diff, 0.0))),
    )
    return graded_dtm, report
