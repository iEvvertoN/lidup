from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio

from .bundle import load_bundle, reconstruct_lidar_point_cloud
from .earthwork import simulate_plateau_earthwork
from .georef import georeference_points
from .synthetic import generate_synthetic_bundle
from .terrain import (
    TerrainRaster,
    build_dtm_raster,
    classify_ground_points,
    compute_slope_percent,
    export_geotiff,
    export_las_laz,
    export_terrain_obj,
    extract_contours_geojson,
    save_geojson,
)


def cmd_make_sample(args: argparse.Namespace) -> None:
    out = generate_synthetic_bundle(args.out, length_x_m=args.length, width_y_m=args.width)
    print(f"[lidup] Pacote sintético gerado em: {out}")


def cmd_process(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(args.bundle)
    print(f"[lidup] Pacote carregado: {bundle.manifest['session_id']} ({len(bundle.frames)} frames)")

    pcd = reconstruct_lidar_point_cloud(
        bundle,
        min_confidence=args.min_confidence,
        max_depth_m=args.max_depth,
        voxel_size_m=args.voxel_size,
    )
    pts_arkit = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors) if pcd.has_colors() else None
    if len(pts_arkit) == 0:
        raise RuntimeError("Nenhum ponto válido encontrado após filtragem de confiança/profundidade.")
    print(f"[lidup] Nuvem LiDAR filtrada: {len(pts_arkit)} pontos")

    pts_geo, geo = georeference_points(pts_arkit, bundle, allow_gnss_scale=args.allow_gnss_scale)
    print(f"[lidup] Georreferenciamento: mode={geo.mode} | CRS={geo.crs_name} (EPSG:{geo.epsg}) | scale={geo.scale:.5f}")
    if geo.rmse_total_m is not None:
        print(f"[lidup] RMSE controle: total={geo.rmse_total_m:.3f}m | xyz={geo.rmse_xyz_m}")

    classes = classify_ground_points(pts_geo, cell_size_m=args.ground_cell, max_height_above_ground_m=args.ground_thresh)
    ground_pts = pts_geo[classes == 2]
    print(f"[lidup] Classificação solo: {len(ground_pts)} solo / {len(pts_geo) - len(ground_pts)} não-solo")

    laz_path = export_las_laz(pts_geo, rgb, classes, epsg=geo.epsg, out_path=out_dir / "cloud.laz")
    dtm = build_dtm_raster(ground_pts, resolution_m=args.dtm_res)
    slope = compute_slope_percent(dtm)

    mdt_path = export_geotiff(dtm.grid, dtm.transform, epsg=geo.epsg, out_path=out_dir / "mdt.tif")
    slope_path = export_geotiff(slope, dtm.transform, epsg=geo.epsg, out_path=out_dir / "slope_pct.tif")
    contours = extract_contours_geojson(dtm, interval_m=args.contour_interval, epsg=geo.epsg)
    contours_path = save_geojson(contours, out_dir / "contours.geojson")
    obj_path = export_terrain_obj(dtm, origin_utm=geo.origin_utm, out_path=out_dir / "terrain_local.obj")

    origin_info = {
        "epsg": geo.epsg,
        "crs_name": geo.crs_name,
        "mode": geo.mode,
        "scale": geo.scale,
        "origin_utm": geo.origin_utm.tolist(),
        "rmse_xyz_m": list(geo.rmse_xyz_m) if geo.rmse_xyz_m else None,
        "rmse_total_m": geo.rmse_total_m,
        "control_points_used": geo.control_points_used,
        "point_count_total": int(len(pts_geo)),
        "point_count_ground": int(len(ground_pts)),
        "dtm_bounds_utm": [dtm.min_x, dtm.min_y, dtm.max_x, dtm.max_y],
        "notes": geo.notes,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(origin_info, indent=2), encoding="utf-8")

    print("[lidup] Arquivos gerados:")
    for p in (laz_path, mdt_path, slope_path, contours_path, obj_path, report_path):
        print(f"  - {p}")


def cmd_level(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.mdt) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        transform = src.transform
        epsg = src.crs.to_epsg() if src.crs else 0
        res = float(abs(transform.a))
        bounds = src.bounds

    dtm = TerrainRaster(
        grid=arr,
        transform=transform,
        resolution_m=res,
        min_x=float(bounds.left),
        max_x=float(bounds.right),
        min_y=float(bounds.bottom),
        max_y=float(bounds.top),
    )

    report_json_path = Path(args.mdt).parent / "report.json"
    if report_json_path.exists():
        origin_utm = np.array(json.loads(report_json_path.read_text(encoding="utf-8"))["origin_utm"], dtype=np.float64)
    else:
        valid_z = arr[~np.isnan(arr)]
        origin_utm = np.array([(bounds.left + bounds.right) / 2.0, (bounds.bottom + bounds.top) / 2.0, float(valid_z.mean())])

    # Se usuário passou coordenadas locais (relativas ao origin_utm), soma origin_utm; senão usa UTM direto
    if args.local:
        rx0 = origin_utm[0] + args.min_x
        ry0 = origin_utm[1] + args.min_y
        rx1 = origin_utm[0] + args.max_x
        ry1 = origin_utm[1] + args.max_y
    else:
        rx0, ry0, rx1, ry1 = args.min_x, args.min_y, args.max_x, args.max_y

    graded, report = simulate_plateau_earthwork(
        dtm,
        rect_utm=(rx0, ry0, rx1, ry1),
        target_z_m=args.target_z,
    )

    export_geotiff(graded.grid, graded.transform, epsg=epsg or 0, out_path=out_dir / "mdt_leveled.tif")
    export_terrain_obj(graded, origin_utm=origin_utm, out_path=out_dir / "terrain_leveled_local.obj")
    (out_dir / "earthwork_report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    print(
        f"[lidup] Platô nivelado em Z={report.target_z_m:.3f}m | Área={report.footprint_area_m2:.1f}m² | "
        f"Corte={report.cut_volume_m3:.2f}m³ | Aterro={report.fill_volume_m3:.2f}m³ | Saldo={report.net_volume_m3:.2f}m³"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lidup", description="Processador topográfico rural para pacotes do iPhone.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("make-sample", help="Gera pacote sintético lidup_bundle_v1 para testes.")
    p_sample.add_argument("--out", required=True, help="Diretório de saída do bundle.")
    p_sample.add_argument("--length", type=float, default=24.0, help="Comprimento X em metros.")
    p_sample.add_argument("--width", type=float, default=12.0, help="Largura Y em metros.")
    p_sample.set_defaults(func=cmd_make_sample)

    p_proc = sub.add_parser("process", help="Processa um lidup_bundle_v1 em LAZ, MDT, curvas e OBJ.")
    p_proc.add_argument("bundle", help="Pasta do pacote lidup_bundle_v1.")
    p_proc.add_argument("--out", required=True, help="Pasta de saída dos produtos.")
    p_proc.add_argument("--min-confidence", type=int, default=2, choices=[0, 1, 2], help="Confiança mínima do LiDAR.")
    p_proc.add_argument("--max-depth", type=float, default=4.5, help="Profundidade máxima útil do LiDAR (m).")
    p_proc.add_argument("--voxel-size", type=float, default=0.04, help="Tamanho do voxel de redução (m).")
    p_proc.add_argument("--ground-cell", type=float, default=1.0, help="Célula base do filtro de solo (m).")
    p_proc.add_argument("--ground-thresh", type=float, default=0.25, help="Altura máxima acima do solo (m).")
    p_proc.add_argument("--dtm-res", type=float, default=0.25, help="Resolução do raster MDT (m).")
    p_proc.add_argument("--contour-interval", type=float, default=0.5, help="Equidistância das curvas de nível (m).")
    p_proc.add_argument("--allow-gnss-scale", action="store_true", help="Permite ajustar escala pelo GNSS (usar só com RTK).")
    p_proc.set_defaults(func=cmd_process)

    p_lvl = sub.add_parser("level", help="Simula nivelamento de platô e calcula corte/aterro em m³.")
    p_lvl.add_argument("mdt", help="Caminho para o arquivo mdt.tif gerado.")
    p_lvl.add_argument("--out", required=True, help="Pasta de saída do terreno nivelado.")
    p_lvl.add_argument("--min-x", type=float, required=True)
    p_lvl.add_argument("--min-y", type=float, required=True)
    p_lvl.add_argument("--max-x", type=float, required=True)
    p_lvl.add_argument("--max-y", type=float, required=True)
    p_lvl.add_argument("--local", action="store_true", help="Coordenadas relativas ao origin_utm do report.json.")
    p_lvl.add_argument("--target-z", type=float, default=None, help="Cota fixa desejada (omete para calcular equilíbrio corte=aterro).")
    p_lvl.set_defaults(func=cmd_level)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
