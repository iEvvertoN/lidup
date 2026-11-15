from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
import rasterio

from .bundle import load_bundle, reconstruct_lidar_point_cloud
from .earthwork import simulate_plateau_earthwork
from .georef import arkit_to_engineering_local, georeference_points
from .gsplat import export_gaussian_splatting_dataset
from .synthetic import generate_synthetic_bundle
from .terrain import (
    TerrainRaster,
    build_dtm_raster,
    classify_ground_points,
    compute_slope_percent,
    export_geotiff,
    export_las_laz,
    export_ply,
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
    ply_path = export_ply(pts_geo, rgb, out_path=out_dir / "cloud_local.ply", origin=geo.origin_utm)
    dtm = build_dtm_raster(ground_pts, resolution_m=args.dtm_res)
    slope = compute_slope_percent(dtm)

    mdt_path = export_geotiff(dtm.grid, dtm.transform, epsg=geo.epsg, out_path=out_dir / "mdt.tif")
    slope_path = export_geotiff(slope, dtm.transform, epsg=geo.epsg, out_path=out_dir / "slope_pct.tif")
    contours = extract_contours_geojson(dtm, interval_m=args.contour_interval, epsg=geo.epsg)
    contours_path = save_geojson(contours, out_dir / "contours.geojson")
    obj_path = export_terrain_obj(dtm, origin_utm=geo.origin_utm, out_path=out_dir / "terrain_local.obj")

    generated: list[Path] = [laz_path, ply_path, mdt_path, slope_path, contours_path, obj_path]
    if args.with_3dgs:
        gs_info = export_gaussian_splatting_dataset(
            bundle,
            out_dir / "gaussian_splatting",
            min_confidence=min(args.min_confidence, 1),
            max_depth_m=max(args.max_depth, 5.0),
            voxel_size_m=min(args.voxel_size, 0.02),
        )
        generated.append(Path(gs_info["transforms_json"]))

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
    generated.append(report_path)

    print("[lidup] Arquivos gerados:")
    for p in generated:
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


def cmd_export_3dgs(args: argparse.Namespace) -> None:
    info = export_gaussian_splatting_dataset(
        args.bundle,
        args.out,
        min_confidence=args.min_confidence,
        max_depth_m=args.max_depth,
        voxel_size_m=args.voxel_size,
        copy_images=args.copy_images,
    )
    print(f"[lidup] Dataset 3DGS exportado em: {info['out_dir']}")
    print(f"  - Frames/câmeras: {info['frame_count']}")
    print(f"  - Pontos iniciais LiDAR: {info['point_count']} ({info['points3d_ply']})")
    print(f"  - Nerfstudio / Brush: {info['transforms_json']}")
    print(f"  - COLMAP sparse: {info['colmap_sparse_dir']}")


def _resolve_view_target(target: Path) -> Path:
    if target.is_file():
        return target
    if not target.is_dir():
        raise FileNotFoundError(f"Caminho não encontrado: {target}")
    if (target / "manifest.json").exists():
        return target
    for candidate in ("cloud_local.ply", "cloud.ply", "points3D.ply", "cloud.laz", "cloud.las", "terrain_local.obj"):
        p = target / candidate
        if p.exists():
            return p
    raise FileNotFoundError(f"Nenhum pacote lidup ou nuvem (.ply/.laz/.obj) encontrado em {target}")


def load_geometry_for_view(
    target: str | Path,
    *,
    min_confidence: int = 1,
    max_depth_m: float = 5.0,
    voxel_size_m: float = 0.02,
) -> tuple[o3d.geometry.Geometry3D, str]:
    """Carrega uma nuvem (.ply/.laz), malha (.obj) ou bundle bruto `lidup_bundle_v1` para visualização 3D."""
    resolved = _resolve_view_target(Path(target))

    if resolved.is_dir():
        bundle = load_bundle(resolved)
        pcd = reconstruct_lidar_point_cloud(
            bundle,
            min_confidence=min_confidence,
            max_depth_m=max_depth_m,
            voxel_size_m=voxel_size_m,
        )
        pts = np.asarray(pcd.points, dtype=np.float64)
        if len(pts) == 0:
            raise RuntimeError("Nenhum ponto válido encontrado para visualizar neste pacote.")
        pts_eng = arkit_to_engineering_local(pts)
        pts_eng -= pts_eng.mean(axis=0)
        pcd.points = o3d.utility.Vector3dVector(pts_eng)
        return pcd, f"Lidup Bundle ({bundle.manifest['session_id']} — {len(pts_eng)} pts)"

    suffix = resolved.suffix.lower()
    if suffix in {".laz", ".las"}:
        las = laspy.read(str(resolved))
        pts = np.asarray(las.xyz, dtype=np.float64)
        if len(pts) == 0:
            raise RuntimeError(f"Arquivo vazio: {resolved}")
        pts = pts - pts.mean(axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
            rgb = np.vstack([las.red, las.green, las.blue]).T.astype(np.float64) / 65535.0
            pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0))
        return pcd, f"Lidup LAZ ({resolved.name} — {len(pts)} pts)"

    if suffix == ".obj":
        mesh = o3d.io.read_triangle_mesh(str(resolved))
        mesh.compute_vertex_normals()
        return mesh, f"Lidup OBJ ({resolved.name} — {len(mesh.vertices)} vértices)"

    pcd = o3d.io.read_point_cloud(str(resolved))
    pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) == 0:
        raise RuntimeError(f"Nuvem vazia ou formato não suportado: {resolved}")
    if np.max(np.abs(pts)) > 1000.0:
        pcd.points = o3d.utility.Vector3dVector(pts - pts.mean(axis=0))
    return pcd, f"Lidup PLY ({resolved.name} — {len(pts)} pts)"


def cmd_view(args: argparse.Namespace) -> None:
    geom, title = load_geometry_for_view(
        args.target,
        min_confidence=args.min_confidence,
        max_depth_m=args.max_depth,
        voxel_size_m=args.voxel_size,
    )
    print(f"[lidup] Abrindo visualizador 3D: {title}")
    print("[lidup] Controles: Arrastar esquerdo = girar | Scroll = zoom | Shift+Arrastar = mover | +/- = tamanho dos pontos | Q = fechar")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=800)
    vis.add_geometry(geom)
    if args.show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(axes)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.08, 0.09, 0.11])
    opt.point_size = float(args.point_size)
    vis.run()
    vis.destroy_window()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lidup", description="Processador topográfico rural para pacotes do iPhone.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("make-sample", help="Gera pacote sintético lidup_bundle_v1 para testes.")
    p_sample.add_argument("--out", required=True, help="Diretório de saída do bundle.")
    p_sample.add_argument("--length", type=float, default=24.0, help="Comprimento X em metros.")
    p_sample.add_argument("--width", type=float, default=12.0, help="Largura Y em metros.")
    p_sample.set_defaults(func=cmd_make_sample)

    p_proc = sub.add_parser("process", help="Processa um lidup_bundle_v1 em LAZ, PLY, MDT, curvas e OBJ.")
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
    p_proc.add_argument("--with-3dgs", action="store_true", help="Gera também pasta gaussian_splatting (transforms.json + COLMAP).")
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

    p_gs = sub.add_parser("export-3dgs", help="Exporta dataset (transforms.json + COLMAP + points3D.ply) para Gaussian Splatting.")
    p_gs.add_argument("bundle", help="Pasta do pacote lidup_bundle_v1.")
    p_gs.add_argument("--out", required=True, help="Pasta de destino do dataset 3DGS.")
    p_gs.add_argument("--min-confidence", type=int, default=1, choices=[0, 1, 2])
    p_gs.add_argument("--max-depth", type=float, default=6.0)
    p_gs.add_argument("--voxel-size", type=float, default=0.02)
    p_gs.add_argument("--copy-images", action="store_true", help="Copia fotos em vez de criar hardlinks.")
    p_gs.set_defaults(func=cmd_export_3dgs)

    p_view = sub.add_parser("view", help="Abre janela 3D interativa para arquivo .ply/.laz/.obj, pasta de saída ou pacote bruto.")
    p_view.add_argument("target", help="Arquivo (.ply/.laz/.obj), pasta de saída ou pacote lidup_bundle_v1.")
    p_view.add_argument("--min-confidence", type=int, default=1, choices=[0, 1, 2])
    p_view.add_argument("--max-depth", type=float, default=5.0)
    p_view.add_argument("--voxel-size", type=float, default=0.02)
    p_view.add_argument("--point-size", type=float, default=3.5)
    p_view.add_argument("--show-axes", action="store_true", help="Exibe eixos XYZ de referência.")
    p_view.set_defaults(func=cmd_view)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
