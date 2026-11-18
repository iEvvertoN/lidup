from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
import rasterio

from .architecture import analyze_room_geometry
from .bundle import load_bundle, reconstruct_lidar_point_cloud, reconstruct_tsdf_details, reconstruct_tsdf_mesh
from .earthwork import simulate_plateau_earthwork
from .georef import arkit_to_engineering_local, georeference_points
from .gsplat import export_gaussian_splatting_dataset
from .synthetic import generate_synthetic_bundle
from .texture import bake_uv_texture_obj
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

    max_inc = None if (args.max_incidence_deg is None or args.max_incidence_deg <= 0 or args.max_incidence_deg >= 90) else float(args.max_incidence_deg)
    tsdf_mesh: o3d.geometry.TriangleMesh | None = None
    tsdf_extracted = []
    tsdf_poses = []
    if args.tsdf:
        upsample = args.depth_upsample if args.depth_upsample is not None else 2
        pcd, tsdf_mesh, tsdf_extracted, tsdf_poses = reconstruct_tsdf_details(
            bundle,
            min_confidence=args.min_confidence,
            max_depth_m=args.max_depth,
            tsdf_voxel_m=args.tsdf_voxel,
            edge_jump_m=args.edge_jump,
            refine_poses=(args.pose_refine != "none") if args.pose_refine else True,
            pose_refine=args.pose_refine,
            max_incidence_deg=max_inc,
            hires_color=True,
            depth_upsample=upsample,
            focus_radius_m=args.tsdf_focus_radius,
            use_mvs=args.mvs,
            photometric_opt=(args.photometric_opt or args.mvs),
        )
    else:
        upsample = args.depth_upsample if args.depth_upsample is not None else 1
        pcd = reconstruct_lidar_point_cloud(
            bundle,
            min_confidence=args.min_confidence,
            max_depth_m=args.max_depth,
            voxel_size_m=args.voxel_size,
            edge_jump_m=args.edge_jump,
            refine_poses=args.refine_poses,
            pose_refine=args.pose_refine,
            max_incidence_deg=max_inc,
            physics_weight=not args.no_physics_weight,
            hires_color=args.hires_color,
            depth_upsample=upsample,
        )
    pts_arkit = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors) if pcd.has_colors() else None
    if len(pts_arkit) == 0:
        raise RuntimeError("Nenhum ponto válido encontrado após filtragem de confiança/profundidade.")
    print(f"[lidup] Nuvem LiDAR filtrada: {len(pts_arkit)} pontos")

    pts_geo, geo = georeference_points(
        pts_arkit,
        bundle,
        allow_gnss_scale=args.allow_gnss_scale,
        lock_gravity=(args.georef_alignment != "free_3d"),
        force_local=(args.georef_alignment == "local"),
    )
    print(f"[lidup] Georreferenciamento: mode={geo.mode} | CRS={geo.crs_name} (EPSG:{geo.epsg}) | scale={geo.scale:.5f}")
    if geo.rmse_total_m is not None:
        print(f"[lidup] RMSE controle: total={geo.rmse_total_m:.3f}m | xyz={geo.rmse_xyz_m}")

    classes = classify_ground_points(
        pts_geo,
        cell_size_m=args.ground_cell,
        max_height_above_ground_m=args.ground_thresh,
        method=args.ground_method,
    )
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
    if tsdf_mesh is not None and len(tsdf_mesh.vertices) > 0:
        mesh_arkit_copy = o3d.geometry.TriangleMesh(tsdf_mesh) if args.uv_texture else None
        verts_arkit = np.asarray(tsdf_mesh.vertices, dtype=np.float64)
        verts_local = (verts_arkit * geo.scale) @ geo.rotation.T + geo.translation - geo.origin_utm
        tsdf_mesh.vertices = o3d.utility.Vector3dVector(verts_local)
        tsdf_mesh.compute_vertex_normals()
        mesh_path = out_dir / "object_mesh_local.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), tsdf_mesh)
        generated.append(mesh_path)
        if args.uv_texture and mesh_arkit_copy is not None and tsdf_extracted:
            uv_obj = bake_uv_texture_obj(
                bundle,
                mesh_arkit_copy,
                verts_local,
                tsdf_extracted,
                tsdf_poses,
                out_dir / "object_textured.obj",
                max_depth_m=args.max_depth,
            )
            generated.append(uv_obj)
            generated.append(uv_obj.with_suffix(".jpg"))
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
    for candidate in ("object_textured.obj", "object_mesh_local.ply", "cloud_local.ply", "cloud.ply", "points3D.ply", "cloud.laz", "cloud.las", "terrain_local.obj"):
        p = target / candidate
        if p.exists():
            return p
    raise FileNotFoundError(f"Nenhum pacote lidup ou nuvem (.ply/.laz/.obj) encontrado em {target}")


def load_geometry_for_view(
    target: str | Path,
    *,
    min_confidence: int = 2,
    max_depth_m: float = 4.5,
    voxel_size_m: float = 0.02,
    edge_jump_m: float = 0.12,
    refine_poses: bool = False,
    pose_refine: str | None = None,
    max_incidence_deg: float | None = 75.0,
    physics_weight: bool = True,
    use_tsdf: bool = False,
    tsdf_voxel_m: float = 0.008,
    hires_color: bool = False,
    depth_upsample: int | None = None,
    tsdf_focus_radius_m: float | None = None,
    use_mvs: bool = False,
    photometric_opt: bool = False,
) -> tuple[o3d.geometry.Geometry3D, str]:
    """Carrega uma nuvem (.ply/.laz), malha (.ply/.obj) ou bundle bruto `lidup_bundle_v1` para visualização 3D."""
    resolved = _resolve_view_target(Path(target))

    if resolved.is_dir():
        bundle = load_bundle(resolved)
        if use_tsdf:
            upsample = depth_upsample if depth_upsample is not None else 2
            _, mesh = reconstruct_tsdf_mesh(
                bundle,
                min_confidence=min_confidence,
                max_depth_m=max_depth_m,
                tsdf_voxel_m=tsdf_voxel_m,
                edge_jump_m=edge_jump_m,
                refine_poses=refine_poses or (pose_refine not in {None, "none"}),
                pose_refine=pose_refine,
                max_incidence_deg=max_incidence_deg,
                hires_color=True,
                depth_upsample=upsample,
                focus_radius_m=tsdf_focus_radius_m,
                use_mvs=use_mvs,
                photometric_opt=(photometric_opt or use_mvs),
            )
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            if len(verts) == 0:
                raise RuntimeError("Nenhum vértice reconstruído via TSDF neste pacote.")
            verts_eng = arkit_to_engineering_local(verts)
            verts_eng -= verts_eng.mean(axis=0)
            mesh.vertices = o3d.utility.Vector3dVector(verts_eng)
            mesh.compute_vertex_normals()
            return mesh, f"Lidup TSDF ({bundle.manifest['session_id']} — {len(verts_eng)} vértices)"
        upsample = depth_upsample if depth_upsample is not None else 1
        pcd = reconstruct_lidar_point_cloud(
            bundle,
            min_confidence=min_confidence,
            max_depth_m=max_depth_m,
            voxel_size_m=voxel_size_m,
            edge_jump_m=edge_jump_m,
            refine_poses=refine_poses,
            pose_refine=pose_refine,
            max_incidence_deg=max_incidence_deg,
            physics_weight=physics_weight,
            hires_color=hires_color,
            depth_upsample=upsample,
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
        mesh = o3d.io.read_triangle_mesh(str(resolved), enable_post_processing=True)
        v = np.asarray(mesh.vertices, dtype=np.float64)
        if len(v) > 0 and np.max(np.abs(v)) > 1000.0:
            mesh.vertices = o3d.utility.Vector3dVector(v - v.mean(axis=0))
        mesh.compute_vertex_normals()
        tag = "OBJ-UV" if mesh.has_textures() else "OBJ"
        return mesh, f"Lidup {tag} ({resolved.name} — {len(mesh.vertices)} vértices)"

    if suffix == ".ply":
        mesh_try = o3d.io.read_triangle_mesh(str(resolved))
        if len(mesh_try.triangles) > 0 and len(mesh_try.vertices) > 0:
            v = np.asarray(mesh_try.vertices, dtype=np.float64)
            if np.max(np.abs(v)) > 1000.0:
                mesh_try.vertices = o3d.utility.Vector3dVector(v - v.mean(axis=0))
            mesh_try.compute_vertex_normals()
            return mesh_try, f"Lidup Mesh ({resolved.name} — {len(v)} vértices)"

    pcd = o3d.io.read_point_cloud(str(resolved))
    pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) == 0:
        raise RuntimeError(f"Nuvem vazia ou formato não suportado: {resolved}")
    if np.max(np.abs(pts)) > 1000.0:
        pcd.points = o3d.utility.Vector3dVector(pts - pts.mean(axis=0))
    return pcd, f"Lidup PLY ({resolved.name} — {len(pts)} pts)"


def _elevation_colors(pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    z = pts[:, 2]
    z_lo, z_hi = float(np.percentile(z, 2.0)), float(np.percentile(z, 98.0))
    t = np.clip((z - z_lo) / max(z_hi - z_lo, 1e-4), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return np.column_stack([r, g, b])


def configure_view_controls(
    vis: o3d.visualization.VisualizerWithKeyCallback,
    geom: o3d.geometry.Geometry3D,
    *,
    start_no_color: bool = False,
) -> dict[str, Any]:
    """Registra atalhos de teclado para aproximar, distanciar, mover, resetar câmera e alternar cor/relevo."""
    bbox = geom.get_axis_aligned_bounding_box()
    max_extent = float(np.max(bbox.get_extent())) if not bbox.is_empty() else 2.0
    move_step = max(0.05, max_extent * 0.06)

    is_mesh = isinstance(geom, o3d.geometry.TriangleMesh)
    pts_ref = np.asarray(geom.vertices if is_mesh else geom.points, dtype=np.float64)
    orig_textures = list(geom.textures) if (is_mesh and geom.has_textures()) else []
    orig_cols = np.asarray(geom.vertex_colors if is_mesh else geom.colors, dtype=np.float64).copy()
    if len(orig_cols) != len(pts_ref):
        orig_cols = np.full((len(pts_ref), 3), 0.72, dtype=np.float64)
    elev_cols = _elevation_colors(pts_ref)
    gray_cols = np.full((len(pts_ref), 3), 0.74, dtype=np.float64)
    palettes = [orig_cols, gray_cols, elev_cols]
    color_state = {"mode": 1 if start_no_color else 0}

    def _apply_palette(mode_idx: int) -> None:
        if is_mesh and orig_textures:
            geom.textures = orig_textures if mode_idx == 0 else []
        vec = o3d.utility.Vector3dVector(palettes[mode_idx])
        if is_mesh:
            geom.vertex_colors = vec
        else:
            geom.colors = vec

    if start_no_color:
        _apply_palette(1)

    def _toggle_color(v: o3d.visualization.Visualizer) -> bool:
        color_state["mode"] = (color_state["mode"] + 1) % 3
        _apply_palette(color_state["mode"])
        v.update_geometry(geom)
        return True

    def _zoom_in(v: o3d.visualization.Visualizer) -> bool:
        vc = v.get_view_control()
        vc.scale(-3.0)
        vc.camera_local_translate(move_step * 0.5, 0.0, 0.0)
        return True

    def _zoom_out(v: o3d.visualization.Visualizer) -> bool:
        vc = v.get_view_control()
        vc.scale(3.0)
        vc.camera_local_translate(-move_step * 0.5, 0.0, 0.0)
        return True

    def _move_forward(v: o3d.visualization.Visualizer) -> bool:
        v.get_view_control().camera_local_translate(move_step, 0.0, 0.0)
        return True

    def _move_backward(v: o3d.visualization.Visualizer) -> bool:
        v.get_view_control().camera_local_translate(-move_step, 0.0, 0.0)
        return True

    def _move_left(v: o3d.visualization.Visualizer) -> bool:
        v.get_view_control().camera_local_translate(0.0, -move_step, 0.0)
        return True

    def _move_right(v: o3d.visualization.Visualizer) -> bool:
        v.get_view_control().camera_local_translate(0.0, move_step, 0.0)
        return True

    def _reset_view(v: o3d.visualization.Visualizer) -> bool:
        v.reset_view_point(True)
        return True

    # Códigos GLFW: Setas (262..265), PageUp/PageDown (266..267)
    callbacks = {
        ord("W"): _zoom_in,
        265: _zoom_in,       # Seta para cima
        ord("S"): _zoom_out,
        264: _zoom_out,      # Seta para baixo
        ord("I"): _move_forward,
        266: _move_forward,  # PageUp
        ord("O"): _move_backward,
        267: _move_backward, # PageDown
        ord("A"): _move_left,
        263: _move_left,     # Seta esquerda
        ord("D"): _move_right,
        262: _move_right,    # Seta direita
        ord("R"): _reset_view,
        ord("C"): _toggle_color,
    }
    for key_code, fn in callbacks.items():
        vis.register_key_callback(key_code, fn)
    return {"move_step": move_step, "callbacks": callbacks}


def compute_picked_measurements(
    points: np.ndarray,
    picked_indices: list[int],
    *,
    up_axis: str = "y",
) -> list[dict[str, float | int]]:
    """Calcula distâncias 3D, deltas ortogonais e declividade (%) entre pontos consecutivos marcados."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return []
    valid = [int(i) for i in picked_indices if 0 <= int(i) < len(pts)]
    axis_idx = 2 if up_axis.strip().lower() == "z" else 1
    horiz_axes = [i for i in range(3) if i != axis_idx]

    segments: list[dict[str, float | int]] = []
    for idx_a, idx_b in zip(valid, valid[1:]):
        p1 = pts[idx_a]
        p2 = pts[idx_b]
        delta = p2 - p1
        dist_3d = float(np.linalg.norm(delta))
        dist_horiz = float(math.hypot(float(delta[horiz_axes[0]]), float(delta[horiz_axes[1]])))
        dz_up = float(delta[axis_idx])
        slope_pct = (abs(dz_up) / dist_horiz * 100.0) if dist_horiz > 1e-6 else 0.0
        segments.append(
            {
                "from_index": idx_a,
                "to_index": idx_b,
                "dist_3d_m": round(dist_3d, 4),
                "dist_horiz_m": round(dist_horiz, 4),
                "dx_m": round(float(delta[0]), 4),
                "dy_m": round(float(delta[1]), 4),
                "dz_m": round(float(delta[2]), 4),
                "dv_up_m": round(dz_up, 4),
                "slope_pct": round(slope_pct, 2),
            }
        )
    return segments


def cmd_analyze_room(args: argparse.Namespace) -> None:
    report = analyze_room_geometry(args.target, args.out, up_axis=args.up_axis)
    print(f"[lidup] Análise arquitetônica salva em: {args.out}")
    print(
        f"[lidup] Cômodo: {report.width_m:.2f} × {report.length_m:.2f} m | "
        f"Área piso: {report.floor_area_m2:.2f} m² | "
        f"Pé-direito ({report.ceiling_source}): {report.ceiling_height_m:.2f} m | "
        f"Planeza piso RMS: {report.floor_flatness_rms_mm:.1f} mm (P95: {report.floor_flatness_p95_mm:.1f} mm)"
    )


def cmd_view(args: argparse.Namespace) -> None:
    max_inc = None if (args.max_incidence_deg is None or args.max_incidence_deg <= 0 or args.max_incidence_deg >= 90) else float(args.max_incidence_deg)
    geom, title = load_geometry_for_view(
        args.target,
        min_confidence=args.min_confidence,
        max_depth_m=args.max_depth,
        voxel_size_m=args.voxel_size,
        edge_jump_m=args.edge_jump,
        refine_poses=args.refine_poses,
        pose_refine=args.pose_refine,
        max_incidence_deg=max_inc,
        physics_weight=not args.no_physics_weight,
        use_tsdf=args.tsdf,
        tsdf_voxel_m=args.tsdf_voxel,
        hires_color=(args.hires_color or args.tsdf),
        depth_upsample=args.depth_upsample,
        tsdf_focus_radius_m=args.tsdf_focus_radius,
        use_mvs=args.mvs,
        photometric_opt=args.photometric_opt,
    )
    if args.measure:
        pick_geom = geom
        if isinstance(geom, o3d.geometry.TriangleMesh):
            pick_geom = o3d.geometry.PointCloud(geom.vertices)
            if geom.has_vertex_colors():
                pick_geom.colors = geom.vertex_colors
            else:
                pick_geom.paint_uniform_color([0.75, 0.78, 0.82])
        print(
            "[lidup] Modo Régua 3D ativo: segure SHIFT + clique esquerdo para marcar pontos (Shift+Dir desfaz) e pressione Q para calcular as distâncias."
        )
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=f"{title} [Régua 3D: Shift+Clique | Q fecha]", width=1280, height=800)
        vis.add_geometry(pick_geom)
        if args.show_axes:
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.08, 0.09, 0.11])
        opt.point_size = float(args.point_size)
        vis.run()
        picked = list(vis.get_picked_points())
        vis.destroy_window()

        up_hint = "z" if "georef" in str(args.target).lower() else "y"
        pts_arr = np.asarray(pick_geom.points)
        segs = compute_picked_measurements(pts_arr, picked, up_axis=up_hint)
        if not picked:
            print("[lidup] Nenhum ponto selecionado.")
        elif len(picked) == 1:
            p0 = pts_arr[picked[0]]
            print(f"[lidup] 1 ponto selecionado (#{picked[0]}): ({p0[0]:.3f}, {p0[1]:.3f}, {p0[2]:.3f}) m — marque ao menos 2 pontos.")
        else:
            total_m = sum(float(s["dist_3d_m"]) for s in segs)
            for i, s in enumerate(segs, start=1):
                print(
                    f"[lidup] Trecho {i} (#{s['from_index']} -> #{s['to_index']}): "
                    f"3D = {s['dist_3d_m']:.3f} m ({s['dist_3d_m'] * 100.0:.1f} cm) | "
                    f"Horiz = {s['dist_horiz_m']:.3f} m | dV = {s['dv_up_m']:+.3f} m | Incl = {s['slope_pct']:.1f}% | "
                    f"(dX={s['dx_m']:+.3f}, dY={s['dy_m']:+.3f}, dZ={s['dz_m']:+.3f})"
                )
            print(f"[lidup] Total acumulado ({len(segs)} trechos): {total_m:.3f} m")
        return

    print(f"[lidup] Abrindo visualizador 3D: {title}")
    print(
        "[lidup] Controles: Arrastar esquerdo = girar | Scroll ou W/↑ = aproximar | S/↓ = distanciar | "
        "I/O ou PgUp/PgDn = entrar/recuar | A/D = mover lados | C = alternar Cor/Cinza/Altura | R = resetar | +/- = tamanho | Q = fechar"
    )

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=1280, height=800)
    vis.add_geometry(geom)
    if args.show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(axes)

    configure_view_controls(vis, geom, start_no_color=args.no_color)
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
    p_proc.add_argument("--edge-jump", type=float, default=0.12, help="Remove saltos de profundidade em bordas acima deste limiar (m).")
    p_proc.add_argument("--max-incidence-deg", type=float, default=75.0, help="Remove raios LiDAR em ângulo rasante acima deste limite em graus (90 = desativado).")
    p_proc.add_argument("--no-physics-weight", action="store_true", help="Desativa ponderação cos(theta)/z^2 na redução por voxel.")
    p_proc.add_argument("--refine-poses", action="store_true", help="Refina poses entre frames via PoseGraph + Loop Closure (use --pose-refine para trocar o método).")
    p_proc.add_argument("--pose-refine", default=None, choices=["none", "posegraph", "colored_icp", "legacy_icp"], help="Método de alinhamento entre frames (padrão: posegraph quando --refine-poses ativo).")
    p_proc.add_argument("--tsdf", action="store_true", help="Gera malha 3D sólida contínua (UniformTSDFVolume) com textura 12 MP (ideal para objetos/interiores).")
    p_proc.add_argument("--tsdf-voxel", type=float, default=0.008, help="Tamanho do voxel da malha TSDF em metros (padrão: 0.008 = 8 mm).")
    p_proc.add_argument("--tsdf-focus-radius", type=float, default=None, help="Raio (m) ao redor do centro de convergência das câmeras 360° para concentrar o voxel fino no objeto.")
    p_proc.add_argument("--depth-upsample", type=int, default=None, choices=[1, 2, 3], help="Super-resolução de profundidade guiada por bordas RGB (padrão: 2 com --tsdf, 1 na nuvem esparsa).")
    p_proc.add_argument("--mvs", action="store_true", help="Ativa refinamento estéreo multi-visão guiado por LiDAR (micro-relevos via ZNCC) + otimização fotométrica.")
    p_proc.add_argument("--photometric-opt", action="store_true", help="Ativa otimização fotométrica de câmera/cor (Open3D ColorMapOptimizer) sobre a malha.")
    p_proc.add_argument("--uv-texture", action="store_true", help="Gera object_textured.obj + .mtl + atlas JPEG 12 MP (usar com --tsdf).")
    p_proc.add_argument("--hires-color", action="store_true", help="Projeta cores diretamente das fotos 12 MP em vez de 256x192.")
    p_proc.add_argument("--georef-alignment", default="gravity_locked", choices=["gravity_locked", "free_3d", "local"], help="Alinhamento GNSS: gravidade travada (padrão), 3D livre legado ou local.")
    p_proc.add_argument("--ground-method", default="csf", choices=["csf", "percentile"], help="Método de classificação de solo: CSF físico (padrão) ou percentil.")
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
    p_view.add_argument("--min-confidence", type=int, default=2, choices=[0, 1, 2])
    p_view.add_argument("--max-depth", type=float, default=4.5)
    p_view.add_argument("--voxel-size", type=float, default=0.02)
    p_view.add_argument("--edge-jump", type=float, default=0.12)
    p_view.add_argument("--max-incidence-deg", type=float, default=75.0)
    p_view.add_argument("--no-physics-weight", action="store_true")
    p_view.add_argument("--refine-poses", action="store_true", help="Realinha frames via PoseGraph antes de exibir pacote bruto.")
    p_view.add_argument("--pose-refine", default=None, choices=["none", "posegraph", "colored_icp", "legacy_icp"])
    p_view.add_argument("--tsdf", action="store_true", help="Reconstrói malha TSDF sólida com cor 12 MP ao abrir pacote bruto.")
    p_view.add_argument("--tsdf-voxel", type=float, default=0.008)
    p_view.add_argument("--tsdf-focus-radius", type=float, default=None)
    p_view.add_argument("--depth-upsample", type=int, default=None, choices=[1, 2, 3])
    p_view.add_argument("--mvs", action="store_true")
    p_view.add_argument("--photometric-opt", action="store_true")
    p_view.add_argument("--hires-color", action="store_true")
    p_view.add_argument("--no-color", action="store_true", help="Abre em cinza neutro (pressione C na janela para alternar Cor/Cinza/Altura).")
    p_view.add_argument("--measure", action="store_true", help="Modo régua 3D interativa (Shift+Clique nos pontos, Q fecha e imprime medidas).")
    p_view.add_argument("--point-size", type=float, default=3.5)
    p_view.add_argument("--show-axes", action="store_true", help="Exibe eixos XYZ de referência.")
    p_view.set_defaults(func=cmd_view)

    p_room = sub.add_parser("analyze-room", help="Extrai planta baixa 2D (SVG/DXF), pé-direito, área m² e planeza do piso em mm.")
    p_room.add_argument("target", help="Arquivo (.ply/.obj/.las/.laz) ou pasta de saída processada.")
    p_room.add_argument("--out", required=True, help="Pasta de saída para floorplan.svg, floorplan.dxf, floor_flatness.ply e room_report.json.")
    p_room.add_argument("--up-axis", default="auto", choices=["auto", "y", "z"], help="Eixo vertical UP (auto, y para nuvem local ARKit, z para georef/ENU).")
    p_room.set_defaults(func=cmd_analyze_room)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
