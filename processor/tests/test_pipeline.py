from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lidup.bundle import load_bundle, reconstruct_lidar_point_cloud
from lidup.cli import configure_view_controls, load_geometry_for_view
from lidup.earthwork import simulate_plateau_earthwork
from lidup.georef import georeference_points, sirgas2000_utm_epsg, umeyama_alignment
from lidup.gsplat import export_gaussian_splatting_dataset
from lidup.synthetic import generate_synthetic_bundle
from lidup.terrain import (
    build_dtm_raster,
    classify_ground_points,
    compute_slope_percent,
    export_geotiff,
    export_las_laz,
    export_ply,
    export_terrain_obj,
    extract_contours_geojson,
)


def test_sirgas2000_utm_epsg_brazil() -> None:
    epsg, name = sirgas2000_utm_epsg(-22.725, -47.649)
    assert epsg == 31983
    assert "23S" in name


def test_umeyama_recovers_rigid_transform() -> None:
    rng = np.random.default_rng(7)
    src = rng.uniform(-10.0, 10.0, size=(20, 3))
    theta = 0.45
    r_true = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t_true = np.array([230000.0, 7485000.0, 540.0])
    dst = src @ r_true.T + t_true

    scale, rot, trans = umeyama_alignment(src, dst, allow_scale=False)
    assert np.isclose(scale, 1.0)
    assert np.allclose(rot, r_true, atol=1e-6)
    assert np.allclose(trans, t_true, atol=1e-6)


def test_end_to_end_synthetic_bundle_and_earthwork(tmp_path: Path) -> None:
    bundle_dir = generate_synthetic_bundle(tmp_path / "sample_bundle", length_x_m=20.0, width_y_m=10.0)
    bundle = load_bundle(bundle_dir)
    assert len(bundle.frames) == 28

    pcd = reconstruct_lidar_point_cloud(bundle, min_confidence=2, max_depth_m=4.5, voxel_size_m=0.08)
    pts_arkit = np.asarray(pcd.points)
    assert len(pts_arkit) > 200

    pts_geo, geo = georeference_points(pts_arkit, bundle)
    assert geo.epsg == 31983
    assert geo.mode == "stationary_gcps"
    assert geo.rmse_total_m is not None and geo.rmse_total_m < 0.5

    classes = classify_ground_points(pts_geo, cell_size_m=1.0, max_height_above_ground_m=0.25)
    ground_pts = pts_geo[classes == 2]
    non_ground_pts = pts_geo[classes != 2]
    assert len(ground_pts) > 100
    assert len(non_ground_pts) > 0  # obstáculos sintéticos devem ser separados do solo

    dtm = build_dtm_raster(ground_pts, resolution_m=0.5)
    slope = compute_slope_percent(dtm)
    assert np.nanmean(slope) > 0.0

    contours = extract_contours_geojson(dtm, interval_m=0.25, epsg=geo.epsg)
    assert len(contours["features"]) > 0

    laz_path = export_las_laz(pts_geo, np.asarray(pcd.colors), classes, epsg=geo.epsg, out_path=tmp_path / "out.laz")
    ply_path = export_ply(pts_geo, np.asarray(pcd.colors), out_path=tmp_path / "cloud_local.ply", origin=geo.origin_utm)
    tif_path = export_geotiff(dtm.grid, dtm.transform, epsg=geo.epsg, out_path=tmp_path / "mdt.tif")
    obj_path = export_terrain_obj(dtm, origin_utm=geo.origin_utm, out_path=tmp_path / "terrain.obj")
    assert laz_path.exists() and laz_path.stat().st_size > 0
    assert ply_path.exists() and ply_path.stat().st_size > 0
    assert tif_path.exists() and tif_path.stat().st_size > 0
    assert obj_path.exists() and "v " in obj_path.read_text(encoding="utf-8")

    # Simula nivelamento equilibrado (corte ≈ aterro) no centro do terreno
    cx, cy = float(geo.origin_utm[0]), float(geo.origin_utm[1])
    graded, report = simulate_plateau_earthwork(dtm, rect_utm=(cx - 4.0, cy - 2.0, cx + 4.0, cy + 2.0))
    assert report.footprint_area_m2 > 20.0
    assert abs(report.cut_volume_m3 - report.fill_volume_m3) < 1e-3
    assert np.isfinite(graded.grid).any()
    json.dumps({"target_z": report.target_z_m})


def test_export_3dgs_and_view_loaders(tmp_path: Path) -> None:
    bundle_dir = generate_synthetic_bundle(tmp_path / "sample_3dgs", length_x_m=12.0, width_y_m=8.0)
    gs_out = tmp_path / "gs_dataset"
    info = export_gaussian_splatting_dataset(bundle_dir, gs_out, copy_images=True)

    assert info["frame_count"] == 28
    assert info["point_count"] > 100
    transforms = json.loads((gs_out / "transforms.json").read_text(encoding="utf-8"))
    assert len(transforms["frames"]) == 28
    assert transforms["ply_file_path"] == "points3D.ply"
    assert (gs_out / "points3D.ply").exists()
    assert (gs_out / "sparse" / "0" / "cameras.txt").exists()
    assert (gs_out / "sparse" / "0" / "images.txt").exists()
    assert (gs_out / "sparse" / "0" / "points3D.txt").exists()

    geom_ply, title_ply = load_geometry_for_view(gs_out / "points3D.ply")
    assert len(np.asarray(geom_ply.points)) == info["point_count"]
    assert "PLY" in title_ply

    geom_bundle, title_bundle = load_geometry_for_view(bundle_dir, min_confidence=2, voxel_size_m=0.1, refine_poses=True)
    assert len(np.asarray(geom_bundle.points)) > 100
    assert "Bundle" in title_bundle

    import open3d as o3d

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(visible=False)
    vis.add_geometry(geom_ply)
    ctrl = configure_view_controls(vis, geom_ply)
    assert ctrl["move_step"] >= 0.05
    z_before = float(vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic[2, 3])
    assert ctrl["callbacks"][ord("W")](vis) is True
    z_after_in = float(vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic[2, 3])
    assert z_after_in < z_before
    assert ctrl["callbacks"][ord("S")](vis) is True
    vis.destroy_window()
