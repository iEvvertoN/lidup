from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lidup.bundle import load_bundle, reconstruct_lidar_point_cloud, reconstruct_tsdf_mesh
from lidup.cli import configure_view_controls, load_geometry_for_view
from lidup.earthwork import simulate_plateau_earthwork
from lidup.georef import georeference_points, gravity_locked_yaw_alignment, sirgas2000_utm_epsg, umeyama_alignment
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
    assert ctrl["callbacks"][ord("C")](vis) is True
    vis.destroy_window()


def test_gravity_locked_alignment_resists_gnss_vertical_noise() -> None:
    rng = np.random.default_rng(123)
    # ARKit (+X dir, +Y cima, -Z frente) -> engenharia (X, -Z, Y)
    pts_arkit = rng.uniform(-5.0, 5.0, size=(12, 3))
    pts_arkit[:, 1] = rng.uniform(-0.2, 0.2, size=12)

    yaw = 0.72
    r_yaw = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy_eng = np.column_stack([pts_arkit[:, 0], -pts_arkit[:, 2]])
    dst_xy = xy_eng @ r_yaw.T + np.array([462500.0, 9221680.0])
    # Altitude GPS ruidosa (+- 8 m) não deve entortar a gravidade (+Y_arkit -> +Z_utm)
    noisy_z = pts_arkit[:, 1] + 710.0 + rng.normal(0.0, 6.0, size=12)
    dst = np.column_stack([dst_xy, noisy_z])

    scale, rot, _ = gravity_locked_yaw_alignment(pts_arkit, dst, allow_scale=False)
    assert np.isclose(scale, 1.0)
    up_in_utm = rot @ np.array([0.0, 1.0, 0.0])
    assert np.allclose(up_in_utm, np.array([0.0, 0.0, 1.0]), atol=1e-9)


def test_csf_and_physics_posegraph_pipeline(tmp_path: Path) -> None:
    bundle_dir = generate_synthetic_bundle(tmp_path / "sample_csf", length_x_m=16.0, width_y_m=10.0)
    bundle = load_bundle(bundle_dir)

    pcd_pg = reconstruct_lidar_point_cloud(
        bundle,
        min_confidence=2,
        max_depth_m=4.5,
        voxel_size_m=0.08,
        pose_refine="posegraph",
        max_incidence_deg=75.0,
        physics_weight=True,
    )
    pts_arkit = np.asarray(pcd_pg.points)
    assert len(pts_arkit) > 150

    pts_geo, geo = georeference_points(pts_arkit, bundle, lock_gravity=True)
    assert geo.mode == "stationary_gcps"
    assert "gravidade travada" in geo.notes

    classes_csf = classify_ground_points(pts_geo, cell_size_m=0.8, max_height_above_ground_m=0.25, method="csf")
    ground_pts = pts_geo[classes_csf == 2]
    non_ground_pts = pts_geo[classes_csf != 2]
    assert len(ground_pts) > 100
    assert len(non_ground_pts) > 0


def test_tsdf_mesh_and_hires_color(tmp_path: Path) -> None:
    import open3d as o3d

    bundle_dir = generate_synthetic_bundle(tmp_path / "sample_tsdf", length_x_m=12.0, width_y_m=8.0)
    bundle = load_bundle(bundle_dir)

    pcd, mesh = reconstruct_tsdf_mesh(
        bundle,
        min_confidence=2,
        max_depth_m=4.5,
        tsdf_voxel_m=0.04,
        pose_refine="posegraph",
        hires_color=True,
    )
    assert len(pcd.points) > 100
    assert len(mesh.vertices) > 100
    assert len(mesh.triangles) > 100

    mesh_path = tmp_path / "object_mesh_local.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    loaded, title = load_geometry_for_view(tmp_path)
    assert "Mesh" in title
    assert len(np.asarray(loaded.triangles)) == len(mesh.triangles)


def test_bake_uv_texture_obj_produces_loadable_textured_mesh(tmp_path: Path) -> None:
    from lidup.bundle import reconstruct_tsdf_details
    from lidup.texture import bake_uv_texture_obj

    bundle_dir = generate_synthetic_bundle(tmp_path / "sample_uv", length_x_m=10.0, width_y_m=8.0)
    bundle = load_bundle(bundle_dir)
    _, mesh, extracted, poses = reconstruct_tsdf_details(
        bundle, min_confidence=2, max_depth_m=4.5, tsdf_voxel_m=0.05, pose_refine="none", depth_upsample=1
    )
    verts_local = np.asarray(mesh.vertices, dtype=np.float64)
    obj_path = bake_uv_texture_obj(
        bundle,
        mesh,
        verts_local,
        extracted,
        poses,
        tmp_path / "object_textured.obj",
        atlas_size=1024,
        max_triangles=8000,
    )
    assert obj_path.exists()
    assert obj_path.with_suffix(".mtl").exists()
    assert obj_path.with_suffix(".jpg").exists()

    loaded, title = load_geometry_for_view(obj_path)
    assert "OBJ-UV" in title
    assert loaded.has_textures()
    assert loaded.has_triangle_uvs()


def test_guided_depth_upsample_edge_unmixing_and_focus() -> None:
    from lidup.bundle import _estimate_orbit_focus_point, _upsample_depth_guided_by_rgb

    d_h, d_w, scale = 32, 40, 2
    depth_lo = np.full((d_h, d_w), 2.5, dtype=np.float64)
    depth_lo[:, :20] = 1.0
    valid_lo = np.ones((d_h, d_w), dtype=bool)
    conf_lo = np.full((d_h, d_w), 2, dtype=np.uint8)

    rgb_lo = np.zeros((d_h, d_w, 3), dtype=np.float64)
    rgb_lo[:, :20] = [0.9, 0.2, 0.2]
    rgb_lo[:, 20:] = [0.1, 0.2, 0.8]

    rgb_hi = np.zeros((d_h * scale, d_w * scale, 3), dtype=np.float64)
    rgb_hi[:, : 20 * scale] = [0.9, 0.2, 0.2]
    rgb_hi[:, 20 * scale :] = [0.1, 0.2, 0.8]

    up_depth, up_mask, _ = _upsample_depth_guided_by_rgb(
        depth_lo, valid_lo, conf_lo, rgb_lo, rgb_hi, scale=scale, edge_jump_m=0.12
    )
    assert up_depth.shape == (d_h * scale, d_w * scale)
    fg = up_depth[:, : 20 * scale][up_mask[:, : 20 * scale]]
    bg = up_depth[:, 20 * scale :][up_mask[:, 20 * scale :]]
    assert len(fg) > 0 and len(bg) > 0
    assert np.all(np.abs(fg - 1.0) < 0.03)
    assert np.all(np.abs(bg - 2.5) < 0.03)

    target = np.array([1.5, 0.5, -2.0])
    poses = []
    for ang in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        cam_pos = target + np.array([2.0 * np.cos(ang), 0.2, 2.0 * np.sin(ang)])
        z_axis = target - cam_pos
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        tf = np.eye(4)
        tf[:3, 0] = x_axis
        tf[:3, 1] = y_axis
        tf[:3, 2] = z_axis
        tf[:3, 3] = cam_pos
        poses.append(tf)

    focus = _estimate_orbit_focus_point(poses)
    assert focus is not None
    assert np.allclose(focus, target, atol=0.15)


def test_depth_only_intermediate_frames_supported(tmp_path: Path) -> None:
    bundle_dir = generate_synthetic_bundle(tmp_path / "sample_depthonly", length_x_m=10.0, width_y_m=8.0)
    frames_csv = bundle_dir / "frames.csv"
    lines = frames_csv.read_text(encoding="utf-8").splitlines()
    # Remove o image_file de um frame intermediário simulando burst somente de depth
    parts = lines[2].split(",")
    parts[2] = ""
    lines[2] = ",".join(parts)
    frames_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    bundle = load_bundle(bundle_dir)
    pcd = reconstruct_lidar_point_cloud(bundle, voxel_size_m=0.06, depth_upsample=2)
    assert len(pcd.points) > 100


def test_warp_rgb_to_depth_grid_corrects_rotation() -> None:
    from scipy.spatial.transform import Rotation
    from lidup.bundle import _warp_rgb_to_depth_grid

    d_h, d_w, scale = 24, 32, 2
    u_h, u_w = d_h * scale, d_w * scale
    depth_lo = np.full((d_h, d_w), 1.0, dtype=np.float64)
    valid_lo = np.ones((d_h, d_w), dtype=bool)
    rgb_hi = np.zeros((u_h, u_w, 3), dtype=np.float64)
    rgb_hi[:, u_w // 2 :] = [1.0, 0.2, 0.2]

    pose_d = np.eye(4)
    pose_r = np.eye(4)
    pose_r[:3, :3] = Rotation.from_euler("y", 3.0, degrees=True).as_matrix()

    warped = _warp_rgb_to_depth_grid(
        rgb_hi, depth_lo, valid_lo, pose_d, pose_r, 40.0, 40.0, u_w / 2.0, u_h / 2.0, scale
    )
    assert warped.shape == (u_h, u_w, 3)
    assert float(warped[u_h // 2, -2, 0]) > 0.8
    assert float(warped[u_h // 2, 2, 0]) < 0.2


def test_filter_floor_mirror_outliers_and_object_focus() -> None:
    from lidup.bundle import _ExtractedFrame, _filter_floor_mirror_outliers, _refine_object_focus_point, FrameRecord

    def make_ext(y_val: float, with_obj: bool = False) -> _ExtractedFrame:
        xs, zs = np.meshgrid(np.linspace(-0.2, 0.2, 12), np.linspace(-0.2, 0.2, 12))
        floor = np.column_stack([xs.ravel(), np.full(144, y_val), zs.ravel()])
        if with_obj:
            ox, oz = np.meshgrid(np.linspace(-0.05, 0.05, 8), np.linspace(-0.05, 0.05, 8))
            obj = np.column_stack([ox.ravel(), np.full(64, y_val + 0.10), oz.ravel()])
            pts = np.vstack([floor, obj])
        else:
            pts = floor
        fr = FrameRecord("0", 0.0, "", "", "", np.eye(4), 100.0, 100.0, 10.0, 10.0, 10.0, "normal", None, None, None, None, None)
        return _ExtractedFrame(
            frame=fr,
            cam_pts=pts,
            weights=np.ones(len(pts)),
            pose_depth=np.eye(4),
            rgb_arr=None,
            depth_filtered_m=np.zeros((20, 20), dtype=np.float32),
            v_idx=np.zeros(len(pts), dtype=np.int64),
            u_idx=np.zeros(len(pts), dtype=np.int64),
            fx=100.0, fy=100.0, cx=10.0, cy=10.0, d_w=20, d_h=20, img_w=20, img_h=20, poses_differ=False,
        )

    exts = [make_ext(-0.30, True) for _ in range(5)] + [make_ext(-0.55, False)]
    poses = [np.eye(4) for _ in exts]
    clean_ext, clean_poses, floor_y = _filter_floor_mirror_outliers(exts, poses)
    assert len(clean_ext) == 5
    assert floor_y is not None and abs(floor_y - (-0.30)) < 0.02
    obj_focus = _refine_object_focus_point(clean_ext, clean_poses, np.array([0.0, -0.25, 0.0]), floor_y)
    assert obj_focus[1] > -0.25


def test_imu_gyro_reader_and_mvs_photometric(tmp_path: Path) -> None:
    bundle_dir = generate_synthetic_bundle(tmp_path / "sample_mvs", length_x_m=8.0, width_y_m=6.0)
    # Grava imu.csv sintético compatível com os timestamps do bundle
    bundle = load_bundle(bundle_dir)
    lines = ["timestamp_s,gx,gy,gz,ax,ay,az,grav_x,grav_y,grav_z,qx,qy,qz,qw"]
    for fr in bundle.frames:
        lines.append(f"{fr.timestamp_s:.6f},0.05,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,0.0,1.0")
    (bundle_dir / "imu.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pcd, mesh = reconstruct_tsdf_mesh(
        bundle,
        min_confidence=2,
        max_depth_m=4.5,
        tsdf_voxel_m=0.04,
        pose_refine="none",
        hires_color=True,
        depth_upsample=1,
        use_mvs=True,
        photometric_opt=True,
    )
    assert len(pcd.points) > 100
    assert len(mesh.vertices) > 100
    assert len(mesh.triangles) > 100


def test_analyze_room_geometry_exports_plan_and_metrics(tmp_path: Path) -> None:
    from lidup.architecture import analyze_room_geometry

    rng = np.random.default_rng(7)
    gx, gz = np.meshgrid(np.linspace(0.0, 4.0, 45), np.linspace(0.0, 5.0, 55))
    floor = np.column_stack([gx.ravel(), rng.normal(0.0, 0.002, gx.size), gz.ravel()])
    ceil = np.column_stack([gx.ravel(), np.full(gx.size, 2.8), gz.ravel()])

    # 4 paredes verticais (eixo Y = up ARKit)
    u4, h4 = np.meshgrid(np.linspace(0.0, 4.0, 40), np.linspace(0.0, 2.8, 28))
    wall_n = np.column_stack([u4.ravel(), h4.ravel(), np.zeros(u4.size)])
    wall_s = np.column_stack([u4.ravel(), h4.ravel(), np.full(u4.size, 5.0)])
    v5, h5 = np.meshgrid(np.linspace(0.0, 5.0, 50), np.linspace(0.0, 2.8, 28))
    wall_w = np.column_stack([np.zeros(v5.size), h5.ravel(), v5.ravel()])
    wall_e = np.column_stack([np.full(v5.size, 4.0), h5.ravel(), v5.ravel()])

    room_pts = np.vstack([floor, ceil, wall_n, wall_s, wall_w, wall_e])
    report = analyze_room_geometry(room_pts, tmp_path / "room_out", up_axis="y")

    assert abs(report.ceiling_height_m - 2.8) < 0.06
    assert abs(report.floor_area_m2 - 20.0) < 1.2
    assert report.floor_flatness_rms_mm < 5.0
    assert report.detected_wall_count == 4
    assert Path(report.outputs["floorplan_svg"]).exists()
    assert Path(report.outputs["floorplan_dxf"]).exists()
    assert Path(report.outputs["floor_flatness_ply"]).exists()
    assert Path(report.outputs["room_report_json"]).exists()
    dxf_txt = Path(report.outputs["floorplan_dxf"]).read_text(encoding="utf-8")
    assert "WALLS" in dxf_txt and "LINE" in dxf_txt


def test_compute_picked_measurements() -> None:
    from lidup.cli import compute_picked_measurements

    pts = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 2.0]])
    segs = compute_picked_measurements(pts, [0, 1, 2, 99], up_axis="y")
    assert len(segs) == 2
    assert abs(segs[0]["dist_3d_m"] - 3.0) < 1e-4
    assert abs(segs[0]["slope_pct"] - 0.0) < 1e-4
    assert abs(segs[1]["dist_3d_m"] - 4.0) < 1e-4
    assert abs(segs[1]["dv_up_m"] - 4.0) < 1e-4


def test_rust_core_bridge_loaded_and_accurate() -> None:
    from lidup._rust_bridge import (
        rust_available,
        rust_colorize_hires_frame,
        rust_csf_cloth_filter,
        rust_depth_float32_to_mm_u16,
        rust_extract_depth_points,
        rust_resize_rgb_u8_to_f64,
        rust_sharpness_laplacian_u8,
        rust_warp_rgb_to_depth,
    )

    assert rust_available() is True

    raw = np.array([[1.25, 0.0], [2.5, np.nan]], dtype=np.float32).tobytes()
    mm = rust_depth_float32_to_mm_u16(raw, stride_bytes=8, width=2, height=2)
    assert mm is not None
    assert mm.tolist() == [[1250, 0], [2500, 0]]

    flat_img = np.full((32, 32), 128, dtype=np.uint8)
    sharp_img = np.zeros((32, 32), dtype=np.uint8)
    sharp_img[:, ::2] = 255
    s_flat = rust_sharpness_laplacian_u8(flat_img, sample_step=1)
    s_sharp = rust_sharpness_laplacian_u8(sharp_img, sample_step=1)
    assert s_flat == 0.0
    assert s_sharp is not None and s_sharp > 1000.0

    # Resize + extract depth points + warp + colorize + CSF
    src_u8 = np.zeros((8, 8, 3), dtype=np.uint8)
    src_u8[:, :, 0] = 255
    resized = rust_resize_rgb_u8_to_f64(src_u8, 4, 4)
    assert resized is not None and resized.shape == (4, 4, 3) and abs(resized[0, 0, 0] - 1.0) < 1e-4

    d_mat = np.full((8, 10), 2.0, dtype=np.float64)
    m_mat = np.ones((8, 10), dtype=bool)
    ext = rust_extract_depth_points(d_mat, m_mat, 100.0, 100.0, 5.0, 4.0, 0.12, 0.25)
    assert ext is not None and len(ext[1]) == 80

    warped = rust_warp_rgb_to_depth(np.ones((8, 10, 3)), d_mat, np.eye(4), 100.0, 100.0, 5.0, 4.0)
    assert warped is not None and warped.shape == (8, 10, 3)

    best_s = np.array([-1.0], dtype=np.float64)
    out_c = np.zeros((1, 3), dtype=np.float64)
    hi_u8 = np.full((8, 10, 3), 204, dtype=np.uint8)
    ok = rust_colorize_hires_frame(
        np.array([[0.0, 0.0, 2.0]]), None, np.eye(4), hi_u8, d_mat.astype(np.float32),
        100.0, 100.0, 5.0, 4.0, 5.0, 0.1, 1.0, best_s, out_c,
    )
    assert ok and best_s[0] > 0 and abs(out_c[0, 1] - 0.8) < 0.01

    tf = np.zeros((5, 5), dtype=np.float64)
    tf[2, 2] = -1.5
    dtm = rust_csf_cloth_filter(tf, 0.0, 2, 100, 0.25)
    assert dtm is not None and dtm.shape == (5, 5) and dtm[2, 2] > -0.8


def test_studio_discovery_and_measure_parser(tmp_path: Path) -> None:
    from lidup.studio import discover_workspace_items, parse_measure_output

    bundle_dir = generate_synthetic_bundle(tmp_path / "2026-11-19_120000_setor", length_x_m=6.0, width_y_m=4.0)
    out_dir = tmp_path / "out_120000"
    out_dir.mkdir()
    (out_dir / "object_textured.obj").write_text("# obj", encoding="utf-8")
    (out_dir / "floorplan.svg").write_text("<svg/>", encoding="utf-8")

    found = discover_workspace_items(tmp_path)
    assert len(found["bundles"]) == 1
    assert found["bundles"][0]["name"] == bundle_dir.name
    assert found["bundles"][0]["frames"] > 0
    assert len(found["outputs"]) == 1
    assert found["outputs"][0]["artifacts"]["obj_uv"] is True
    assert found["outputs"][0]["artifacts"]["svg"] is True

    sample_out = (
        "[lidup] Trecho 1 (#10 -> #25): 3D = 2.500 m (250.0 cm) | "
        "Horiz = 2.400 m | dV = +0.700 m | Incl = 29.2% | (dX=+2.400, dY=+0.700, dZ=+0.000)\n"
    )
    rows = parse_measure_output(sample_out)
    assert len(rows) == 1
    assert rows[0]["dist_m"] == "2.500"
    assert rows[0]["slope_pct"] == "29.2"


def test_simulations_diff_drainage_and_clearance(tmp_path: Path) -> None:
    from lidup.simulations import (
        analyze_clearance_and_viewshed,
        diff_point_clouds,
        simulate_water_ponding,
    )

    gx, gz = np.meshgrid(np.linspace(0.0, 4.0, 45), np.linspace(0.0, 4.0, 45))
    y_flat = np.zeros(gx.size)
    before = np.column_stack([gx.ravel(), y_flat, gz.ravel()])

    # Adiciona um bloco elevado (+0.15 m) no centro em "after"
    after = before.copy()
    bump = (after[:, 0] >= 1.5) & (after[:, 0] <= 2.5) & (after[:, 2] >= 1.5) & (after[:, 2] <= 2.5)
    after[bump, 1] = 0.15
    d_rep = diff_point_clouds(before, after, tmp_path / "diff_out", align=False, tol_mm=15.0)
    assert d_rep.max_mm > 120.0
    assert d_rep.added_pct > 2.0
    assert Path(d_rep.outputs["diff_heatmap_ply"]).exists()

    # Bacia de empoçamento de -6 cm no centro de um piso 4x4m
    bowl = before.copy()
    dist_c = np.hypot(bowl[:, 0] - 2.0, bowl[:, 2] - 2.0)
    bowl[:, 1] = np.where(dist_c < 0.7, -0.06 * (1.0 - dist_c / 0.7), 0.0)
    dr_rep = simulate_water_ponding(bowl, tmp_path / "drain_out", up_axis="y", cell_m=0.08)
    assert dr_rep.max_depth_mm > 35.0
    assert dr_rep.water_volume_liters > 5.0
    assert Path(dr_rep.outputs["ponding_map_svg"]).exists()

    # Sala 4x4m com parede divisória central contendo um pilar/obstáculo -> testa passagem + CFTV
    obs_x = np.full(80, 2.0)
    obs_z = np.concatenate([np.linspace(0.0, 1.2, 40), np.linspace(2.8, 4.0, 40)])
    obs_pts = np.column_stack([obs_x, np.full(80, 1.0), obs_z])
    room_obs = np.vstack([before, obs_pts])
    cl_rep = analyze_clearance_and_viewshed(
        room_obs,
        tmp_path / "clr_out",
        box_width_m=0.60,
        box_depth_m=0.80,
        box_height_m=1.50,
        sensor_xy_m=(0.5, 2.0),
        up_axis="y",
        cell_m=0.10,
    )
    assert cl_rep.passable_pct > 30.0
    assert 10.0 < cl_rep.viewshed_pct < 99.0
    assert Path(cl_rep.outputs["clearance_map_svg"]).exists()
    assert Path(cl_rep.outputs["viewshed_map_ply"]).exists()


def test_ruler_ray_picking_and_segments() -> None:
    import open3d as o3d
    from lidup.ruler import compute_ruler_segments, pick_point_on_ray, raycast_geometry

    pts = np.array([[0.0, 0.0, 2.0], [1.5, 0.0, 2.0], [1.5, 1.0, 2.0]], dtype=np.float64)
    hit = pick_point_on_ray(pts, np.array([1.5, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert hit is not None and np.allclose(hit, [1.5, 0.0, 2.0])

    # TriangleMesh raycast exact hit
    box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(box)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(t_mesh)
    mesh_hit = raycast_geometry(box, np.array([0.5, 0.5, 3.0]), np.array([0.0, 0.0, -1.0]), raycaster=rc)
    assert mesh_hit is not None and abs(mesh_hit[2] - 1.0) < 1e-4

    segs = compute_ruler_segments(pts, up_axis="y")
    assert len(segs) == 2
    assert abs(segs[0]["dist_3d_m"] - 1.5) < 1e-4
    assert abs(segs[1]["dv_up_m"] - 1.0) < 1e-4


def test_fase2_plumbness_volume_and_reflective(tmp_path: Path) -> None:
    from PIL import Image
    from lidup.simulations import compute_object_or_pile_volume, detect_reflective_surfaces

    # Cubo 1.0 x 1.0 x 0.5 m em cima de piso 4 x 4 m -> volume esperado ~ 0.50 m³
    gx, gz = np.meshgrid(np.linspace(0.0, 4.0, 60), np.linspace(0.0, 4.0, 60))
    pts_y = np.zeros(gx.size)
    in_pile = (gx.ravel() >= 1.5) & (gx.ravel() < 2.5) & (gz.ravel() >= 1.5) & (gz.ravel() < 2.5)
    pts_y[in_pile] = 0.50
    pile_pts = np.column_stack([gx.ravel(), pts_y, gz.ravel()])

    v_rep = compute_object_or_pile_volume(pile_pts, tmp_path / "vol_out", up_axis="y", cell_m=0.05)
    assert abs(v_rep.volume_m3 - 0.50) < 0.12
    assert abs(v_rep.max_height_m - 0.50) < 0.05
    assert Path(v_rep.outputs["volume_heatmap_ply"]).exists()

    # Detector de vidros/espelhos via confidence == 0 no bundle
    bdir = generate_synthetic_bundle(tmp_path / "glass_bundle", length_x_m=6.0, width_y_m=4.0)
    conf_files = sorted((bdir / "confidence").glob("*.png"))
    arr = np.asarray(Image.open(conf_files[0]), dtype=np.uint8).copy()
    arr[:40, :40] = 0
    Image.fromarray(arr).save(conf_files[0])

    r_rep = detect_reflective_surfaces(bdir, tmp_path / "ref_out", frame_stride=1)
    assert r_rep.reflective_point_count >= 100
    assert Path(r_rep.outputs["reflective_zones_ply"]).exists()


def test_pixel_ray_and_camera_navigation() -> None:
    from lidup.ruler import compute_pixel_ray, preset_camera_pose, step_camera_pose

    class _FakeCam:
        def unproject(self, x: float, y: float, z: float, w: float, h: float) -> np.ndarray:
            if z >= 0.99:
                return np.array([float("nan"), float("nan"), float("-inf")])
            nx = (x - w * 0.5) / max(w * 0.5, 1.0)
            ny = -(y - h * 0.5) / max(h * 0.5, 1.0)
            depth = 0.1 + 9.9 * z
            return np.array([nx * depth * 0.5, ny * depth * 0.5, 5.0 - depth])

    ray = compute_pixel_ray(_FakeCam(), 400.0, 300.0, 800.0, 600.0)
    assert ray is not None
    orig, direction = ray
    assert np.all(np.isfinite(orig))
    assert np.allclose(direction, [0.0, 0.0, -1.0], atol=1e-6)

    vm = np.eye(4, dtype=np.float64)
    vm[:3, 3] = [0.0, 0.0, -4.0]  # eye = [0, 0, 4], looking -Z
    c_fwd, e_fwd, _ = step_camera_pose(vm, "forward", step_m=0.5, up_axis="y")
    assert np.allclose(e_fwd, [0.0, 0.0, 3.5], atol=1e-6)
    assert c_fwd[2] < e_fwd[2]

    _, e_right, _ = step_camera_pose(vm, "right", step_m=0.5, up_axis="y")
    assert np.allclose(e_right, [0.5, 0.0, 4.0], atol=1e-6)

    _, e_up, _ = step_camera_pose(vm, "up", step_m=0.5, up_axis="y")
    assert np.allclose(e_up, [0.0, 0.5, 4.0], atol=1e-6)

    c_yaw, e_yaw, _ = step_camera_pose(vm, "yaw_right", turn_rad=0.25, pivot_dist=2.0, up_axis="y")
    assert np.allclose(e_yaw, [0.0, 0.0, 4.0], atol=1e-6)
    assert (c_yaw - e_yaw)[0] > 0.0

    ctr, eye_top, up_top = preset_camera_pose([-1.0, 0.0, -1.0], [1.0, 2.0, 1.0], "top", up_axis="y")
    assert np.allclose(ctr, [0.0, 1.0, 0.0])
    assert eye_top[1] > ctr[1]
    assert abs(float(np.dot(up_top, [0.0, 1.0, 0.0]))) < 0.1


def test_lidar_only_bundle_without_images(tmp_path: Path) -> None:
    import csv
    import shutil
    from lidup.bundle import load_bundle, reconstruct_lidar_point_cloud, reconstruct_tsdf_mesh

    bdir = generate_synthetic_bundle(tmp_path / "lidar_only_bundle", length_x_m=6.0, width_y_m=4.0)
    shutil.rmtree(bdir / "images")

    frames_path = bdir / "frames.csv"
    rows = list(csv.DictReader(frames_path.open("r", encoding="utf-8", newline="")))
    fieldnames = list(rows[0].keys())
    for r in rows:
        r["image_file"] = ""
    with frames_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    bundle = load_bundle(bdir)
    pcd = reconstruct_lidar_point_cloud(bundle, min_confidence=2, voxel_size_m=0.04, hires_color=False)
    assert len(pcd.points) > 100

    _, mesh = reconstruct_tsdf_mesh(bundle, min_confidence=2, tsdf_voxel_m=0.04, refine_poses=False, hires_color=False)
    assert len(mesh.vertices) > 100







