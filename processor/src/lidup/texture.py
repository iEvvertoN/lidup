from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from lidup.bundle import LidupBundle, _ExtractedFrame, _load_imu_gyro_map


@dataclass
class _ChartSpec:
    frame_idx: int
    tri_indices: np.ndarray
    vert_indices: np.ndarray
    u_px: np.ndarray
    v_px: np.ndarray
    crop_box: tuple[int, int, int, int]  # u0, v0, u1, v1 na imagem original
    hi_w: int
    hi_h: int
    atlas_rect: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h no atlas PIL


def _pack_shelves(
    rects: list[tuple[int, int]],
    canvas_w: int,
    usable_h: int,
    gap: int = 4,
) -> tuple[bool, list[tuple[int, int, int, int]]]:
    order = sorted(range(len(rects)), key=lambda idx: rects[idx][1], reverse=True)
    placed: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)] * len(rects)
    cur_x = gap
    cur_y = gap
    row_h = 0

    for idx in order:
        w, h = rects[idx]
        w = max(8, w)
        h = max(8, h)
        if w + 2 * gap > canvas_w:
            return False, placed
        if cur_x + w + gap > canvas_w:
            cur_x = gap
            cur_y += row_h + gap
            row_h = 0
        if cur_y + h + gap > usable_h:
            return False, placed
        placed[idx] = (cur_x, cur_y, w, h)
        cur_x += w + gap
        row_h = max(row_h, h)

    return True, placed


def bake_uv_texture_obj(
    bundle: LidupBundle,
    mesh_arkit: o3d.geometry.TriangleMesh,
    verts_export: np.ndarray,
    extracted: list[_ExtractedFrame],
    poses_arkit: list[np.ndarray],
    out_obj_path: str | Path,
    *,
    max_depth_m: float = 4.5,
    occ_tol_m: float = 0.035,
    atlas_size: int = 4096,
    max_triangles: int = 220_000,
) -> Path:
    """
    Gera malha Wavefront OBJ com atlas UV fotográfico (12 MP) projetando cada gráfico
    de triângulos diretamente na câmera mais frontal, nítida e estável na IMU.
    """
    out_obj = Path(out_obj_path)
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = out_obj.with_suffix(".mtl")
    tex_path = out_obj.with_suffix(".jpg")

    work_mesh = o3d.geometry.TriangleMesh(mesh_arkit)
    if max_triangles > 0 and len(work_mesh.triangles) > max_triangles:
        work_mesh = work_mesh.simplify_quadric_decimation(target_number_of_triangles=int(max_triangles))
        work_mesh.remove_degenerate_triangles()
        work_mesh.remove_duplicated_vertices()
        work_mesh.compute_vertex_normals()
        verts_export_use = None  # sera reconstruído pela transformação rígida abaixo
    else:
        verts_export_use = np.asarray(verts_export, dtype=np.float64)

    verts_arkit = np.asarray(work_mesh.vertices, dtype=np.float64)
    tris = np.asarray(work_mesh.triangles, dtype=np.int64)
    if len(verts_arkit) == 0 or len(tris) == 0:
        raise RuntimeError("Malha vazia para texturização UV.")

    if verts_export_use is None:
        # Infere transformação rígida/escala exata entre mesh_arkit original e verts_export
        src_orig = np.asarray(mesh_arkit.vertices, dtype=np.float64)
        dst_orig = np.asarray(verts_export, dtype=np.float64)
        if len(src_orig) >= 4 and len(src_orig) == len(dst_orig):
            idx_s = np.linspace(0, len(src_orig) - 1, min(200, len(src_orig)), dtype=np.int64)
            s_sub = src_orig[idx_s]
            d_sub = dst_orig[idx_s]
            mu_s = s_sub.mean(axis=0)
            mu_d = d_sub.mean(axis=0)
            sc = s_sub - mu_s
            dc = d_sub - mu_d
            h_mat = sc.T @ dc
            u_m, _, vt_m = np.linalg.svd(h_mat)
            rot = vt_m.T @ u_m.T
            if np.linalg.det(rot) < 0:
                vt_m[-1, :] *= -1
                rot = vt_m.T @ u_m.T
            scale = float(np.sqrt(np.sum(dc * dc) / max(np.sum(sc * sc), 1e-12)))
            verts_export_use = (verts_arkit - mu_s) @ rot.T * scale + mu_d
        else:
            verts_export_use = verts_arkit.copy()

    # Normais no referencial final exportado
    export_mesh = o3d.geometry.TriangleMesh()
    export_mesh.vertices = o3d.utility.Vector3dVector(verts_export_use)
    export_mesh.triangles = o3d.utility.Vector3iVector(tris)
    export_mesh.compute_vertex_normals()
    v_norms_export = np.asarray(export_mesh.vertex_normals, dtype=np.float64)

    work_mesh.compute_triangle_normals()
    tri_norms = np.asarray(work_mesh.triangle_normals, dtype=np.float64)
    tri_cents = verts_arkit[tris].mean(axis=1)

    v_cols = np.asarray(work_mesh.vertex_colors, dtype=np.float64)
    if len(v_cols) != len(verts_arkit):
        v_cols = np.full((len(verts_arkit), 3), 0.65, dtype=np.float64)

    imu_map = _load_imu_gyro_map(bundle)
    sharp_vals = [e.frame.sharpness for e in extracted if e.frame.sharpness > 0]
    med_sharp = float(np.median(sharp_vals)) if sharp_vals else 0.0

    best_score = np.full(len(tris), -1.0, dtype=np.float64)
    best_frame = np.full(len(tris), -1, dtype=np.int64)
    img_sizes: dict[int, tuple[int, int]] = {}

    for k, (ext, pose) in enumerate(zip(extracted, poses_arkit)):
        if not ext.frame.image_file:
            continue
        img_path = bundle.root / ext.frame.image_file
        if not img_path.exists():
            continue

        with Image.open(img_path) as im:
            hi_w, hi_h = im.size
        img_sizes[k] = (hi_w, hi_h)

        rel_rgb = np.linalg.inv(ext.pose_depth) @ ext.frame.t_wc
        pose_rgb = pose @ rel_rgb

        sx = hi_w / float(max(ext.img_w, 1))
        sy = hi_h / float(max(ext.img_h, 1))
        fx_hi = ext.frame.fx * sx
        fy_hi = ext.frame.fy * sy
        cx_hi = ext.frame.cx * sx
        cy_hi = ext.frame.cy * sy

        # Verifica primeiro quais vértices caem dentro do quadro
        cam_v = (verts_arkit - pose_rgb[:3, 3]) @ pose_rgb[:3, :3]
        zv = cam_v[:, 2]
        in_z_v = (zv > 0.20) & (zv <= max_depth_m)
        safe_zv = np.where(in_z_v, zv, 1.0)
        uv_x = fx_hi * (cam_v[:, 0] / safe_zv) + cx_hi
        uv_y = fy_hi * (cam_v[:, 1] / safe_zv) + cy_hi
        v_inside = in_z_v & (uv_x >= 1.0) & (uv_x <= hi_w - 2.0) & (uv_y >= 1.0) & (uv_y <= hi_h - 2.0)

        tri_v_ok = v_inside[tris[:, 0]] & v_inside[tris[:, 1]] & v_inside[tris[:, 2]]
        if not np.any(tri_v_ok):
            continue

        cand_t = np.nonzero(tri_v_ok)[0]
        cam_c = (tri_cents[cand_t] - pose_rgb[:3, 3]) @ pose_rgb[:3, :3]
        zc = cam_c[:, 2]
        uc = fx_hi * (cam_c[:, 0] / zc) + cx_hi
        vc = fy_hi * (cam_c[:, 1] / zc) + cy_hi

        in_c = (zc > 0.20) & (zc <= max_depth_m) & (uc >= 0.0) & (uc < hi_w) & (vc >= 0.0) & (vc < hi_h)
        if not np.any(in_c):
            continue

        cand_t = cand_t[in_c]
        cam_c = cam_c[in_c]
        zc = zc[in_c]
        uc = uc[in_c]
        vc = vc[in_c]

        u_d = np.clip(np.rint(uc * (ext.d_w / float(hi_w))).astype(np.int64), 0, ext.d_w - 1)
        v_d = np.clip(np.rint(vc * (ext.d_h / float(hi_h))).astype(np.int64), 0, ext.d_h - 1)
        d_obs = ext.depth_filtered_m[v_d, u_d]
        not_occ = (d_obs > 0.0) & (np.abs(zc - d_obs) <= occ_tol_m)
        if not np.any(not_occ):
            continue

        cand_t = cand_t[not_occ]
        cam_c = cam_c[not_occ]
        zc = zc[not_occ]

        view_dir = -cam_c / np.maximum(np.linalg.norm(cam_c, axis=1, keepdims=True), 1e-8)
        norms_cam = tri_norms[cand_t] @ pose_rgb[:3, :3]
        cos_v = np.sum(norms_cam * view_dir, axis=1)
        facing = cos_v > 0.22
        if not np.any(facing):
            continue

        cand_t = cand_t[facing]
        cos_v = cos_v[facing]
        zc = zc[facing]

        sharp_w = float(np.clip(np.sqrt(ext.frame.sharpness / med_sharp), 0.6, 1.4)) if (med_sharp > 0 and ext.frame.sharpness > 0) else 1.0
        gyro_val = ext.gyro_deg_s if ext.gyro_deg_s is not None else imu_map.get(ext.frame.frame_id)
        imu_w = float(np.clip(20.0 / max(gyro_val, 12.0), 0.55, 1.30)) if gyro_val is not None else 1.0

        scores = (cos_v * sharp_w * imu_w) / np.maximum(zc, 0.25) ** 2
        better = scores > best_score[cand_t]
        win_t = cand_t[better]
        best_score[win_t] = scores[better]
        best_frame[win_t] = k

    # Constroi specs de gráficos (um por câmera vencedora)
    charts: list[_ChartSpec] = []
    for k in sorted(img_sizes.keys()):
        t_idx = np.nonzero(best_frame == k)[0]
        if len(t_idx) == 0:
            continue
        ext = extracted[k]
        pose = poses_arkit[k]
        hi_w, hi_h = img_sizes[k]

        rel_rgb = np.linalg.inv(ext.pose_depth) @ ext.frame.t_wc
        pose_rgb = pose @ rel_rgb
        sx = hi_w / float(max(ext.img_w, 1))
        sy = hi_h / float(max(ext.img_h, 1))
        fx_hi = ext.frame.fx * sx
        fy_hi = ext.frame.fy * sy
        cx_hi = ext.frame.cx * sx
        cy_hi = ext.frame.cy * sy

        v_idx = np.unique(tris[t_idx].ravel())
        cam_v = (verts_arkit[v_idx] - pose_rgb[:3, 3]) @ pose_rgb[:3, :3]
        z = np.maximum(cam_v[:, 2], 0.1)
        u_px = np.clip(fx_hi * (cam_v[:, 0] / z) + cx_hi, 0.0, float(hi_w - 1))
        v_px = np.clip(fy_hi * (cam_v[:, 1] / z) + cy_hi, 0.0, float(hi_h - 1))

        pad = 6
        u0 = max(0, int(np.floor(u_px.min())) - pad)
        v0 = max(0, int(np.floor(v_px.min())) - pad)
        u1 = min(hi_w, int(np.ceil(u_px.max())) + pad)
        v1 = min(hi_h, int(np.ceil(v_px.max())) + pad)
        charts.append(
            _ChartSpec(
                frame_idx=k,
                tri_indices=t_idx,
                vert_indices=v_idx,
                u_px=u_px,
                v_px=v_px,
                crop_box=(u0, v0, u1, v1),
                hi_w=hi_w,
                hi_h=hi_h,
            )
        )

    # Reserva faixa inferior de 64px para cor sólida dos triângulos sem visão (oclusos)
    fallback_strip_h = 64
    usable_h = atlas_size - fallback_strip_h
    nat_sizes = [(max(16, c.crop_box[2] - c.crop_box[0]), max(16, c.crop_box[3] - c.crop_box[1])) for c in charts]
    total_area = sum(w * h for w, h in nat_sizes)
    scale = min(1.0, float(np.sqrt((0.70 * atlas_size * usable_h) / max(total_area, 1))))

    placed: list[tuple[int, int, int, int]] = []
    for _ in range(18):
        scaled = [(max(16, int(round(w * scale))), max(16, int(round(h * scale)))) for w, h in nat_sizes]
        ok, rects = _pack_shelves(scaled, atlas_size, usable_h, gap=4)
        if ok:
            placed = rects
            break
        scale *= 0.85

    if charts and not placed:
        raise RuntimeError("Não foi possível empacotar os gráficos UV no atlas de textura.")

    atlas_img = Image.new("RGB", (atlas_size, atlas_size), (165, 165, 165))
    for chart, rect in zip(charts, placed):
        chart.atlas_rect = rect
        ext = extracted[chart.frame_idx]
        img_path = bundle.root / ext.frame.image_file
        u0, v0, u1, v1 = chart.crop_box
        ax, ay, aw, ah = rect
        with Image.open(img_path) as im:
            patch = im.convert("RGB").crop((u0, v0, u1, v1)).resize((aw, ah), Image.Resampling.LANCZOS)
            atlas_img.paste(patch, (ax, ay))

    # Pinta faixa de fallback (paleta 64 blocos de cor quantizada 4x4x4 no rodapé do atlas)
    # Bloco (r, g, b) em {0..3}^3 -> 64 células de 16x64 px (largura total 1024 px)
    block_w = 16
    for q_idx in range(64):
        rq = (q_idx >> 4) & 3
        gq = (q_idx >> 2) & 3
        bq = q_idx & 3
        col_u8 = (int(rq * 85), int(gq * 85), int(bq * 85))
        cell = Image.new("RGB", (block_w, fallback_strip_h), col_u8)
        atlas_img.paste(cell, (q_idx * block_w, usable_h))

    atlas_img.save(tex_path, quality=95, subsampling=0)
    mtl_path.write_text(f"newmtl lidup_mat\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\nd 1.0\nmap_Kd {tex_path.name}\n", encoding="utf-8")

    # Emite Wavefront OBJ (vt flip Y porque OBJ usa origem embaixo-esquerda e PIL usa topo-esquerda)
    vt_lines: list[str] = []
    tri_vt = np.zeros((len(tris), 3), dtype=np.int64)
    vt_counter = 1

    for chart in charts:
        u0, v0, u1, v1 = chart.crop_box
        ax, ay, aw, ah = chart.atlas_rect
        cw = float(max(u1 - u0, 1))
        ch = float(max(v1 - v0, 1))
        local_map: dict[int, int] = {}
        for vi, upx, vpx in zip(chart.vert_indices, chart.u_px, chart.v_px):
            u_norm = (ax + ((upx - u0) / cw) * aw) / float(atlas_size)
            y_atlas = ay + ((vpx - v0) / ch) * ah
            v_norm = 1.0 - (y_atlas / float(atlas_size))
            vt_lines.append(f"vt {u_norm:.6f} {v_norm:.6f}\n")
            local_map[int(vi)] = vt_counter
            vt_counter += 1

        for ti in chart.tri_indices:
            a, b, c = tris[ti]
            tri_vt[ti, 0] = local_map[int(a)]
            tri_vt[ti, 1] = local_map[int(b)]
            tri_vt[ti, 2] = local_map[int(c)]

    # Fallback UVs para triângulos sem câmera visível
    miss_t = np.nonzero(best_frame < 0)[0]
    if len(miss_t) > 0:
        fallback_vt: dict[int, int] = {}
        for ti in miss_t:
            col = np.clip(v_cols[tris[ti]].mean(axis=0), 0.0, 1.0)
            rq = min(3, int(round(col[0] * 3.0)))
            gq = min(3, int(round(col[1] * 3.0)))
            bq = min(3, int(round(col[2] * 3.0)))
            q_idx = (rq << 4) | (gq << 2) | bq
            if q_idx not in fallback_vt:
                cx_f = (q_idx * block_w + block_w * 0.5) / float(atlas_size)
                cy_f = 1.0 - ((usable_h + fallback_strip_h * 0.5) / float(atlas_size))
                vt_lines.append(f"vt {cx_f:.6f} {cy_f:.6f}\n")
                fallback_vt[q_idx] = vt_counter
                vt_counter += 1
            vt_id = fallback_vt[q_idx]
            tri_vt[ti, :] = vt_id

    with out_obj.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_path.name}\nusemtl lidup_mat\n")
        for vx, vy, vz in verts_export_use:
            f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for nx, ny, nz in v_norms_export:
            f.write(f"vn {nx:.5f} {ny:.5f} {nz:.5f}\n")
        f.writelines(vt_lines)
        for ti, (a, b, c) in enumerate(tris):
            i0, i1, i2 = int(a) + 1, int(b) + 1, int(c) + 1
            t0, t1, t2 = int(tri_vt[ti, 0]), int(tri_vt[ti, 1]), int(tri_vt[ti, 2])
            f.write(f"f {i0}/{t0}/{i0} {i1}/{t1}/{i1} {i2}/{t2}/{i2}\n")

    return out_obj
