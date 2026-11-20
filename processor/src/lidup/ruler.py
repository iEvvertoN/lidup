from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import open3d as o3d


def compute_ruler_segments(
    picked_points: list[np.ndarray] | np.ndarray,
    *,
    up_axis: str = "y",
) -> list[dict[str, float | int]]:
    """Calcula distâncias 3D, horizontais, desnível e inclinação (%) entre pontos consecutivos da régua."""
    pts = np.asarray(picked_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return []

    axis_idx = 2 if up_axis.strip().lower() == "z" else 1
    horiz_axes = [i for i in range(3) if i != axis_idx]

    segments: list[dict[str, float | int]] = []
    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        delta = p2 - p1
        dist_3d = float(np.linalg.norm(delta))
        dist_horiz = float(math.hypot(float(delta[horiz_axes[0]]), float(delta[horiz_axes[1]])))
        dv = float(delta[axis_idx])
        slope_pct = (abs(dv) / dist_horiz * 100.0) if dist_horiz > 1e-6 else 0.0
        segments.append(
            {
                "segment_index": i + 1,
                "from_index": i,
                "to_index": i + 1,
                "dist_3d_m": round(dist_3d, 4),
                "dist_horiz_m": round(dist_horiz, 4),
                "dx_m": round(float(delta[0]), 4),
                "dy_m": round(float(delta[1]), 4),
                "dz_m": round(float(delta[2]), 4),
                "dv_up_m": round(dv, 4),
                "slope_pct": round(slope_pct, 2),
            }
        )
    return segments


def compute_pixel_ray(
    cam: Any,
    lx: float,
    ly: float,
    width: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Constrói o raio 3D (origem, direção unitária) para o pixel (lx, ly).

    Em Open3D/Filament, z=1.0 mapeia no plano infinito e retorna [nan, nan, -inf];
    por isso usamos z_mid finitos (0.85 -> fallback 0.5) junto a z_near=0.0.
    """
    w = float(width)
    h = float(height)
    if w <= 1.0 or h <= 1.0:
        return None

    p_near = np.asarray(cam.unproject(float(lx), float(ly), 0.0, w, h), dtype=np.float64).ravel()
    if p_near.shape != (3,) or not np.all(np.isfinite(p_near)):
        return None

    p_mid = None
    for z_probe in (0.85, 0.5, 0.25):
        cand = np.asarray(cam.unproject(float(lx), float(ly), z_probe, w, h), dtype=np.float64).ravel()
        if cand.shape == (3,) and np.all(np.isfinite(cand)) and np.linalg.norm(cand - p_near) > 1e-6:
            p_mid = cand
            break

    if p_mid is None:
        return None

    ray_dir = p_mid - p_near
    dn = float(np.linalg.norm(ray_dir))
    if dn < 1e-9:
        return None
    return p_near, ray_dir / dn


def pick_point_on_ray(
    points: np.ndarray,
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    *,
    cone_rad: float = 0.018,
) -> np.ndarray | None:
    """Seleciona o ponto 3D da nuvem mais próximo ao longo do raio clicado pelo mouse."""
    if len(points) == 0:
        return None
    o = np.asarray(ray_origin, dtype=np.float64)
    d = np.asarray(ray_dir, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None
    d = d / norm

    rel = points - o
    t = rel @ d
    front = t > 0.02
    if not np.any(front):
        return None

    rel_f = rel[front]
    t_f = t[front]
    pts_f = points[front]

    proj = np.outer(t_f, d)
    perp = np.linalg.norm(rel_f - proj, axis=1)

    max_perp = np.maximum(0.020, float(cone_rad) * t_f)
    in_cone = perp <= max_perp
    if np.any(in_cone):
        cand_idx = np.nonzero(in_cone)[0]
        score = t_f[cand_idx] + perp[cand_idx] * 2.5
        best = int(cand_idx[int(np.argmin(score))])
        return pts_f[best].copy()

    # Fallback angular para nuvens bem esparsas
    ang = perp / np.maximum(t_f, 0.1)
    best_idx = int(np.argmin(ang))
    if float(ang[best_idx]) <= 0.08:
        return pts_f[best_idx].copy()
    return None


def raycast_geometry(
    geom: o3d.geometry.Geometry,
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    *,
    raycaster: Any = None,
    pts_cache: np.ndarray | None = None,
) -> np.ndarray | None:
    """Intersepta o raio com TriangleMesh (RaycastingScene) ou PointCloud."""
    o = np.asarray(ray_origin, dtype=np.float64)
    d = np.asarray(ray_dir, dtype=np.float64)
    dn = float(np.linalg.norm(d))
    if dn < 1e-9 or not np.all(np.isfinite(o)) or not np.all(np.isfinite(d)):
        return None
    d = d / dn

    if raycaster is not None:
        rays = o3d.core.Tensor(
            [[float(o[0]), float(o[1]), float(o[2]), float(d[0]), float(d[1]), float(d[2])]],
            dtype=o3d.core.Dtype.Float32,
        )
        ans = raycaster.cast_rays(rays)
        t_hit = float(ans["t_hit"].numpy()[0])
        if math.isfinite(t_hit) and 0.02 < t_hit < 500.0:
            return o + t_hit * d

    pts = pts_cache
    if pts is None:
        if isinstance(geom, o3d.geometry.TriangleMesh):
            pts = np.asarray(geom.vertices, dtype=np.float64)
        elif isinstance(geom, o3d.geometry.PointCloud):
            pts = np.asarray(geom.points, dtype=np.float64)
        else:
            pts = np.empty((0, 3), dtype=np.float64)
    return pick_point_on_ray(pts, o, d)


def _rodrigues(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    k = np.asarray(axis, dtype=np.float64)
    kn = float(np.linalg.norm(k))
    if kn < 1e-9:
        return vec
    k = k / kn
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return vec * c + np.cross(k, vec) * s + k * (float(np.dot(k, vec)) * (1.0 - c))


def step_camera_pose(
    view_matrix: np.ndarray,
    action: str,
    *,
    step_m: float = 0.25,
    turn_rad: float = 0.08,
    pivot_dist: float = 2.0,
    up_axis: str = "y",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula nova pose (center, eye, up) intuitiva a partir da view_matrix atual.

    Ações de translação: forward, backward, left, right, up, down.
    Ações de olhar/girar: yaw_left, yaw_right, pitch_up, pitch_down.
    """
    vm = np.asarray(view_matrix, dtype=np.float64)
    R = vm[:3, :3]
    t = vm[:3, 3]
    eye = -R.T @ t
    right = R[0]
    cam_up = R[1]
    forward = -R[2]

    world_up = np.array([0.0, 0.0, 1.0] if up_axis.strip().lower() == "z" else [0.0, 1.0, 0.0], dtype=np.float64)
    pdist = max(0.5, float(pivot_dist))

    new_eye = eye.copy()
    new_fwd = forward.copy()
    new_up = cam_up.copy()

    act = action.strip().lower()
    if act == "forward":
        new_eye = eye + step_m * forward
    elif act == "backward":
        new_eye = eye - step_m * forward
    elif act == "right":
        new_eye = eye + step_m * right
    elif act == "left":
        new_eye = eye - step_m * right
    elif act == "up":
        new_eye = eye + step_m * world_up
    elif act == "down":
        new_eye = eye - step_m * world_up
    elif act == "yaw_left":
        new_fwd = _rodrigues(forward, world_up, turn_rad)
        new_up = _rodrigues(cam_up, world_up, turn_rad)
    elif act == "yaw_right":
        new_fwd = _rodrigues(forward, world_up, -turn_rad)
        new_up = _rodrigues(cam_up, world_up, -turn_rad)
    elif act in {"pitch_up", "pitch_down"}:
        ang = turn_rad if act == "pitch_up" else -turn_rad
        cand_fwd = _rodrigues(forward, right, ang)
        if abs(float(np.dot(cand_fwd, world_up))) < 0.96:
            new_fwd = cand_fwd
            cand_up = np.cross(right, new_fwd)
            un = float(np.linalg.norm(cand_up))
            if un > 1e-9:
                new_up = cand_up / un

    fn = float(np.linalg.norm(new_fwd))
    if fn > 1e-9:
        new_fwd = new_fwd / fn
    un = float(np.linalg.norm(new_up))
    if un > 1e-9:
        new_up = new_up / un

    new_center = new_eye + pdist * new_fwd
    return new_center, new_eye, new_up


def preset_camera_pose(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    preset: str,
    *,
    up_axis: str = "y",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna (center, eye, up) alinhado ao eixo vertical do modelo."""
    bmin = np.asarray(bbox_min, dtype=np.float64)
    bmax = np.asarray(bbox_max, dtype=np.float64)
    center = 0.5 * (bmin + bmax)
    extent = np.maximum(bmax - bmin, 0.25)
    dist = max(1.2, float(np.linalg.norm(extent)) * 1.15)
    z_up = up_axis.strip().lower() == "z"
    p = preset.strip().lower()

    if z_up:
        up_default = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        offsets = {
            "front": (np.array([0.0, -dist, 0.12 * dist]), up_default),
            "back": (np.array([0.0, dist, 0.12 * dist]), up_default),
            "right": (np.array([dist, 0.0, 0.12 * dist]), up_default),
            "left": (np.array([-dist, 0.0, 0.12 * dist]), up_default),
            "top": (np.array([0.0, -0.001 * dist, dist]), np.array([0.0, 1.0, 0.0])),
            "iso": (np.array([0.65 * dist, -0.75 * dist, 0.45 * dist]), up_default),
        }
    else:
        up_default = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        offsets = {
            "front": (np.array([0.0, 0.12 * dist, dist]), up_default),
            "back": (np.array([0.0, 0.12 * dist, -dist]), up_default),
            "right": (np.array([dist, 0.12 * dist, 0.0]), up_default),
            "left": (np.array([-dist, 0.12 * dist, 0.0]), up_default),
            "top": (np.array([0.0, dist, 0.001 * dist]), np.array([0.0, 0.0, -1.0])),
            "iso": (np.array([0.65 * dist, 0.45 * dist, 0.75 * dist]), up_default),
        }

    off, up = offsets.get(p, offsets["iso"])
    return center, center + off, up


def _height_colors(points: np.ndarray, up_axis: str) -> np.ndarray:
    axis_idx = 2 if up_axis.strip().lower() == "z" else 1
    vals = points[:, axis_idx]
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    norm = np.clip((vals - lo) / max(hi - lo, 1e-4), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)
    return np.column_stack([r, g, b])


def run_interactive_ruler(
    geom: o3d.geometry.Geometry,
    *,
    title: str = "LidUp 3D",
    up_axis: str = "y",
    point_size: float = 3.5,
    show_axes: bool = False,
    measure_default: bool = True,
    start_no_color: bool = False,
) -> list[dict[str, float | int]]:
    """
    Visualizador 3D moderno unificado + Régua a Laser interativa:
    - Clique curto esquerdo marca pontos P1 -> P2 (quando Régua 1-Clique ativa) sem impedir arrasto
    - Botão de alternância Orbitar (Mouse) <-> Voo Livre (WASD) + presets (Frente/Trás/Esq/Dir/Topo/Centrar)
    - W/S/A/D + E/X (subir/descer) + Setas (olhar) + +/- (velocidade) + C (cor) + Z (desfazer) + Q (sair)
    """
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()

    geom_work = copy.deepcopy(geom)
    is_mesh = isinstance(geom_work, o3d.geometry.TriangleMesh)
    pts_ref = (
        np.asarray(geom_work.vertices, dtype=np.float64)
        if is_mesh
        else np.asarray(geom_work.points, dtype=np.float64)
    )
    has_orig_textures = is_mesh and geom_work.has_textures() and len(geom_work.textures) > 0
    orig_textures = list(geom_work.textures) if has_orig_textures else []
    if is_mesh:
        orig_cols = (
            np.asarray(geom_work.vertex_colors, dtype=np.float64).copy()
            if geom_work.has_vertex_colors()
            else np.full((len(pts_ref), 3), 0.78, dtype=np.float64)
        )
    else:
        orig_cols = (
            np.asarray(geom_work.colors, dtype=np.float64).copy()
            if geom_work.has_colors()
            else np.full((len(pts_ref), 3), 0.78, dtype=np.float64)
        )

    elev_cols = _height_colors(pts_ref, up_axis) if len(pts_ref) > 0 else np.empty((0, 3))
    gray_cols = np.full((len(pts_ref), 3), 0.78, dtype=np.float64)
    palettes = [orig_cols, gray_cols, elev_cols]
    palette_names = ["Original", "Cinza", "Altura"]
    color_state = {"mode": 1 if start_no_color else 0}

    bbox = geom_work.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(bbox.get_extent())) if not bbox.is_empty() else 2.0
    marker_radius = max(0.004, min(0.014, diag * 0.0032))

    raycaster = None
    if is_mesh and len(geom_work.triangles) > 0:
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(geom_work)
        raycaster = o3d.t.geometry.RaycastingScene()
        raycaster.add_triangles(t_mesh)

    win_title = f"{title} [{'Régua 3D' if measure_default else 'Visor 3D'}]"
    win = app.create_window(win_title, 1340, 860)
    em = win.theme.font_size

    scene_w = gui.SceneWidget()
    scene_w.scene = rendering.Open3DScene(win.renderer)
    scene_w.scene.set_background([0.05, 0.05, 0.07, 1.0])

    def _build_main_material(mode_idx: int) -> Any:
        mat = rendering.MaterialRecord()
        if is_mesh:
            mat.shader = "defaultLit"
            if mode_idx == 0 and orig_textures:
                geom_work.textures = orig_textures
                mat.albedo_img = o3d.geometry.Image(np.asarray(orig_textures[0]))
            else:
                geom_work.textures = []
        else:
            mat.shader = "defaultUnlit"
            mat.point_size = float(point_size) * win.scaling
        return mat

    def _apply_palette(mode_idx: int, *, update_scene: bool) -> None:
        if len(pts_ref) > 0 and not (mode_idx == 0 and orig_textures):
            vec = o3d.utility.Vector3dVector(palettes[mode_idx])
            if is_mesh:
                geom_work.vertex_colors = vec
            else:
                geom_work.colors = vec
        mat = _build_main_material(mode_idx)
        if update_scene:
            if scene_w.scene.has_geometry("main"):
                scene_w.scene.remove_geometry("main")
            scene_w.scene.add_geometry("main", geom_work, mat)
            scene_w.force_redraw()
        else:
            scene_w.scene.add_geometry("main", geom_work, mat)

    _apply_palette(color_state["mode"], update_scene=False)

    if show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.25, diag * 0.12))
        mat_ax = rendering.MaterialRecord()
        mat_ax.shader = "defaultLit"
        scene_w.scene.add_geometry("axes", axes, mat_ax)

    scene_w.setup_camera(60.0, bbox, bbox.get_center())
    scene_w.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    # HUD Superior em 2 linhas
    hud = gui.Vert(0.25 * em, gui.Margins(0.6 * em, 0.35 * em, 0.6 * em, 0.35 * em))
    hud.background_color = gui.Color(0.06, 0.06, 0.08, 1.0)

    row1 = gui.Horiz(0.45 * em)
    row2 = gui.Horiz(0.40 * em)

    chk_ruler = gui.Checkbox("Régua 1-Clique")
    chk_ruler.checked = bool(measure_default)

    combo_nav = gui.Combobox()
    combo_nav.add_item("Orbitar (Mouse)")
    combo_nav.add_item("Voo Livre (WASD)")
    combo_nav.selected_index = 0

    btn_undo = gui.Button("Desfazer (Z)")
    btn_clear = gui.Button("Limpar")
    btn_color = gui.Button(f"Cor: {palette_names[color_state['mode']]}")
    btn_close = gui.Button("Concluir (Q)")

    lbl_mode_hint = gui.Label(
        "Clique curto = marca ponto | Arrastar = gira | Scroll = zoom"
        if measure_default
        else "Arrastar = gira | WASD + E/X = anda | Setas = olha ao redor"
    )

    row1.add_child(chk_ruler)
    row1.add_fixed(0.4 * em)
    row1.add_child(combo_nav)
    row1.add_fixed(0.4 * em)
    row1.add_child(btn_undo)
    row1.add_child(btn_clear)
    row1.add_child(btn_color)
    row1.add_child(btn_close)
    row1.add_fixed(0.6 * em)
    row1.add_child(lbl_mode_hint)

    btn_front = gui.Button("Frente")
    btn_back = gui.Button("Trás")
    btn_left = gui.Button("Esq")
    btn_right = gui.Button("Dir")
    btn_top = gui.Button("Topo")
    btn_reset = gui.Button("Centrar (R)")
    btn_slower = gui.Button("- Vel")
    btn_faster = gui.Button("+ Vel")
    lbl_info = gui.Label(
        "Clique no objeto/nuvem para marcar P1 -> P2"
        if measure_default
        else "Ative 'Régua 1-Clique' a qualquer momento para medir distâncias em metros/cm"
    )

    for b in (btn_front, btn_back, btn_left, btn_right, btn_top, btn_reset, btn_slower, btn_faster):
        row2.add_child(b)
    row2.add_fixed(0.6 * em)
    row2.add_child(lbl_info)

    hud.add_child(row1)
    hud.add_child(row2)
    win.add_child(hud)
    win.add_child(scene_w)

    def _on_layout(_ctx: Any) -> None:
        r = win.content_rect
        hud_h = int(4.6 * em)
        hud.frame = gui.Rect(r.x, r.y, r.width, hud_h)
        scene_w.frame = gui.Rect(r.x, r.y + hud_h, r.width, max(10, r.height - hud_h))

    win.set_on_layout(_on_layout)

    nav_state = {"step_m": max(0.06, min(0.8, diag * 0.045))}
    picked_pts: list[np.ndarray] = []
    label_handles: list[Any] = []
    marker_names: list[str] = []
    drag_state = {"down": False, "x": 0.0, "y": 0.0}

    def _current_cam_eye() -> np.ndarray | None:
        try:
            vm = np.asarray(scene_w.scene.camera.get_view_matrix(), dtype=np.float64)
            return -vm[:3, :3].T @ vm[:3, 3]
        except Exception:
            return None

    def _rebuild_ruler_overlay() -> None:
        for h in label_handles:
            try:
                scene_w.remove_3d_label(h)
            except Exception:
                pass
        label_handles.clear()

        for mn in marker_names:
            if scene_w.scene.has_geometry(mn):
                scene_w.scene.remove_geometry(mn)
        marker_names.clear()

        if scene_w.scene.has_geometry("ruler_lines"):
            scene_w.scene.remove_geometry("ruler_lines")

        eye = _current_cam_eye()
        mat_dot = rendering.MaterialRecord()
        mat_dot.shader = "defaultUnlit"
        mat_dot.base_color = [1.0, 0.90, 0.18, 1.0]

        up_vec = np.array([0.0, 0.0, 1.0] if up_axis.strip().lower() == "z" else [0.0, 1.0, 0.0])
        display_pts: list[np.ndarray] = []
        for idx, pt in enumerate(picked_pts):
            if eye is not None:
                to_cam = eye - pt
                tc_n = float(np.linalg.norm(to_cam))
                nudge = (to_cam / tc_n) * min(0.015, marker_radius * 0.8) if tc_n > 1e-6 else up_vec * 0.005
            else:
                nudge = up_vec * 0.005
            display_pts.append(pt + nudge)

            sph = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius, resolution=12)
            sph.translate(pt)
            sph.paint_uniform_color([1.0, 0.90, 0.18])
            name = f"ruler_pt_{idx}"
            scene_w.scene.add_geometry(name, sph, mat_dot)
            marker_names.append(name)
            label_handles.append(scene_w.add_3d_label(pt + nudge * 1.8 + up_vec * (marker_radius * 1.4), f"P{idx + 1}"))

        segs = compute_ruler_segments(picked_pts, up_axis=up_axis)
        if segs:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.asarray(display_pts, dtype=np.float64))
            line_set.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(picked_pts) - 1)])
            line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.90, 0.18] for _ in segs])
            mat_line = rendering.MaterialRecord()
            mat_line.shader = "unlitLine"
            mat_line.line_width = 5.0 * win.scaling
            scene_w.scene.add_geometry("ruler_lines", line_set, mat_line)

            for i, seg in enumerate(segs):
                mid = 0.5 * (display_pts[i] + display_pts[i + 1])
                txt = f"{seg['dist_3d_m']:.3f} m" if seg["dist_3d_m"] >= 1.0 else f"{seg['dist_3d_m'] * 100.0:.1f} cm"
                label_handles.append(scene_w.add_3d_label(mid, txt))

            last = segs[-1]
            total = sum(float(s["dist_3d_m"]) for s in segs)
            lbl_info.text = (
                f"Último: {last['dist_3d_m']:.3f} m ({last['dist_3d_m'] * 100.0:.1f} cm) | "
                f"Horiz: {last['dist_horiz_m']:.3f} m | dV: {last['dv_up_m']:+.3f} m ({last['slope_pct']:.1f}%) | "
                f"Total ({len(segs)}): {total:.3f} m"
            )
        elif len(picked_pts) == 1:
            lbl_info.text = "P1 marcado — clique no próximo ponto (P2) para medir"
        else:
            lbl_info.text = (
                "Régua ativa: clique curto sobre o modelo/nuvem para marcar P1"
                if chk_ruler.checked
                else "Ative 'Régua 1-Clique' ou segure SHIFT+Clique para medir"
            )

        win.set_needs_layout()
        scene_w.force_redraw()

    def _apply_nav_mode(idx: int) -> None:
        if idx == 1:
            scene_w.set_view_controls(gui.SceneWidget.Controls.FLY)
            lbl_mode_hint.text = "Modo VOO: Arrastar = olhar ao redor | WASD + E/X = caminhar"
        else:
            scene_w.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
            lbl_mode_hint.text = (
                "Clique curto = marca ponto | Arrastar = gira | Scroll = zoom"
                if chk_ruler.checked
                else "Arrastar = gira | WASD + E/X = anda | Setas = olha ao redor"
            )
        win.set_needs_layout()

    def _on_chk_ruler(is_checked: bool) -> None:
        _apply_nav_mode(combo_nav.selected_index)
        if not picked_pts:
            _rebuild_ruler_overlay()

    chk_ruler.set_on_checked(_on_chk_ruler)
    combo_nav.set_on_selection_changed(lambda _txt, idx: _apply_nav_mode(idx))

    def _set_preset(preset: str) -> None:
        center, eye, up = preset_camera_pose(
            bbox.get_min_bound(), bbox.get_max_bound(), preset, up_axis=up_axis
        )
        scene_w.look_at(center, eye, up)
        scene_w.force_redraw()

    def _steer_camera(action: str) -> None:
        vm = np.asarray(scene_w.scene.camera.get_view_matrix(), dtype=np.float64)
        center, eye, up = step_camera_pose(
            vm,
            action,
            step_m=nav_state["step_m"],
            turn_rad=0.07,
            pivot_dist=max(0.8, diag * 0.35),
            up_axis=up_axis,
        )
        scene_w.look_at(center, eye, up)
        scene_w.force_redraw()

    def _cycle_color() -> None:
        color_state["mode"] = (color_state["mode"] + 1) % 3
        _apply_palette(color_state["mode"], update_scene=True)
        btn_color.text = f"Cor: {palette_names[color_state['mode']]}"
        win.set_needs_layout()

    def _undo() -> None:
        if picked_pts:
            picked_pts.pop()
            _rebuild_ruler_overlay()

    def _clear() -> None:
        if picked_pts:
            picked_pts.clear()
            _rebuild_ruler_overlay()

    def _adjust_speed(factor: float) -> None:
        nav_state["step_m"] = max(0.02, min(2.5, nav_state["step_m"] * factor))
        lbl_info.text = f"Passo de movimento: {nav_state['step_m']:.2f} m"
        win.set_needs_layout()

    btn_undo.set_on_clicked(_undo)
    btn_clear.set_on_clicked(_clear)
    btn_color.set_on_clicked(_cycle_color)
    btn_close.set_on_clicked(win.close)
    btn_front.set_on_clicked(lambda: _set_preset("front"))
    btn_back.set_on_clicked(lambda: _set_preset("back"))
    btn_left.set_on_clicked(lambda: _set_preset("left"))
    btn_right.set_on_clicked(lambda: _set_preset("right"))
    btn_top.set_on_clicked(lambda: _set_preset("top"))
    btn_reset.set_on_clicked(lambda: (scene_w.setup_camera(60.0, bbox, bbox.get_center()), scene_w.force_redraw()))
    btn_slower.set_on_clicked(lambda: _adjust_speed(0.7))
    btn_faster.set_on_clicked(lambda: _adjust_speed(1.4))

    def _try_pick_at(x_win: float, y_win: float) -> None:
        fr = scene_w.frame
        lx = float(x_win - fr.x)
        ly = float(y_win - fr.y)
        if lx < 0 or ly < 0 or lx > fr.width or ly > fr.height:
            return
        ray = compute_pixel_ray(scene_w.scene.camera, lx, ly, float(fr.width), float(fr.height))
        if ray is None:
            lbl_info.text = "Câmera sem projeção válida para clique"
            win.set_needs_layout()
            return
        ray_origin, ray_dir = ray
        hit = raycast_geometry(geom_work, ray_origin, ray_dir, raycaster=raycaster, pts_cache=pts_ref)
        if hit is not None:
            picked_pts.append(hit)
            _rebuild_ruler_overlay()
        else:
            lbl_info.text = "Clique fora da superfície 3D — aponte diretamente para o modelo/nuvem"
            win.set_needs_layout()

    def _on_mouse(event: Any) -> Any:
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
            drag_state["down"] = True
            drag_state["x"] = float(event.x)
            drag_state["y"] = float(event.y)
            return gui.Widget.EventCallbackResult.IGNORED

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            drag_state["down"] = False
            return gui.Widget.EventCallbackResult.IGNORED

        if event.type == gui.MouseEvent.Type.BUTTON_UP and drag_state["down"]:
            drag_state["down"] = False
            moved = math.hypot(float(event.x) - drag_state["x"], float(event.y) - drag_state["y"])
            wants_pick = chk_ruler.checked or event.is_modifier_down(gui.KeyModifier.SHIFT)
            if moved <= 5.0 and wants_pick:
                _try_pick_at(float(event.x), float(event.y))
                return gui.Widget.EventCallbackResult.CONSUMED

        return gui.Widget.EventCallbackResult.IGNORED

    def _on_key(event: Any) -> Any:
        if event.type != gui.KeyEvent.Type.DOWN:
            return False
        key_map = {
            gui.KeyName.W: lambda: _steer_camera("forward"),
            gui.KeyName.S: lambda: _steer_camera("backward"),
            gui.KeyName.A: lambda: _steer_camera("left"),
            gui.KeyName.D: lambda: _steer_camera("right"),
            gui.KeyName.E: lambda: _steer_camera("up"),
            gui.KeyName.PAGE_UP: lambda: _steer_camera("up"),
            gui.KeyName.X: lambda: _steer_camera("down"),
            gui.KeyName.PAGE_DOWN: lambda: _steer_camera("down"),
            gui.KeyName.LEFT: lambda: _steer_camera("yaw_left"),
            gui.KeyName.RIGHT: lambda: _steer_camera("yaw_right"),
            gui.KeyName.UP: lambda: _steer_camera("pitch_up"),
            gui.KeyName.DOWN: lambda: _steer_camera("pitch_down"),
            gui.KeyName.R: lambda: (scene_w.setup_camera(60.0, bbox, bbox.get_center()), scene_w.force_redraw()),
            gui.KeyName.C: _cycle_color,
            gui.KeyName.Z: _undo,
            gui.KeyName.BACKSPACE: _undo,
            gui.KeyName.EQUALS: lambda: _adjust_speed(1.4),
            gui.KeyName.PLUS: lambda: _adjust_speed(1.4),
            gui.KeyName.MINUS: lambda: _adjust_speed(0.7),
            gui.KeyName.ONE: lambda: _set_preset("front"),
            gui.KeyName.TWO: lambda: _set_preset("right"),
            gui.KeyName.THREE: lambda: _set_preset("top"),
            gui.KeyName.FOUR: lambda: _set_preset("iso"),
        }
        if event.key in (gui.KeyName.Q, gui.KeyName.ESCAPE):
            win.close()
            return True
        fn = key_map.get(event.key)
        if fn is not None:
            fn()
            return True
        return False

    scene_w.set_on_mouse(_on_mouse)
    win.set_on_key(_on_key)
    app.run()

    return compute_ruler_segments(picked_pts, up_axis=up_axis)
