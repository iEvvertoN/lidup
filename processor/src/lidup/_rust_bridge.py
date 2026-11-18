from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

_LIB: ctypes.CDLL | None = None
_TRIED = False


def _candidate_lib_paths() -> list[Path]:
    if sys.platform.startswith("win"):
        names = ["lidup_core.dll"]
    elif sys.platform == "darwin":
        names = ["liblidup_core.dylib"]
    else:
        names = ["liblidup_core.so"]

    here = Path(__file__).resolve()
    repo_root = here.parents[3] if len(here.parents) > 3 else here.parent
    roots = [
        here.parent,
        repo_root / "core" / "lidup_core" / "target" / "release",
        repo_root / "core" / "lidup_core" / "target" / "debug",
    ]
    return [r / n for r in roots for n in names]


def _load_rust_lib() -> ctypes.CDLL | None:
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True

    c_f32_p = ctypes.POINTER(ctypes.c_float)
    c_f64_p = ctypes.POINTER(ctypes.c_double)
    c_u8_p = ctypes.POINTER(ctypes.c_uint8)
    c_u16_p = ctypes.POINTER(ctypes.c_uint16)
    c_i32_p = ctypes.POINTER(ctypes.c_int32)
    c_i64_p = ctypes.POINTER(ctypes.c_int64)
    c_f32_pp = ctypes.POINTER(c_f32_p)

    for cand in _candidate_lib_paths():
        if not cand.exists():
            continue
        try:
            lib = ctypes.CDLL(str(cand))
            lib.lidup_core_abi_version.argtypes = []
            lib.lidup_core_abi_version.restype = ctypes.c_uint32
            if int(lib.lidup_core_abi_version()) != 2:
                continue

            lib.lidup_depth_float32_to_mm_u16.argtypes = [
                c_u8_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, c_u16_p
            ]
            lib.lidup_depth_float32_to_mm_u16.restype = ctypes.c_int32

            lib.lidup_sharpness_laplacian_u8.argtypes = [
                c_u8_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32
            ]
            lib.lidup_sharpness_laplacian_u8.restype = ctypes.c_float

            lib.lidup_upsample_depth_guided_f32.argtypes = [
                c_f32_p, c_u8_p, c_u8_p, ctypes.c_int32, c_f32_p, c_f32_p,
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                ctypes.c_float, c_f32_p, c_u8_p,
            ]
            lib.lidup_upsample_depth_guided_f32.restype = ctypes.c_int32

            lib.lidup_extract_depth_points_f64.argtypes = [
                c_f64_p, c_u8_p, ctypes.c_int32, ctypes.c_int32,
                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_double, ctypes.c_double,
                c_u8_p, c_f64_p, c_f64_p, c_i64_p, c_i64_p, c_i32_p,
            ]
            lib.lidup_extract_depth_points_f64.restype = ctypes.c_int32

            lib.lidup_resize_rgb_u8_to_f32.argtypes = [
                c_u8_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, c_f32_p
            ]
            lib.lidup_resize_rgb_u8_to_f32.restype = ctypes.c_int32

            lib.lidup_warp_rgb_to_depth_f32.argtypes = [
                c_f32_p, c_f32_p, c_f64_p,
                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_int32, ctypes.c_int32, c_f32_p,
            ]
            lib.lidup_warp_rgb_to_depth_f32.restype = ctypes.c_int32

            lib.lidup_colorize_hires_frame_u8.argtypes = [
                c_f64_p, c_f64_p, ctypes.c_int32, c_f64_p, c_u8_p,
                ctypes.c_int32, ctypes.c_int32, c_f32_p, ctypes.c_int32, ctypes.c_int32,
                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_double, ctypes.c_double, ctypes.c_double, c_f64_p, c_f64_p,
            ]
            lib.lidup_colorize_hires_frame_u8.restype = ctypes.c_int32

            lib.lidup_mvs_sweep_f32.argtypes = [
                c_f32_p, c_u8_p, c_f32_p, ctypes.c_int32, ctypes.c_int32,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_int32, c_f32_pp, c_f32_pp, c_f32_p, c_i32_p, c_f32_p,
                ctypes.c_float, ctypes.c_int32, c_f32_p, c_f32_p, c_f32_p,
            ]
            lib.lidup_mvs_sweep_f32.restype = ctypes.c_int32

            lib.lidup_csf_cloth_filter_f64.argtypes = [
                c_f64_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_double,
                ctypes.c_int32, ctypes.c_int32, ctypes.c_double, c_f64_p,
            ]
            lib.lidup_csf_cloth_filter_f64.restype = ctypes.c_int32

            _LIB = lib
            return _LIB
        except OSError:
            continue
    return None


def rust_available() -> bool:
    return _load_rust_lib() is not None


def rust_depth_float32_to_mm_u16(raw_f32_bytes: bytes, stride_bytes: int, width: int, height: int) -> np.ndarray | None:
    lib = _load_rust_lib()
    if lib is None:
        return None
    src = np.frombuffer(raw_f32_bytes, dtype=np.uint8)
    out = np.empty((height, width), dtype=np.uint16)
    rc = int(
        lib.lidup_depth_float32_to_mm_u16(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(len(src)),
            ctypes.c_int32(stride_bytes),
            ctypes.c_int32(width),
            ctypes.c_int32(height),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        )
    )
    return out if rc == 0 else None


def rust_sharpness_laplacian_u8(y_plane_u8: np.ndarray, sample_step: int = 2) -> float | None:
    lib = _load_rust_lib()
    if lib is None:
        return None
    y_c = np.ascontiguousarray(y_plane_u8, dtype=np.uint8)
    h, w = y_c.shape[:2]
    return float(
        lib.lidup_sharpness_laplacian_u8(
            y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(y_c.nbytes),
            ctypes.c_int32(w),
            ctypes.c_int32(w),
            ctypes.c_int32(h),
            ctypes.c_int32(sample_step),
        )
    )


def rust_resize_rgb_u8_to_f64(src_rgb_u8: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray | None:
    lib = _load_rust_lib()
    if lib is None:
        return None
    src = np.ascontiguousarray(src_rgb_u8, dtype=np.uint8)
    sh, sw = src.shape[:2]
    out = np.empty((dst_h, dst_w, 3), dtype=np.float32)
    rc = int(
        lib.lidup_resize_rgb_u8_to_f32(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(sw),
            ctypes.c_int32(sh),
            ctypes.c_int32(dst_w),
            ctypes.c_int32(dst_h),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
    )
    return out.astype(np.float64) if rc == 0 else None


def rust_warp_rgb_to_depth(
    rgb_hi: np.ndarray,
    d_hi_filled: np.ndarray,
    rel_mat4x4: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray | None:
    lib = _load_rust_lib()
    if lib is None:
        return None
    h, w = rgb_hi.shape[:2]
    r_f32 = np.ascontiguousarray(rgb_hi, dtype=np.float32)
    d_f32 = np.ascontiguousarray(d_hi_filled, dtype=np.float32)
    mat_f64 = np.ascontiguousarray(rel_mat4x4, dtype=np.float64).reshape(16)
    out = np.empty((h, w, 3), dtype=np.float32)
    rc = int(
        lib.lidup_warp_rgb_to_depth_f32(
            r_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            d_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mat_f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(fx),
            ctypes.c_double(fy),
            ctypes.c_double(cx),
            ctypes.c_double(cy),
            ctypes.c_int32(w),
            ctypes.c_int32(h),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
    )
    return out.astype(np.float64) if rc == 0 else None


def rust_extract_depth_points(
    depth_m: np.ndarray,
    in_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    edge_jump_m: float,
    cos_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    lib = _load_rust_lib()
    if lib is None:
        return None
    h, w = depth_m.shape[:2]
    n_pix = w * h
    d_f64 = np.ascontiguousarray(depth_m, dtype=np.float64)
    m_u8 = np.ascontiguousarray(in_mask.astype(np.uint8))

    out_mask = np.empty((h, w), dtype=np.uint8)
    out_pts = np.empty((n_pix, 3), dtype=np.float64)
    out_wts = np.empty(n_pix, dtype=np.float64)
    out_v = np.empty(n_pix, dtype=np.int64)
    out_u = np.empty(n_pix, dtype=np.int64)
    out_cnt = ctypes.c_int32(0)

    rc = int(
        lib.lidup_extract_depth_points_f64(
            d_f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            m_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(w),
            ctypes.c_int32(h),
            ctypes.c_double(fx),
            ctypes.c_double(fy),
            ctypes.c_double(cx),
            ctypes.c_double(cy),
            ctypes.c_double(edge_jump_m),
            ctypes.c_double(cos_thresh),
            out_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            out_pts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_wts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_v.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            out_u.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.byref(out_cnt),
        )
    )
    if rc != 0:
        return None
    cnt = int(out_cnt.value)
    return out_mask.astype(bool), out_pts[:cnt].copy(), out_wts[:cnt].copy(), out_v[:cnt].copy(), out_u[:cnt].copy()


def rust_colorize_hires_frame(
    pts_arkit: np.ndarray,
    norms_arkit: np.ndarray | None,
    pose_rgb: np.ndarray,
    hi_rgb_u8: np.ndarray,
    depth_f32: np.ndarray,
    fx_hi: float,
    fy_hi: float,
    cx_hi: float,
    cy_hi: float,
    max_depth_m: float,
    occ_tol_m: float,
    sharp_imu_w: float,
    best_score: np.ndarray,
    out_colors: np.ndarray,
) -> bool:
    lib = _load_rust_lib()
    if lib is None:
        return False
    n_pts = len(pts_arkit)
    hi_h, hi_w = hi_rgb_u8.shape[:2]
    d_h, d_w = depth_f32.shape[:2]

    pts_c = np.ascontiguousarray(pts_arkit, dtype=np.float64)
    norms_c = np.ascontiguousarray(norms_arkit, dtype=np.float64) if (norms_arkit is not None and len(norms_arkit) == n_pts) else None
    pose_c = np.ascontiguousarray(pose_rgb, dtype=np.float64).reshape(16)
    hi_c = np.ascontiguousarray(hi_rgb_u8, dtype=np.uint8)
    dep_c = np.ascontiguousarray(depth_f32, dtype=np.float32)

    rc = int(
        lib.lidup_colorize_hires_frame_u8(
            pts_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            norms_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if norms_c is not None else ctypes.POINTER(ctypes.c_double)(),
            ctypes.c_int32(n_pts),
            pose_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            hi_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(hi_w),
            ctypes.c_int32(hi_h),
            dep_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(d_w),
            ctypes.c_int32(d_h),
            ctypes.c_double(fx_hi),
            ctypes.c_double(fy_hi),
            ctypes.c_double(cx_hi),
            ctypes.c_double(cy_hi),
            ctypes.c_double(max_depth_m),
            ctypes.c_double(occ_tol_m),
            ctypes.c_double(sharp_imu_w),
            best_score.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_colors.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    return rc == 0


def rust_upsample_depth_guided(
    depth_m: np.ndarray,
    valid_lo: np.ndarray,
    conf: np.ndarray | None,
    rgb_lo: np.ndarray,
    rgb_hi: np.ndarray,
    scale: int,
    edge_jump_m: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    lib = _load_rust_lib()
    if lib is None:
        return None

    d_h, d_w = depth_m.shape[:2]
    u_h, u_w = rgb_hi.shape[:2]

    d_f32 = np.ascontiguousarray(depth_m, dtype=np.float32)
    v_u8 = np.ascontiguousarray(valid_lo.astype(np.uint8))
    has_conf = 1 if (conf is not None and conf.shape[:2] == (d_h, d_w)) else 0
    c_u8 = np.ascontiguousarray(conf.astype(np.uint8)) if has_conf else np.empty((1,), dtype=np.uint8)
    rlo_f32 = np.ascontiguousarray(rgb_lo, dtype=np.float32)
    rhi_f32 = np.ascontiguousarray(rgb_hi, dtype=np.float32)

    out_d = np.empty((u_h, u_w), dtype=np.float32)
    out_m = np.empty((u_h, u_w), dtype=np.uint8)

    rc = int(
        lib.lidup_upsample_depth_guided_f32(
            d_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            c_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(has_conf),
            rlo_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rhi_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(d_w),
            ctypes.c_int32(d_h),
            ctypes.c_int32(u_w),
            ctypes.c_int32(u_h),
            ctypes.c_int32(scale),
            ctypes.c_float(edge_jump_m),
            out_d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_m.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
    )
    if rc != 0:
        return None
    return out_d.astype(np.float64), out_m.astype(bool)


def rust_mvs_sweep(
    d0: np.ndarray,
    valid: np.ndarray,
    ref_nhp: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    nb_nhp_list: list[np.ndarray],
    nb_depth_list: list[np.ndarray],
    nb_t_mats: list[np.ndarray],
    nb_dims: list[tuple[int, int]],
    nb_intr: list[tuple[float, float, float, float]],
    max_shift_m: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    lib = _load_rust_lib()
    if lib is None or not nb_nhp_list:
        return None

    d_h, d_w = d0.shape[:2]
    d0_f32 = np.ascontiguousarray(d0, dtype=np.float32)
    valid_u8 = np.ascontiguousarray(valid.astype(np.uint8))
    ref_f32 = np.ascontiguousarray(ref_nhp, dtype=np.float32)

    nn = len(nb_nhp_list)
    nhp_keep = [np.ascontiguousarray(arr, dtype=np.float32) for arr in nb_nhp_list]
    dep_keep = [np.ascontiguousarray(arr, dtype=np.float32) for arr in nb_depth_list]

    c_f32_p = ctypes.POINTER(ctypes.c_float)
    nhp_ptrs = (c_f32_p * nn)(*[arr.ctypes.data_as(c_f32_p) for arr in nhp_keep])
    dep_ptrs = (c_f32_p * nn)(*[arr.ctypes.data_as(c_f32_p) for arr in dep_keep])
    mats_f32 = np.ascontiguousarray(np.asarray(nb_t_mats, dtype=np.float32).reshape(nn, 16))
    dims_i32 = np.ascontiguousarray(np.asarray(nb_dims, dtype=np.int32).reshape(nn, 2))
    intr_f32 = np.ascontiguousarray(np.asarray(nb_intr, dtype=np.float32).reshape(nn, 4))

    out_dz = np.empty((d_h, d_w), dtype=np.float32)
    out_sc = np.empty((d_h, d_w), dtype=np.float32)
    out_vis = np.empty((d_h, d_w), dtype=np.float32)

    rc = int(
        lib.lidup_mvs_sweep_f32(
            d0_f32.ctypes.data_as(c_f32_p),
            valid_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ref_f32.ctypes.data_as(c_f32_p),
            ctypes.c_int32(d_w),
            ctypes.c_int32(d_h),
            ctypes.c_float(fx),
            ctypes.c_float(fy),
            ctypes.c_float(cx),
            ctypes.c_float(cy),
            ctypes.c_int32(nn),
            nhp_ptrs,
            dep_ptrs,
            mats_f32.ctypes.data_as(c_f32_p),
            dims_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            intr_f32.ctypes.data_as(c_f32_p),
            ctypes.c_float(max_shift_m),
            ctypes.c_int32(n_steps),
            out_dz.ctypes.data_as(c_f32_p),
            out_sc.ctypes.data_as(c_f32_p),
            out_vis.ctypes.data_as(c_f32_p),
        )
    )
    if rc != 0:
        return None
    return out_dz.astype(np.float64), out_sc.astype(np.float64), out_vis.astype(np.float64)


def rust_csf_cloth_filter(
    terrain_filled: np.ndarray,
    z_inv_max: float,
    rigidness: int,
    iterations: int,
    gravity_step_m: float,
) -> np.ndarray | None:
    lib = _load_rust_lib()
    if lib is None:
        return None
    ny, nx = terrain_filled.shape[:2]
    tf_c = np.ascontiguousarray(terrain_filled, dtype=np.float64)
    out = np.empty((ny, nx), dtype=np.float64)
    rc = int(
        lib.lidup_csf_cloth_filter_f64(
            tf_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int32(nx),
            ctypes.c_int32(ny),
            ctypes.c_double(z_inv_max),
            ctypes.c_int32(rigidness),
            ctypes.c_int32(iterations),
            ctypes.c_double(gravity_step_m),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    return out if rc == 0 else None
