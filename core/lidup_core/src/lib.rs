mod capture;
mod depth;
mod photogrammetry;
mod terrain;

use std::slice;

use capture::{
    center_depth_meters_slice, convert_float32_depth_to_mm_u16_slice, se3_moved_enough_col_major,
    sharpness_laplacian_channel_slice, weighted_gnss_average_slice,
};
use depth::{extract_depth_points_impl, upsample_depth_guided_impl};
use photogrammetry::{
    colorize_hires_frame_u8_impl, mvs_sweep_impl, resize_rgb_u8_to_f32_impl,
    warp_rgb_to_depth_impl, NeighborSlice,
};
use terrain::csf_cloth_simulate_impl;

const ABI_VERSION: u32 = 2;

#[no_mangle]
pub extern "C" fn lidup_core_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub unsafe extern "C" fn lidup_depth_float32_to_mm_u16(
    src_bytes: *const u8,
    src_byte_count: usize,
    src_stride_bytes: i32,
    width: i32,
    height: i32,
    out_mm: *mut u16,
) -> i32 {
    if src_bytes.is_null() || out_mm.is_null() || width <= 0 || height <= 0 || src_stride_bytes <= 0 {
        return -1;
    }
    let w = width as usize;
    let h = height as usize;
    let stride = src_stride_bytes as usize;
    if stride < w * 4 {
        return -2;
    }
    let src = unsafe { slice::from_raw_parts(src_bytes, src_byte_count) };
    let out = unsafe { slice::from_raw_parts_mut(out_mm, w * h) };
    convert_float32_depth_to_mm_u16_slice(src, stride, w, h, out);
    0
}

#[no_mangle]
pub unsafe extern "C" fn lidup_center_depth_m_f32(
    src_bytes: *const u8,
    src_byte_count: usize,
    src_stride_bytes: i32,
    width: i32,
    height: i32,
) -> f32 {
    if src_bytes.is_null() || width < 5 || height < 5 || src_stride_bytes <= 0 {
        return -1.0;
    }
    let src = unsafe { slice::from_raw_parts(src_bytes, src_byte_count) };
    center_depth_meters_slice(src, src_stride_bytes as usize, width as usize, height as usize)
        .unwrap_or(-1.0)
}

#[no_mangle]
pub unsafe extern "C" fn lidup_sharpness_laplacian_u8(
    y_bytes: *const u8,
    y_byte_count: usize,
    stride_bytes: i32,
    width: i32,
    height: i32,
    sample_step: i32,
) -> f32 {
    if y_bytes.is_null() || width < 4 || height < 4 || stride_bytes < width {
        return 0.0;
    }
    let src = unsafe { slice::from_raw_parts(y_bytes, y_byte_count) };
    sharpness_laplacian_channel_slice(
        src,
        stride_bytes as usize,
        width as usize,
        height as usize,
        1,
        0,
        sample_step.max(1) as usize,
    )
}

#[no_mangle]
pub unsafe extern "C" fn lidup_sharpness_laplacian_bgra_u8(
    bgra_bytes: *const u8,
    bgra_byte_count: usize,
    stride_bytes: i32,
    width: i32,
    height: i32,
    sample_step: i32,
) -> f32 {
    if bgra_bytes.is_null() || width < 4 || height < 4 || stride_bytes < width * 4 {
        return 0.0;
    }
    let src = unsafe { slice::from_raw_parts(bgra_bytes, bgra_byte_count) };
    sharpness_laplacian_channel_slice(
        src,
        stride_bytes as usize,
        width as usize,
        height as usize,
        4,
        1,
        sample_step.max(1) as usize,
    )
}

#[no_mangle]
pub unsafe extern "C" fn lidup_weighted_gnss_average(
    lat: *const f64,
    lon: *const f64,
    alt: *const f64,
    h_acc: *const f64,
    v_acc: *const f64,
    count: i32,
    out_lat: *mut f64,
    out_lon: *mut f64,
    out_alt: *mut f64,
    out_h_acc_mean: *mut f64,
    out_v_acc_mean: *mut f64,
) -> i32 {
    if lat.is_null()
        || lon.is_null()
        || alt.is_null()
        || h_acc.is_null()
        || v_acc.is_null()
        || out_lat.is_null()
        || out_lon.is_null()
        || out_alt.is_null()
        || out_h_acc_mean.is_null()
        || out_v_acc_mean.is_null()
        || count <= 0
    {
        return 0;
    }
    let n = count as usize;
    let res = weighted_gnss_average_slice(
        unsafe { slice::from_raw_parts(lat, n) },
        unsafe { slice::from_raw_parts(lon, n) },
        unsafe { slice::from_raw_parts(alt, n) },
        unsafe { slice::from_raw_parts(h_acc, n) },
        unsafe { slice::from_raw_parts(v_acc, n) },
    );
    match res {
        Some((vc, la, lo, al, ha, va)) => {
            unsafe {
                *out_lat = la;
                *out_lon = lo;
                *out_alt = al;
                *out_h_acc_mean = ha;
                *out_v_acc_mean = va;
            }
            vc as i32
        }
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn lidup_se3_moved_enough(
    prev_col_major_16: *const f32,
    curr_col_major_16: *const f32,
    min_dist_m: f32,
    min_angle_deg: f32,
) -> i32 {
    if prev_col_major_16.is_null() || curr_col_major_16.is_null() {
        return 1;
    }
    let p: &[f32; 16] = unsafe { &*(prev_col_major_16 as *const [f32; 16]) };
    let c: &[f32; 16] = unsafe { &*(curr_col_major_16 as *const [f32; 16]) };
    if se3_moved_enough_col_major(p, c, min_dist_m, min_angle_deg) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn lidup_upsample_depth_guided_f32(
    depth_m_f32: *const f32,
    valid_lo_u8: *const u8,
    conf_u8: *const u8,
    has_conf: i32,
    rgb_lo_f32: *const f32,
    rgb_hi_f32: *const f32,
    d_w: i32,
    d_h: i32,
    u_w: i32,
    u_h: i32,
    scale: i32,
    edge_jump_m: f32,
    out_depth_f32: *mut f32,
    out_mask_u8: *mut u8,
) -> i32 {
    if depth_m_f32.is_null()
        || valid_lo_u8.is_null()
        || rgb_lo_f32.is_null()
        || rgb_hi_f32.is_null()
        || out_depth_f32.is_null()
        || out_mask_u8.is_null()
        || d_w <= 0
        || d_h <= 0
        || u_w <= 0
        || u_h <= 0
        || scale <= 0
    {
        return -1;
    }
    let dw = d_w as usize;
    let dh = d_h as usize;
    let uw = u_w as usize;
    let uh = u_h as usize;
    let d_len = dw * dh;
    let u_len = uw * uh;
    upsample_depth_guided_impl(
        unsafe { slice::from_raw_parts(depth_m_f32, d_len) },
        unsafe { slice::from_raw_parts(valid_lo_u8, d_len) },
        if has_conf != 0 && !conf_u8.is_null() {
            Some(unsafe { slice::from_raw_parts(conf_u8, d_len) })
        } else {
            None
        },
        unsafe { slice::from_raw_parts(rgb_lo_f32, d_len * 3) },
        unsafe { slice::from_raw_parts(rgb_hi_f32, u_len * 3) },
        dw,
        dh,
        uw,
        uh,
        scale as f64,
        edge_jump_m as f64,
        unsafe { slice::from_raw_parts_mut(out_depth_f32, u_len) },
        unsafe { slice::from_raw_parts_mut(out_mask_u8, u_len) },
    );
    0
}

#[no_mangle]
pub unsafe extern "C" fn lidup_extract_depth_points_f64(
    depth_m_f64: *const f64,
    in_mask_u8: *const u8,
    width: i32,
    height: i32,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    edge_jump_m: f64,
    cos_thresh: f64,
    out_mask_u8: *mut u8,
    out_pts_f64: *mut f64,
    out_weights_f64: *mut f64,
    out_v_idx: *mut i64,
    out_u_idx: *mut i64,
    out_count: *mut i32,
) -> i32 {
    if depth_m_f64.is_null()
        || in_mask_u8.is_null()
        || out_mask_u8.is_null()
        || out_pts_f64.is_null()
        || out_weights_f64.is_null()
        || out_v_idx.is_null()
        || out_u_idx.is_null()
        || out_count.is_null()
        || width <= 0
        || height <= 0
    {
        return -1;
    }
    let w = width as usize;
    let h = height as usize;
    let n_pix = w * h;
    let cnt = extract_depth_points_impl(
        unsafe { slice::from_raw_parts(depth_m_f64, n_pix) },
        unsafe { slice::from_raw_parts(in_mask_u8, n_pix) },
        w,
        h,
        fx,
        fy,
        cx,
        cy,
        edge_jump_m,
        cos_thresh,
        unsafe { slice::from_raw_parts_mut(out_mask_u8, n_pix) },
        unsafe { slice::from_raw_parts_mut(out_pts_f64, n_pix * 3) },
        unsafe { slice::from_raw_parts_mut(out_weights_f64, n_pix) },
        unsafe { slice::from_raw_parts_mut(out_v_idx, n_pix) },
        unsafe { slice::from_raw_parts_mut(out_u_idx, n_pix) },
    );
    unsafe { *out_count = cnt as i32 };
    0
}

#[no_mangle]
pub unsafe extern "C" fn lidup_resize_rgb_u8_to_f32(
    src_rgb_u8: *const u8,
    src_w: i32,
    src_h: i32,
    dst_w: i32,
    dst_h: i32,
    out_rgb_f32: *mut f32,
) -> i32 {
    if src_rgb_u8.is_null() || out_rgb_f32.is_null() || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0 {
        return -1;
    }
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    resize_rgb_u8_to_f32_impl(
        unsafe { slice::from_raw_parts(src_rgb_u8, sw * sh * 3) },
        sw,
        sh,
        dw,
        dh,
        unsafe { slice::from_raw_parts_mut(out_rgb_f32, dw * dh * 3) },
    );
    0
}

#[no_mangle]
pub unsafe extern "C" fn lidup_warp_rgb_to_depth_f32(
    rgb_hi_f32: *const f32,
    d_hi_filled_f32: *const f32,
    rel_mat4x4_f64: *const f64,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    width: i32,
    height: i32,
    out_rgb_f32: *mut f32,
) -> i32 {
    if rgb_hi_f32.is_null() || d_hi_filled_f32.is_null() || rel_mat4x4_f64.is_null() || out_rgb_f32.is_null() || width <= 0 || height <= 0 {
        return -1;
    }
    let w = width as usize;
    let h = height as usize;
    let mat: &[f64; 16] = unsafe { &*(rel_mat4x4_f64 as *const [f64; 16]) };
    warp_rgb_to_depth_impl(
        unsafe { slice::from_raw_parts(rgb_hi_f32, w * h * 3) },
        unsafe { slice::from_raw_parts(d_hi_filled_f32, w * h) },
        mat,
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        unsafe { slice::from_raw_parts_mut(out_rgb_f32, w * h * 3) },
    );
    0
}

#[no_mangle]
pub unsafe extern "C" fn lidup_colorize_hires_frame_u8(
    pts_arkit_f64: *const f64,
    norms_arkit_f64: *const f64,
    n_pts: i32,
    pose_rgb_4x4_f64: *const f64,
    hi_rgb_u8: *const u8,
    hi_w: i32,
    hi_h: i32,
    depth_f32: *const f32,
    d_w: i32,
    d_h: i32,
    fx_hi: f64,
    fy_hi: f64,
    cx_hi: f64,
    cy_hi: f64,
    max_depth_m: f64,
    occ_tol_m: f64,
    sharp_imu_w: f64,
    best_score_f64: *mut f64,
    out_colors_f64: *mut f64,
) -> i32 {
    if pts_arkit_f64.is_null()
        || pose_rgb_4x4_f64.is_null()
        || hi_rgb_u8.is_null()
        || depth_f32.is_null()
        || best_score_f64.is_null()
        || out_colors_f64.is_null()
        || n_pts <= 0
        || hi_w <= 1
        || hi_h <= 1
        || d_w <= 0
        || d_h <= 0
    {
        return -1;
    }
    let np = n_pts as usize;
    let hw = hi_w as usize;
    let hh = hi_h as usize;
    let dw = d_w as usize;
    let dh = d_h as usize;
    let pose: &[f64; 16] = unsafe { &*(pose_rgb_4x4_f64 as *const [f64; 16]) };
    colorize_hires_frame_u8_impl(
        unsafe { slice::from_raw_parts(pts_arkit_f64, np * 3) },
        if norms_arkit_f64.is_null() {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(norms_arkit_f64, np * 3) })
        },
        np,
        pose,
        unsafe { slice::from_raw_parts(hi_rgb_u8, hw * hh * 3) },
        hw,
        hh,
        unsafe { slice::from_raw_parts(depth_f32, dw * dh) },
        dw,
        dh,
        fx_hi,
        fy_hi,
        cx_hi,
        cy_hi,
        max_depth_m,
        occ_tol_m,
        sharp_imu_w,
        unsafe { slice::from_raw_parts_mut(best_score_f64, np) },
        unsafe { slice::from_raw_parts_mut(out_colors_f64, np * 3) },
    );
    0
}

#[no_mangle]
pub unsafe extern "C" fn lidup_mvs_sweep_f32(
    d0_f32: *const f32,
    valid_u8: *const u8,
    ref_nhp_f32: *const f32,
    d_w: i32,
    d_h: i32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    n_neighbors: i32,
    nb_nhp_ptrs: *const *const f32,
    nb_depth_ptrs: *const *const f32,
    nb_t_mats_f32: *const f32,
    nb_dims_i32: *const i32,
    nb_intr_f32: *const f32,
    max_shift_m: f32,
    n_steps: i32,
    out_dz_raw_f32: *mut f32,
    out_best_score_f32: *mut f32,
    out_best_vis_f32: *mut f32,
) -> i32 {
    if d0_f32.is_null()
        || valid_u8.is_null()
        || ref_nhp_f32.is_null()
        || nb_nhp_ptrs.is_null()
        || nb_depth_ptrs.is_null()
        || nb_t_mats_f32.is_null()
        || nb_dims_i32.is_null()
        || nb_intr_f32.is_null()
        || out_dz_raw_f32.is_null()
        || out_best_score_f32.is_null()
        || out_best_vis_f32.is_null()
        || d_w <= 0
        || d_h <= 0
        || n_neighbors <= 0
        || n_steps < 3
    {
        return -1;
    }

    let dw = d_w as usize;
    let dh = d_h as usize;
    let n_pix = dw * dh;
    let nn = n_neighbors as usize;
    let steps = n_steps as usize;

    let ptrs_nhp = unsafe { slice::from_raw_parts(nb_nhp_ptrs, nn) };
    let ptrs_depth = unsafe { slice::from_raw_parts(nb_depth_ptrs, nn) };
    let mats = unsafe { slice::from_raw_parts(nb_t_mats_f32, nn * 16) };
    let dims = unsafe { slice::from_raw_parts(nb_dims_i32, nn * 2) };
    let intrs = unsafe { slice::from_raw_parts(nb_intr_f32, nn * 4) };

    let mut neighbors = Vec::with_capacity(nn);
    for j in 0..nn {
        let nw = dims[j * 2];
        let nh = dims[j * 2 + 1];
        if nw <= 0 || nh <= 0 || ptrs_nhp[j].is_null() || ptrs_depth[j].is_null() {
            return -2;
        }
        let n_len = (nw as usize) * (nh as usize);
        let m = &mats[j * 16..(j + 1) * 16];
        let mut t12 = [0.0f64; 12];
        for r in 0..3 {
            for c in 0..4 {
                t12[r * 4 + c] = m[r * 4 + c] as f64;
            }
        }
        neighbors.push(NeighborSlice {
            nhp: unsafe { slice::from_raw_parts(ptrs_nhp[j], n_len) },
            depth: unsafe { slice::from_raw_parts(ptrs_depth[j], n_len) },
            w: nw as usize,
            h: nh as usize,
            fx: intrs[j * 4] as f64,
            fy: intrs[j * 4 + 1] as f64,
            cx: intrs[j * 4 + 2] as f64,
            cy: intrs[j * 4 + 3] as f64,
            t: t12,
        });
    }

    mvs_sweep_impl(
        unsafe { slice::from_raw_parts(d0_f32, n_pix) },
        unsafe { slice::from_raw_parts(valid_u8, n_pix) },
        unsafe { slice::from_raw_parts(ref_nhp_f32, n_pix) },
        dw,
        dh,
        fx as f64,
        fy as f64,
        cx as f64,
        cy as f64,
        &neighbors,
        max_shift_m as f64,
        steps,
        unsafe { slice::from_raw_parts_mut(out_dz_raw_f32, n_pix) },
        unsafe { slice::from_raw_parts_mut(out_best_score_f32, n_pix) },
        unsafe { slice::from_raw_parts_mut(out_best_vis_f32, n_pix) },
    );
    0
}

#[no_mangle]
pub unsafe extern "C" fn lidup_csf_cloth_filter_f64(
    terrain_filled_f64: *const f64,
    nx: i32,
    ny: i32,
    z_inv_max: f64,
    rigidness: i32,
    iterations: i32,
    gravity_step_m: f64,
    out_dtm_f64: *mut f64,
) -> i32 {
    if terrain_filled_f64.is_null() || out_dtm_f64.is_null() || nx <= 0 || ny <= 0 {
        return -1;
    }
    let n = (nx as usize) * (ny as usize);
    csf_cloth_simulate_impl(
        unsafe { slice::from_raw_parts(terrain_filled_f64, n) },
        nx as usize,
        ny as usize,
        z_inv_max,
        rigidness,
        iterations,
        gravity_step_m,
        unsafe { slice::from_raw_parts_mut(out_dtm_f64, n) },
    );
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_float32_to_mm_u16_and_center() {
        let mut raw = Vec::new();
        for i in 0..25 {
            raw.extend_from_slice(&((1.5f32) + (i as f32) * 0.01).to_le_bytes());
        }
        let mut out = [0u16; 25];
        convert_float32_depth_to_mm_u16_slice(&raw, 20, 5, 5, &mut out);
        assert_eq!(out[0], 1500);
        let c = center_depth_meters_slice(&raw, 20, 5, 5).unwrap();
        assert!((c - 1.62).abs() < 1e-4);
    }

    #[test]
    fn test_gnss_weighted_avg_and_se3_motion() {
        let lat = [-23.5, -23.6, -23.5];
        let lon = [-46.6, -46.6, -46.6];
        let alt = [750.0, 900.0, 750.0];
        let h_acc = [2.0, 150.0, 2.0];
        let v_acc = [3.0, 20.0, 3.0];
        let (cnt, la, _, al, ha, _) = weighted_gnss_average_slice(&lat, &lon, &alt, &h_acc, &v_acc).unwrap();
        assert_eq!(cnt, 2);
        assert!((la - (-23.5)).abs() < 1e-6);
        assert!((al - 750.0).abs() < 1e-6);
        assert!((ha - 2.0).abs() < 1e-6);

        let a = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut b = a;
        assert!(!se3_moved_enough_col_major(&a, &b, 0.12, 5.0));
        b[12] = 0.20;
        assert!(se3_moved_enough_col_major(&a, &b, 0.12, 5.0));
    }

    #[test]
    fn test_extract_depth_points_and_resize_and_colorize() {
        let (w, h) = (10, 8);
        let depth = vec![2.0f64; w * h];
        let mask = vec![1u8; w * h];
        let mut out_m = vec![0u8; w * h];
        let mut pts = vec![0.0f64; w * h * 3];
        let mut wts = vec![0.0f64; w * h];
        let mut vi = vec![0i64; w * h];
        let mut ui = vec![0i64; w * h];
        let cnt = extract_depth_points_impl(
            &depth,
            &mask,
            w,
            h,
            100.0,
            100.0,
            5.0,
            4.0,
            0.12,
            0.25,
            &mut out_m,
            &mut pts,
            &mut wts,
            &mut vi,
            &mut ui,
        );
        assert_eq!(cnt, w * h);
        assert!(wts[0] > 0.2 && wts[0] <= 0.3);

        // Resize 4x4 red image -> 2x2 red float32
        let mut src = vec![0u8; 4 * 4 * 3];
        for px in src.chunks_mut(3) {
            px[0] = 255;
        }
        let mut dst = vec![0.0f32; 2 * 2 * 3];
        resize_rgb_u8_to_f32_impl(&src, 4, 4, 2, 2, &mut dst);
        assert!((dst[0] - 1.0).abs() < 1e-4 && dst[1].abs() < 1e-4);

        // Colorize 1 point directly from green 10x8 buffer
        let mut hi_u8 = vec![0u8; 10 * 8 * 3];
        for px in hi_u8.chunks_mut(3) {
            px[1] = 204;
        }
        let d_f32 = vec![2.0f32; 10 * 8];
        let pose = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let p_in = [0.0, 0.0, 2.0];
        let mut sc = [-1.0];
        let mut cols = [0.0, 0.0, 0.0];
        colorize_hires_frame_u8_impl(
            &p_in, None, 1, &pose, &hi_u8, 10, 8, &d_f32, 10, 8,
            100.0, 100.0, 5.0, 4.0, 5.0, 0.1, 1.0, &mut sc, &mut cols,
        );
        assert!(sc[0] > 0.0);
        assert!((cols[1] - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_csf_cloth_simulation() {
        let mut terrain = vec![0.0f64; 25];
        terrain[12] = -2.0; // outlier tree peak in inverted space (-(-2)=+2 above z=0) -> wait z_inv=-z so ground=0, bush=-2? No: in -Z, ground=0, tree=+2 (higher in normal Z => lower in -Z)!
        // Let's keep normal plane z_inv=0 with one dip at -1.0 (tall bump in Z), cloth bridges over dip!
        let mut dtm = vec![0.0f64; 25];
        csf_cloth_simulate_impl(&terrain, 5, 5, 1.0, 2, 100, 0.25, &mut dtm);
        assert!(dtm[0].abs() < 0.05);
        assert!(dtm[12] > -0.8);
    }

    #[test]
    fn test_upsample_preserves_sharp_step() {
        let (dw, dh, scale) = (12, 10, 2);
        let (uw, uh) = (dw * scale, dh * scale);
        let mut depth = vec![2.5f32; dw * dh];
        let valid = vec![1u8; dw * dh];
        let mut rgb_lo = vec![0.0f32; dw * dh * 3];
        for y in 0..dh {
            for x in 0..dw {
                let i = y * dw + x;
                if x < dw / 2 {
                    depth[i] = 1.0;
                    rgb_lo[i * 3] = 0.9;
                    rgb_lo[i * 3 + 2] = 0.1;
                } else {
                    rgb_lo[i * 3] = 0.1;
                    rgb_lo[i * 3 + 2] = 0.9;
                }
            }
        }
        let mut rgb_hi = vec![0.0f32; uw * uh * 3];
        for y in 0..uh {
            for x in 0..uw {
                let b = (y * uw + x) * 3;
                if x < uw / 2 {
                    rgb_hi[b] = 0.9;
                    rgb_hi[b + 2] = 0.1;
                } else {
                    rgb_hi[b] = 0.1;
                    rgb_hi[b + 2] = 0.9;
                }
            }
        }
        let mut out_d = vec![0.0f32; uw * uh];
        let mut out_m = vec![0u8; uw * uh];
        upsample_depth_guided_impl(
            &depth,
            &valid,
            None,
            &rgb_lo,
            &rgb_hi,
            dw,
            dh,
            uw,
            uh,
            scale as f64,
            0.12,
            &mut out_d,
            &mut out_m,
        );
        assert!((out_d[5 * uw + 2] - 1.0).abs() < 0.05);
        assert!((out_d[5 * uw + (uw - 3)] - 2.5).abs() < 0.05);
    }

    #[test]
    fn test_mvs_recovers_positive_depth_shift() {
        let (w, h) = (256, 48);
        let (fx, fy, cx, cy) = (220.0f64, 220.0f64, 128.0f64, 24.0f64);
        let true_d = 1.508f64;
        let d_init = vec![1.500f32; w * h];
        let valid = vec![1u8; w * h];

        let mut ref_nhp = vec![0.0f32; w * h];
        let mut nb_nhp = vec![0.0f32; w * h];
        let nb_depth = vec![true_d as f32; w * h];

        let tx = 0.35f64;
        for y in 0..h {
            for x in 0..w {
                let px = (x as f64 - cx) * true_d / fx;
                ref_nhp[y * w + x] = ((px * 55.0).sin() + 0.6 * (px * 95.0).cos()) as f32;
                let px_nb = tx + (x as f64 - cx) * true_d / fx;
                nb_nhp[y * w + x] = ((px_nb * 55.0).sin() + 0.6 * (px_nb * 95.0).cos()) as f32;
            }
        }
        let nb = NeighborSlice {
            nhp: &nb_nhp,
            depth: &nb_depth,
            w,
            h,
            fx,
            fy,
            cx,
            cy,
            t: [1.0, 0.0, 0.0, -tx, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        };
        let mut out_dz = vec![0.0f32; w * h];
        let mut out_sc = vec![0.0f32; w * h];
        let mut out_vis = vec![0.0f32; w * h];
        mvs_sweep_impl(
            &d_init,
            &valid,
            &ref_nhp,
            w,
            h,
            fx,
            fy,
            cx,
            cy,
            &[nb],
            0.012,
            13,
            &mut out_dz,
            &mut out_sc,
            &mut out_vis,
        );
        let center_dz = out_dz[24 * w + 160];
        assert!((center_dz - 0.008).abs() < 0.003, "got center_dz={}", center_dz);
    }
}
