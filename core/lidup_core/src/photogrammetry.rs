use std::thread;

pub(crate) fn resize_rgb_u8_to_f32_impl(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    out: &mut [f32],
) {
    let scale_x = (src_w as f64) / (dst_w as f64);
    let scale_y = (src_h as f64) / (dst_h as f64);
    let sup_x = scale_x.max(1.0);
    let sup_y = scale_y.max(1.0);
    let inv_sup_x = 1.0 / sup_x;
    let inv_sup_y = 1.0 / sup_y;
    let inv_255 = 1.0f64 / 255.0f64;

    // Separable tent-filter resize (antialiased downsampling matching PIL Bilinear)
    let mut x_taps: Vec<(usize, usize, Vec<f64>)> = Vec::with_capacity(dst_w);
    for dx in 0..dst_w {
        let center = (dx as f64 + 0.5) * scale_x - 0.5;
        let x0 = ((center - sup_x).ceil() as isize).clamp(0, (src_w - 1) as isize) as usize;
        let x1 = ((center + sup_x).floor() as isize).clamp(0, (src_w - 1) as isize) as usize;
        let mut ws = Vec::with_capacity(x1 - x0 + 1);
        let mut sum_w = 0.0;
        for sx in x0..=x1 {
            let w = (1.0 - (sx as f64 - center).abs() * inv_sup_x).max(0.0);
            ws.push(w);
            sum_w += w;
        }
        let inv = if sum_w > 0.0 { inv_255 / sum_w } else { inv_255 };
        for w in &mut ws {
            *w *= inv;
        }
        x_taps.push((x0, x1, ws));
    }

    let mut y_taps: Vec<(usize, usize, Vec<f64>)> = Vec::with_capacity(dst_h);
    for dy in 0..dst_h {
        let center = (dy as f64 + 0.5) * scale_y - 0.5;
        let y0 = ((center - sup_y).ceil() as isize).clamp(0, (src_h - 1) as isize) as usize;
        let y1 = ((center + sup_y).floor() as isize).clamp(0, (src_h - 1) as isize) as usize;
        let mut ws = Vec::with_capacity(y1 - y0 + 1);
        let mut sum_w = 0.0;
        for sy in y0..=y1 {
            let w = (1.0 - (sy as f64 - center).abs() * inv_sup_y).max(0.0);
            ws.push(w);
            sum_w += w;
        }
        let inv = if sum_w > 0.0 { 1.0 / sum_w } else { 1.0 };
        for w in &mut ws {
            *w *= inv;
        }
        y_taps.push((y0, y1, ws));
    }

    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // Pass 1: horizontal convolution (src_h x dst_w x 3)
    let mut mid = vec![0.0f32; src_h * dst_w * 3];
    let h_chunks = n_threads.min(src_h.max(1));
    let h_rows_per = (src_h + h_chunks - 1) / h_chunks.max(1);
    let x_taps_ref = &x_taps;

    thread::scope(|s| {
        for (c_idx, mid_chunk) in mid.chunks_mut(h_rows_per * dst_w * 3).enumerate() {
            let y_start = c_idx * h_rows_per;
            s.spawn(move || {
                let rows = mid_chunk.len() / (dst_w * 3);
                for ly in 0..rows {
                    let sy = y_start + ly;
                    let src_row = sy * src_w * 3;
                    let dst_row = ly * dst_w * 3;
                    for (dx, (x0, x1, ws)) in x_taps_ref.iter().enumerate() {
                        let mut r = 0.0f64;
                        let mut g = 0.0f64;
                        let mut b = 0.0f64;
                        for (k, sx) in (*x0..=*x1).enumerate() {
                            let w = ws[k];
                            let idx = src_row + sx * 3;
                            r += (src[idx] as f64) * w;
                            g += (src[idx + 1] as f64) * w;
                            b += (src[idx + 2] as f64) * w;
                        }
                        let ob = dst_row + dx * 3;
                        mid_chunk[ob] = r as f32;
                        mid_chunk[ob + 1] = g as f32;
                        mid_chunk[ob + 2] = b as f32;
                    }
                }
            });
        }
    });

    // Pass 2: vertical convolution (dst_h x dst_w x 3)
    let mid_ref = &mid;
    let v_chunks = n_threads.min(dst_h.max(1));
    let v_rows_per = (dst_h + v_chunks - 1) / v_chunks.max(1);
    let y_taps_ref = &y_taps;

    thread::scope(|s| {
        for (c_idx, out_chunk) in out.chunks_mut(v_rows_per * dst_w * 3).enumerate() {
            let dy_start = c_idx * v_rows_per;
            s.spawn(move || {
                let rows = out_chunk.len() / (dst_w * 3);
                for ly in 0..rows {
                    let dy = dy_start + ly;
                    let (y0, y1, ref ws) = y_taps_ref[dy];
                    let dst_row = ly * dst_w * 3;
                    for dx in 0..dst_w {
                        let mut r = 0.0f64;
                        let mut g = 0.0f64;
                        let mut b = 0.0f64;
                        for (k, sy) in (y0..=y1).enumerate() {
                            let w = ws[k];
                            let idx = (sy * dst_w + dx) * 3;
                            r += (mid_ref[idx] as f64) * w;
                            g += (mid_ref[idx + 1] as f64) * w;
                            b += (mid_ref[idx + 2] as f64) * w;
                        }
                        let ob = dst_row + dx * 3;
                        out_chunk[ob] = r.clamp(0.0, 1.0) as f32;
                        out_chunk[ob + 1] = g.clamp(0.0, 1.0) as f32;
                        out_chunk[ob + 2] = b.clamp(0.0, 1.0) as f32;
                    }
                }
            });
        }
    });
}

pub(crate) fn warp_rgb_to_depth_impl(
    rgb_hi: &[f32],
    d_hi_filled: &[f32],
    rel_mat4x4: &[f64; 16],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    w: usize,
    h: usize,
    out: &mut [f32],
) {
    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(h.max(1));
    let chunk_rows = (h + n_threads - 1) / n_threads.max(1);

    thread::scope(|s| {
        for (c_idx, dst_chunk) in out.chunks_mut(chunk_rows * w * 3).enumerate() {
            let y_start = c_idx * chunk_rows;
            s.spawn(move || {
                let rows = dst_chunk.len() / (w * 3);
                for ly in 0..rows {
                    let y = y_start + ly;
                    let ny = (y as f64 - cy) / fy;
                    for x in 0..w {
                        let d = d_hi_filled[y * w + x] as f64;
                        let px = ((x as f64 - cx) / fx) * d;
                        let py = ny * d;
                        let pz = d;

                        let xr = rel_mat4x4[0] * px + rel_mat4x4[1] * py + rel_mat4x4[2] * pz + rel_mat4x4[3];
                        let yr = rel_mat4x4[4] * px + rel_mat4x4[5] * py + rel_mat4x4[6] * pz + rel_mat4x4[7];
                        let zr = (rel_mat4x4[8] * px + rel_mat4x4[9] * py + rel_mat4x4[10] * pz + rel_mat4x4[11]).max(0.05);

                        let ur = (fx * (xr / zr) + cx).clamp(0.0, (w - 1) as f64);
                        let vr = (fy * (yr / zr) + cy).clamp(0.0, (h - 1) as f64);

                        let u0 = ur.floor() as usize;
                        let v0 = vr.floor() as usize;
                        let u1 = (u0 + 1).min(w - 1);
                        let v1 = (v0 + 1).min(h - 1);
                        let du = (ur - u0 as f64) as f32;
                        let dv = (vr - v0 as f64) as f32;
                        let w00 = (1.0 - du) * (1.0 - dv);
                        let w10 = du * (1.0 - dv);
                        let w01 = (1.0 - du) * dv;
                        let w11 = du * dv;

                        let i00 = (v0 * w + u0) * 3;
                        let i10 = (v0 * w + u1) * 3;
                        let i01 = (v1 * w + u0) * 3;
                        let i11 = (v1 * w + u1) * 3;
                        let ob = (ly * w + x) * 3;
                        for c in 0..3 {
                            dst_chunk[ob + c] = (rgb_hi[i00 + c] * w00
                                + rgb_hi[i10 + c] * w10
                                + rgb_hi[i01 + c] * w01
                                + rgb_hi[i11 + c] * w11)
                                .clamp(0.0, 1.0);
                        }
                    }
                }
            });
        }
    });
}

/// Projects all N 3D points into one 12 MP frame (`hi_rgb_u8`), performs depth-map occlusion + normal facing
/// checks + score update, and samples winning RGB colors directly in-place using multi-core point slices.
pub(crate) fn colorize_hires_frame_u8_impl(
    pts_arkit: &[f64],
    norms_arkit: Option<&[f64]>,
    n_pts: usize,
    pose_rgb_4x4: &[f64; 16],
    hi_rgb_u8: &[u8],
    hi_w: usize,
    hi_h: usize,
    depth_f32: &[f32],
    d_w: usize,
    d_h: usize,
    fx_hi: f64,
    fy_hi: f64,
    cx_hi: f64,
    cy_hi: f64,
    max_depth_m: f64,
    occ_tol_m: f64,
    sharp_imu_w: f64,
    best_score: &mut [f64],
    out_colors: &mut [f64],
) {
    // pose_rgb transforms cam -> world; inverse rotation is R^T and t = pose[:3, 3]
    let tx = pose_rgb_4x4[3];
    let ty = pose_rgb_4x4[7];
    let tz = pose_rgb_4x4[11];

    let r00 = pose_rgb_4x4[0];
    let r10 = pose_rgb_4x4[4];
    let r20 = pose_rgb_4x4[8];
    let r01 = pose_rgb_4x4[1];
    let r11 = pose_rgb_4x4[5];
    let r21 = pose_rgb_4x4[9];
    let r02 = pose_rgb_4x4[2];
    let r12 = pose_rgb_4x4[6];
    let r22 = pose_rgb_4x4[10];

    let hi_w_max = hi_w as f64 - 1.001;
    let hi_h_max = hi_h as f64 - 1.001;
    let d_scale_x = (d_w as f64) / (hi_w as f64);
    let d_scale_y = (d_h as f64) / (hi_h as f64);
    let inv_255 = 1.0f64 / 255.0f64;

    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(n_pts.max(1));
    let chunk_pts = (n_pts + n_threads - 1) / n_threads.max(1);

    thread::scope(|s| {
        for (c_idx, (sc_chunk, col_chunk)) in best_score
            .chunks_mut(chunk_pts)
            .zip(out_colors.chunks_mut(chunk_pts * 3))
            .enumerate()
        {
            let start_p = c_idx * chunk_pts;
            s.spawn(move || {
                for local_i in 0..sc_chunk.len() {
                    let gi = start_p + local_i;
                    let pb = gi * 3;
                    let dx = pts_arkit[pb] - tx;
                    let dy = pts_arkit[pb + 1] - ty;
                    let dz = pts_arkit[pb + 2] - tz;

                    // cam = (p - t) @ R
                    let cz = dx * r02 + dy * r12 + dz * r22;
                    if cz <= 0.20 || cz > max_depth_m {
                        continue;
                    }
                    let cx_p = dx * r00 + dy * r10 + dz * r20;
                    let cy_p = dx * r01 + dy * r11 + dz * r21;

                    let uf = fx_hi * (cx_p / cz) + cx_hi;
                    let vf = fy_hi * (cy_p / cz) + cy_hi;
                    if uf < 0.0 || uf > hi_w_max || vf < 0.0 || vf > hi_h_max {
                        continue;
                    }

                    let ud = ((uf * d_scale_x).round() as usize).min(d_w - 1);
                    let vd = ((vf * d_scale_y).round() as usize).min(d_h - 1);
                    let d_obs = depth_f32[vd * d_w + ud] as f64;
                    if d_obs <= 0.0 || (cz - d_obs).abs() > occ_tol_m {
                        continue;
                    }

                    let cam_norm = (cx_p * cx_p + cy_p * cy_p + cz * cz).sqrt().max(1e-8);
                    let cos_v = if let Some(norms) = norms_arkit {
                        let nx = norms[pb];
                        let ny = norms[pb + 1];
                        let nz = norms[pb + 2];
                        let ncx = nx * r00 + ny * r10 + nz * r20;
                        let ncy = nx * r01 + ny * r11 + nz * r21;
                        let ncz = nx * r02 + ny * r12 + nz * r22;
                        let cv = -(ncx * cx_p + ncy * cy_p + ncz * cz) / cam_norm;
                        if cv <= 0.20 {
                            continue;
                        }
                        cv
                    } else {
                        (cz / cam_norm).clamp(0.1, 1.0)
                    };

                    let z_clamped = cz.max(0.25);
                    let score = (cos_v * sharp_imu_w) / (z_clamped * z_clamped);
                    if score <= sc_chunk[local_i] {
                        continue;
                    }

                    let u0 = uf.floor() as usize;
                    let v0 = vf.floor() as usize;
                    let du = uf - u0 as f64;
                    let dv = vf - v0 as f64;
                    let w00 = (1.0 - du) * (1.0 - dv) * inv_255;
                    let w10 = du * (1.0 - dv) * inv_255;
                    let w01 = (1.0 - du) * dv * inv_255;
                    let w11 = du * dv * inv_255;

                    let i00 = (v0 * hi_w + u0) * 3;
                    let i10 = (v0 * hi_w + u0 + 1) * 3;
                    let i01 = ((v0 + 1) * hi_w + u0) * 3;
                    let i11 = ((v0 + 1) * hi_w + u0 + 1) * 3;

                    let cb = local_i * 3;
                    col_chunk[cb] = (hi_rgb_u8[i00] as f64) * w00
                        + (hi_rgb_u8[i10] as f64) * w10
                        + (hi_rgb_u8[i01] as f64) * w01
                        + (hi_rgb_u8[i11] as f64) * w11;
                    col_chunk[cb + 1] = (hi_rgb_u8[i00 + 1] as f64) * w00
                        + (hi_rgb_u8[i10 + 1] as f64) * w10
                        + (hi_rgb_u8[i01 + 1] as f64) * w01
                        + (hi_rgb_u8[i11 + 1] as f64) * w11;
                    col_chunk[cb + 2] = (hi_rgb_u8[i00 + 2] as f64) * w00
                        + (hi_rgb_u8[i10 + 2] as f64) * w10
                        + (hi_rgb_u8[i01 + 2] as f64) * w01
                        + (hi_rgb_u8[i11 + 2] as f64) * w11;
                    sc_chunk[local_i] = score;
                }
            });
        }
    });
}

pub(crate) struct NeighborSlice<'a> {
    pub nhp: &'a [f32],
    pub depth: &'a [f32],
    pub w: usize,
    pub h: usize,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub t: [f64; 12],
}

#[inline]
fn box5_mean_separable(src: &[f32], tmp: &mut [f32], dst: &mut [f32], w: usize, h: usize) {
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut acc = 0.0f32;
            for dx in -2isize..=2 {
                let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                acc += src[row + xx];
            }
            tmp[row + x] = acc * 0.2;
        }
    }
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut acc = 0.0f32;
            for dy in -2isize..=2 {
                let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                acc += tmp[yy * w + x];
            }
            dst[row + x] = acc * 0.2;
        }
    }
}

pub(crate) fn mvs_sweep_impl(
    d0: &[f32],
    valid: &[u8],
    ref_nhp: &[f32],
    w: usize,
    h: usize,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    neighbors: &[NeighborSlice<'_>],
    max_shift_m: f64,
    steps: usize,
    out_dz: &mut [f32],
    out_sc: &mut [f32],
    out_vis: &mut [f32],
) {
    let n_pix = w * h;
    let step_m = (2.0 * max_shift_m) / ((steps - 1) as f64);

    let mut ray_x = Vec::with_capacity(n_pix);
    let mut ray_y = Vec::with_capacity(n_pix);
    for y in 0..h {
        let ry = (y as f64 - cy) / fy;
        for x in 0..w {
            ray_x.push((x as f64 - cx) / fx);
            ray_y.push(ry);
        }
    }

    let mut score_vol = vec![0.0f32; steps * n_pix];
    let mut vis_vol = vec![0.0f32; steps * n_pix];

    let ray_x_ref = &ray_x;
    let ray_y_ref = &ray_y;

    thread::scope(|s| {
        for (k_idx, (sc_plane, vis_plane)) in score_vol
            .chunks_mut(n_pix)
            .zip(vis_vol.chunks_mut(n_pix))
            .enumerate()
        {
            s.spawn(move || {
                let dz = -max_shift_m + (k_idx as f64) * step_m;
                let mut prod = vec![0.0f32; n_pix];
                let mut vis = vec![0u8; n_pix];
                let mut tmp_box = vec![0.0f32; n_pix];
                let mut corr_box = vec![0.0f32; n_pix];
                let mut sum_corr = vec![0.0f64; n_pix];
                let mut cnt_vis = vec![0.0f64; n_pix];

                for nb in neighbors {
                    prod.fill(0.0);
                    vis.fill(0);
                    let t = &nb.t;
                    let nw_max = nb.w as f64 - 1.001;
                    let nh_max = nb.h as f64 - 1.001;

                    for idx in 0..n_pix {
                        if valid[idx] == 0 {
                            continue;
                        }
                        let dc = ((d0[idx] as f64) + dz).max(0.15);
                        let px = ray_x_ref[idx] * dc;
                        let py = ray_y_ref[idx] * dc;

                        let zj = t[8] * px + t[9] * py + t[10] * dc + t[11];
                        if zj <= 0.15 {
                            continue;
                        }
                        let xj = t[0] * px + t[1] * py + t[2] * dc + t[3];
                        let yj = t[4] * px + t[5] * py + t[6] * dc + t[7];
                        let inv_z = 1.0 / zj;
                        let uj = nb.fx * (xj * inv_z) + nb.cx;
                        let vj = nb.fy * (yj * inv_z) + nb.cy;
                        if uj < 0.0 || uj > nw_max || vj < 0.0 || vj > nh_max {
                            continue;
                        }

                        let u_near = (uj.round() as usize).min(nb.w - 1);
                        let v_near = (vj.round() as usize).min(nb.h - 1);
                        let d_obs = nb.depth[v_near * nb.w + u_near] as f64;
                        if d_obs > 0.0 && (zj - d_obs).abs() > 0.04 {
                            continue;
                        }

                        let u0 = (uj.floor() as usize).min(nb.w - 1);
                        let v0 = (vj.floor() as usize).min(nb.h - 1);
                        let u1 = (u0 + 1).min(nb.w - 1);
                        let v1 = (v0 + 1).min(nb.h - 1);
                        let du = (uj - u0 as f64) as f32;
                        let dv = (vj - v0 as f64) as f32;

                        let c00 = nb.nhp[v0 * nb.w + u0];
                        let c10 = nb.nhp[v0 * nb.w + u1];
                        let c01 = nb.nhp[v1 * nb.w + u0];
                        let c11 = nb.nhp[v1 * nb.w + u1];
                        let sampled = c00 * (1.0 - du) * (1.0 - dv)
                            + c10 * du * (1.0 - dv)
                            + c01 * (1.0 - du) * dv
                            + c11 * du * dv;

                        prod[idx] = ref_nhp[idx] * sampled;
                        vis[idx] = 1;
                    }

                    box5_mean_separable(&prod, &mut tmp_box, &mut corr_box, w, h);
                    for idx in 0..n_pix {
                        if vis[idx] != 0 {
                            let c = corr_box[idx].max(0.0) as f64;
                            sum_corr[idx] += c;
                            cnt_vis[idx] += 1.0;
                        }
                    }
                }

                for idx in 0..n_pix {
                    let cv = cnt_vis[idx];
                    sc_plane[idx] = (sum_corr[idx] / cv.max(1.0)) as f32;
                    vis_plane[idx] = cv as f32;
                }
            });
        }
    });

    for idx in 0..n_pix {
        let mut best_k = 0usize;
        let mut best_s = score_vol[idx];
        for k in 1..steps {
            let s = score_vol[k * n_pix + idx];
            if s > best_s {
                best_s = s;
                best_k = k;
            }
        }
        let best_s64 = best_s as f64;
        let sub_off = if best_k > 0 && best_k + 1 < steps {
            let sm = score_vol[(best_k - 1) * n_pix + idx] as f64;
            let sp = score_vol[(best_k + 1) * n_pix + idx] as f64;
            let denom = 2.0 * (sm - 2.0 * best_s64 + sp);
            if denom < -1e-4 {
                ((sm - sp) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        } else {
            0.0
        };
        let dz_grid_val = -max_shift_m + (best_k as f64) * step_m;
        out_dz[idx] = (dz_grid_val + sub_off * step_m) as f32;
        out_sc[idx] = best_s;
        out_vis[idx] = vis_vol[best_k * n_pix + idx];
    }
}
