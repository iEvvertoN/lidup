use std::thread;

pub(crate) fn upsample_depth_guided_impl(
    depth_m: &[f32],
    valid_lo: &[u8],
    conf: Option<&[u8]>,
    rgb_lo: &[f32],
    rgb_hi: &[f32],
    d_w: usize,
    d_h: usize,
    u_w: usize,
    u_h: usize,
    scale_f: f64,
    edge_jump_m: f64,
    out_depth: &mut [f32],
    out_mask: &mut [u8],
) {
    let mut is_edge_lo = vec![0u8; d_w * d_h];
    if edge_jump_m > 0.0 {
        for y in 0..d_h {
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(d_h - 1);
            for x in 0..d_w {
                let idx = y * d_w + x;
                if valid_lo[idx] == 0 {
                    continue;
                }
                let x0 = x.saturating_sub(1);
                let x1 = (x + 1).min(d_w - 1);
                let mut d_hi = 0.0f64;
                let mut d_lo = 1e6f64;
                for yy in y0..=y1 {
                    for xx in x0..=x1 {
                        let ni = yy * d_w + xx;
                        if valid_lo[ni] != 0 {
                            let dv = depth_m[ni] as f64;
                            if dv > d_hi {
                                d_hi = dv;
                            }
                            if dv < d_lo {
                                d_lo = dv;
                            }
                        }
                    }
                }
                if (d_hi - d_lo) > edge_jump_m {
                    is_edge_lo[idx] = 1;
                }
            }
        }
    }

    let inv_2ss2 = 1.0 / (2.0 * 0.75 * 0.75);
    let inv_2sr2_edge = 1.0 / (2.0 * 0.065 * 0.065);
    let inv_2sr2_flat = 1.0 / (2.0 * 0.16 * 0.16);
    let max_std = if edge_jump_m > 0.0 {
        (edge_jump_m * 0.45).max(0.04)
    } else {
        0.08
    };

    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(u_h.max(1));
    let chunk_rows = (u_h + n_threads - 1) / n_threads.max(1);

    let mut stage_mask = vec![0u8; u_w * u_h];

    thread::scope(|s| {
        for (chunk_idx, (d_chunk, m_chunk)) in out_depth
            .chunks_mut(chunk_rows * u_w)
            .zip(stage_mask.chunks_mut(chunk_rows * u_w))
            .enumerate()
        {
            let y_start = chunk_idx * chunk_rows;
            let is_edge_ref = &is_edge_lo;
            s.spawn(move || {
                let rows_in_chunk = d_chunk.len() / u_w;
                for local_y in 0..rows_in_chunk {
                    let uy = y_start + local_y;
                    let y_f = (uy as f64 + 0.5) / scale_f - 0.5;
                    let y0 = (y_f.round() as isize).clamp(0, d_h as isize - 1);

                    for ux in 0..u_w {
                        let x_f = (ux as f64 + 0.5) / scale_f - 0.5;
                        let x0 = (x_f.round() as isize).clamp(0, d_w as isize - 1);

                        let edge_flag = is_edge_ref[(y0 as usize) * d_w + (x0 as usize)] != 0;
                        let inv_2sr2 = if edge_flag { inv_2sr2_edge } else { inv_2sr2_flat };

                        let hi_base = (uy * u_w + ux) * 3;
                        let hr = rgb_hi[hi_base] as f64;
                        let hg = rgb_hi[hi_base + 1] as f64;
                        let hb = rgb_hi[hi_base + 2] as f64;

                        let mut num = 0.0f64;
                        let mut num2 = 0.0f64;
                        let mut den = 0.0f64;
                        let mut max_wr = 0.0f64;

                        for dy in -1..=1 {
                            let yr = y0 + dy;
                            if yr < 0 || yr >= d_h as isize {
                                continue;
                            }
                            let yi = yr as usize;
                            let dy2 = (y_f - yi as f64) * (y_f - yi as f64);

                            for dx in -1..=1 {
                                let xr = x0 + dx;
                                if xr < 0 || xr >= d_w as isize {
                                    continue;
                                }
                                let xi = xr as usize;
                                let ni = yi * d_w + xi;
                                if valid_lo[ni] == 0 {
                                    continue;
                                }

                                let ds2 = dy2 + (x_f - xi as f64) * (x_f - xi as f64);
                                let ws = (-ds2 * inv_2ss2).exp();

                                let lo_base = ni * 3;
                                let dr = hr - rgb_lo[lo_base] as f64;
                                let dg = hg - rgb_lo[lo_base + 1] as f64;
                                let db = hb - rgb_lo[lo_base + 2] as f64;
                                let wr = (-(dr * dr + dg * dg + db * db) * inv_2sr2).exp();
                                if wr > max_wr {
                                    max_wr = wr;
                                }

                                let cw = match conf {
                                    Some(c) if c[ni] >= 2 => 1.5,
                                    _ => 1.0,
                                };
                                let w = ws * wr * cw;
                                let d_nb = depth_m[ni] as f64;
                                num += w * d_nb;
                                num2 += w * d_nb * d_nb;
                                den += w;
                            }
                        }

                        let out_idx = local_y * u_w + ux;
                        if den > 1e-4 {
                            let up_d = num / den;
                            let var_d = (num2 / den - up_d * up_d).max(0.0);
                            let std_d = var_d.sqrt();
                            let unmixed_ok = !edge_flag || (max_wr >= 0.20 && std_d <= max_std);
                            d_chunk[out_idx] = up_d as f32;
                            m_chunk[out_idx] = if unmixed_ok { 1 } else { 0 };
                        } else {
                            d_chunk[out_idx] = 0.0;
                            m_chunk[out_idx] = 0;
                        }
                    }
                }
            });
        }
    });

    if edge_jump_m > 0.0 {
        let thresh = edge_jump_m * 1.25;
        for y in 0..u_h {
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(u_h - 1);
            for x in 0..u_w {
                let idx = y * u_w + x;
                if stage_mask[idx] == 0 {
                    out_mask[idx] = 0;
                    continue;
                }
                let x0 = x.saturating_sub(1);
                let x1 = (x + 1).min(u_w - 1);
                let mut hi = 0.0f64;
                let mut lo = 1e6f64;
                for yy in y0..=y1 {
                    for xx in x0..=x1 {
                        let ni = yy * u_w + xx;
                        if stage_mask[ni] != 0 {
                            let dv = out_depth[ni] as f64;
                            if dv > hi {
                                hi = dv;
                            }
                            if dv < lo {
                                lo = dv;
                            }
                        }
                    }
                }
                out_mask[idx] = if (hi - lo) <= thresh { 1 } else { 0 };
            }
        }
    } else {
        out_mask.copy_from_slice(&stage_mask);
    }
}

/// Performs optional 3x3 edge jump filter, backprojection, local incidence angle cosine calculation,
/// grazing-angle pruning, and packs valid (cam_pts, weights, v_idx, u_idx) in one parallel-friendly pass.
pub(crate) fn extract_depth_points_impl(
    depth_m: &[f64],
    in_mask: &[u8],
    w: usize,
    h: usize,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    edge_jump_m: f64,
    cos_thresh: f64,
    out_mask: &mut [u8],
    out_pts: &mut [f64],
    out_weights: &mut [f64],
    out_v_idx: &mut [i64],
    out_u_idx: &mut [i64],
) -> usize {
    let n_pix = w * h;
    // 1. Optional 3x3 edge-jump filter
    let mut stage_mask = vec![0u8; n_pix];
    let mut stage_count = 0usize;
    if edge_jump_m > 0.0 {
        for y in 0..h {
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(h - 1);
            for x in 0..w {
                let idx = y * w + x;
                if in_mask[idx] == 0 {
                    continue;
                }
                let x0 = x.saturating_sub(1);
                let x1 = (x + 1).min(w - 1);
                let mut hi = 0.0f64;
                let mut lo = 1e6f64;
                for yy in y0..=y1 {
                    for xx in x0..=x1 {
                        let ni = yy * w + xx;
                        if in_mask[ni] != 0 {
                            let dv = depth_m[ni];
                            if dv > hi {
                                hi = dv;
                            }
                            if dv < lo {
                                lo = dv;
                            }
                        }
                    }
                }
                if (hi - lo) <= edge_jump_m {
                    stage_mask[idx] = 1;
                    stage_count += 1;
                }
            }
        }
    } else {
        for i in 0..n_pix {
            if in_mask[i] != 0 {
                stage_mask[i] = 1;
                stage_count += 1;
            }
        }
    }

    if stage_count == 0 {
        out_mask.fill(0);
        return 0;
    }

    // 2. Backproject full grid to compute finite-difference normals + cos_inc matching np.gradient
    let mut grid = vec![0.0f64; n_pix * 3];
    for y in 0..h {
        let ny = (y as f64 - cy) / fy;
        for x in 0..w {
            let idx = y * w + x;
            let z = depth_m[idx];
            let base = idx * 3;
            grid[base] = ((x as f64 - cx) / fx) * z;
            grid[base + 1] = ny * z;
            grid[base + 2] = z;
        }
    }

    let mut cos_inc = vec![0.0f64; n_pix];
    let mut n_norm = vec![0.0f64; n_pix];
    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(h.max(1));
    let chunk_rows = (h + n_threads - 1) / n_threads.max(1);
    let grid_ref = &grid;

    thread::scope(|s| {
        for (c_idx, (cos_chunk, norm_chunk)) in cos_inc
            .chunks_mut(chunk_rows * w)
            .zip(n_norm.chunks_mut(chunk_rows * w))
            .enumerate()
        {
            let y_start = c_idx * chunk_rows;
            s.spawn(move || {
                let rows = cos_chunk.len() / w;
                for ly in 0..rows {
                    let y = y_start + ly;
                    for x in 0..w {
                        let idx = y * w + x;
                        let out_i = ly * w + x;

                        // np.gradient uses forward/backward diff on borders, central diff inside
                        let (pu_l, pu_r, pu_scale) = if w <= 1 {
                            (idx, idx, 1.0)
                        } else if x == 0 {
                            (idx, idx + 1, 1.0)
                        } else if x == w - 1 {
                            (idx - 1, idx, 1.0)
                        } else {
                            (idx - 1, idx + 1, 0.5)
                        };

                        let (pv_t, pv_b, pv_scale) = if h <= 1 {
                            (idx, idx, 1.0)
                        } else if y == 0 {
                            (idx, idx + w, 1.0)
                        } else if y == h - 1 {
                            (idx - w, idx, 1.0)
                        } else {
                            (idx - w, idx + w, 0.5)
                        };

                        let dux = (grid_ref[pu_r * 3] - grid_ref[pu_l * 3]) * pu_scale;
                        let duy = (grid_ref[pu_r * 3 + 1] - grid_ref[pu_l * 3 + 1]) * pu_scale;
                        let duz = (grid_ref[pu_r * 3 + 2] - grid_ref[pu_l * 3 + 2]) * pu_scale;

                        let dvx = (grid_ref[pv_b * 3] - grid_ref[pv_t * 3]) * pv_scale;
                        let dvy = (grid_ref[pv_b * 3 + 1] - grid_ref[pv_t * 3 + 1]) * pv_scale;
                        let dvz = (grid_ref[pv_b * 3 + 2] - grid_ref[pv_t * 3 + 2]) * pv_scale;

                        let nx = duy * dvz - duz * dvy;
                        let ny = duz * dvx - dux * dvz;
                        let nz = dux * dvy - duy * dvx;
                        let nn = (nx * nx + ny * ny + nz * nz).sqrt();
                        norm_chunk[out_i] = nn;

                        let px = grid_ref[idx * 3];
                        let py = grid_ref[idx * 3 + 1];
                        let pz = grid_ref[idx * 3 + 2];
                        let rn = (px * px + py * py + pz * pz).sqrt().max(1e-8);
                        if nn > 1e-8 {
                            let dot = (-px * nx - py * ny - pz * nz) / (nn * rn);
                            cos_chunk[out_i] = dot.abs();
                        } else {
                            cos_chunk[out_i] = 0.0;
                        }
                    }
                }
            });
        }
    });

    // 3. Incidence angle gate (matches fallback threshold in Python)
    out_mask.copy_from_slice(&stage_mask);
    if cos_thresh > 0.0 {
        let mut phys_count = 0usize;
        for i in 0..n_pix {
            if stage_mask[i] != 0 && cos_inc[i] >= cos_thresh && n_norm[i] > 1e-6 {
                phys_count += 1;
            }
        }
        let min_keep = 20usize.max((stage_count as f64 * 0.15) as usize);
        if phys_count >= min_keep {
            for i in 0..n_pix {
                if stage_mask[i] != 0 && (cos_inc[i] < cos_thresh || n_norm[i] <= 1e-6) {
                    out_mask[i] = 0;
                }
            }
        }
    }

    // 4. Pack valid points + physics weights
    let mut out_cnt = 0usize;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if out_mask[i] == 0 {
                continue;
            }
            let z = grid[i * 3 + 2];
            out_pts[out_cnt * 3] = grid[i * 3];
            out_pts[out_cnt * 3 + 1] = grid[i * 3 + 1];
            out_pts[out_cnt * 3 + 2] = z;
            let denom = z.max(0.30);
            out_weights[out_cnt] = cos_inc[i].clamp(0.1, 1.0) / (denom * denom);
            out_v_idx[out_cnt] = y as i64;
            out_u_idx[out_cnt] = x as i64;
            out_cnt += 1;
        }
    }
    out_cnt
}
