/// Converts raw Float32 depth bytes (with row stride) into tightly packed UInt16 millimeters.
pub(crate) fn convert_float32_depth_to_mm_u16_slice(
    src: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    out: &mut [u16],
) {
    for y in 0..h {
        let row_off = y * stride;
        if row_off + w * 4 > src.len() {
            break;
        }
        let dst_row = &mut out[y * w..(y + 1) * w];
        for x in 0..w {
            let b_off = row_off + x * 4;
            let meters = f32::from_le_bytes([src[b_off], src[b_off + 1], src[b_off + 2], src[b_off + 3]]);
            if meters.is_finite() && meters > 0.0 {
                let mm = (meters * 1000.0).round();
                dst_row[x] = mm.clamp(0.0, u16::MAX as f32) as u16;
            } else {
                dst_row[x] = 0;
            }
        }
    }
}

/// Computes average depth (meters) on the 5x5 center region of a Float32 depth buffer.
pub(crate) fn center_depth_meters_slice(src: &[u8], stride: usize, w: usize, h: usize) -> Option<f32> {
    if w < 5 || h < 5 || stride < w * 4 {
        return None;
    }
    let cy = h / 2;
    let cx = w / 2;
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for dy in -2isize..=2 {
        let y = (cy as isize + dy) as usize;
        let row_off = y * stride;
        if row_off + (cx + 3) * 4 > src.len() {
            continue;
        }
        for dx in -2isize..=2 {
            let x = (cx as isize + dx) as usize;
            let b_off = row_off + x * 4;
            let d = f32::from_le_bytes([src[b_off], src[b_off + 1], src[b_off + 2], src[b_off + 3]]);
            if d.is_finite() && d > 0.05 && d < 12.0 {
                sum += d;
                count += 1;
            }
        }
    }
    if count > 0 {
        Some(sum / (count as f32))
    } else {
        None
    }
}

/// Fast Laplacian variance sharpness metric on an 8-bit strided image channel.
pub(crate) fn sharpness_laplacian_channel_slice(
    src: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    bytes_per_pixel: usize,
    channel_offset: usize,
    step: usize,
) -> f32 {
    if w < 4 || h < 4 || bytes_per_pixel == 0 || channel_offset >= bytes_per_pixel {
        return 0.0;
    }
    let step = step.max(1);
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;

    let mut y = step;
    while y + step < h {
        if (y + step) * stride + w * bytes_per_pixel > src.len() {
            break;
        }
        let row_c = y * stride;
        let row_u = (y - step) * stride;
        let row_d = (y + step) * stride;
        let mut x = step;
        while x + step < w {
            let cx = x * bytes_per_pixel + channel_offset;
            let lx = (x - step) * bytes_per_pixel + channel_offset;
            let rx = (x + step) * bytes_per_pixel + channel_offset;
            let lap = (src[row_u + cx] as i32)
                + (src[row_d + cx] as i32)
                + (src[row_c + lx] as i32)
                + (src[row_c + rx] as i32)
                - 4 * (src[row_c + cx] as i32);
            let lf = lap as f64;
            sum += lf;
            sum_sq += lf * lf;
            count += 1;
            x += step;
        }
        y += step;
    }
    if count == 0 {
        return 0.0;
    }
    let mean = sum / (count as f64);
    ((sum_sq / (count as f64)) - mean * mean).max(0.0) as f32
}

/// Weighted inverse-variance average over GNSS samples (shared logic for iOS & georef).
pub(crate) fn weighted_gnss_average_slice(
    lat: &[f64],
    lon: &[f64],
    alt: &[f64],
    h_acc: &[f64],
    v_acc: &[f64],
) -> Option<(usize, f64, f64, f64, f64, f64)> {
    let n = lat.len().min(lon.len()).min(alt.len()).min(h_acc.len()).min(v_acc.len());
    let mut valid_count = 0usize;
    let mut w_lat = 0.0f64;
    let mut w_lon = 0.0f64;
    let mut w_h_tot = 0.0f64;
    let mut w_alt = 0.0f64;
    let mut w_v_tot = 0.0f64;
    let mut h_acc_sum = 0.0f64;
    let mut v_acc_sum = 0.0f64;
    let mut v_count = 0usize;
    let mut raw_alt_sum = 0.0f64;

    for i in 0..n {
        let ha = h_acc[i];
        if !(ha.is_finite() && ha > 0.0 && ha < 100.0) {
            continue;
        }
        valid_count += 1;
        let wh = 1.0 / (ha * ha).max(0.25);
        w_lat += lat[i] * wh;
        w_lon += lon[i] * wh;
        w_h_tot += wh;
        h_acc_sum += ha;
        raw_alt_sum += alt[i];

        let va = v_acc[i];
        if va.is_finite() && va > 0.0 {
            let wv = 1.0 / (va * va).max(0.25);
            w_alt += alt[i] * wv;
            w_v_tot += wv;
            v_acc_sum += va;
            v_count += 1;
        }
    }

    if valid_count == 0 || w_h_tot <= 0.0 {
        return None;
    }
    let mean_alt = if w_v_tot > 0.0 {
        w_alt / w_v_tot
    } else {
        raw_alt_sum / (valid_count as f64)
    };
    let mean_v_acc = if v_count > 0 {
        v_acc_sum / (v_count as f64)
    } else {
        -1.0
    };
    Some((
        valid_count,
        w_lat / w_h_tot,
        w_lon / w_h_tot,
        mean_alt,
        h_acc_sum / (valid_count as f64),
        mean_v_acc,
    ))
}

/// Checks whether two column-major 4x4 ARKit transforms moved >= min_dist OR rotated forward vector >= min_angle_deg.
pub(crate) fn se3_moved_enough_col_major(
    prev_cm16: &[f32; 16],
    curr_cm16: &[f32; 16],
    min_dist: f32,
    min_angle_deg: f32,
) -> bool {
    let dx = curr_cm16[12] - prev_cm16[12];
    let dy = curr_cm16[13] - prev_cm16[13];
    let dz = curr_cm16[14] - prev_cm16[14];
    if (dx * dx + dy * dy + dz * dz).sqrt() >= min_dist {
        return true;
    }

    // Forward vector is -columns.2 (-m[8], -m[9], -m[10])
    let mut f0 = [-prev_cm16[8], -prev_cm16[9], -prev_cm16[10]];
    let mut f1 = [-curr_cm16[8], -curr_cm16[9], -curr_cm16[10]];
    let n0 = (f0[0] * f0[0] + f0[1] * f0[1] + f0[2] * f0[2]).sqrt().max(1e-8);
    let n1 = (f1[0] * f1[0] + f1[1] * f1[1] + f1[2] * f1[2]).sqrt().max(1e-8);
    for k in 0..3 {
        f0[k] /= n0;
        f1[k] /= n1;
    }
    let cosine = (f0[0] * f1[0] + f0[1] * f1[1] + f0[2] * f1[2]).clamp(-1.0, 1.0);
    let min_rad = min_angle_deg * std::f32::consts::PI / 180.0;
    cosine.acos() >= min_rad
}
