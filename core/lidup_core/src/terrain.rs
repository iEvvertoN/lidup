/// Cloth Simulation Filter (Zhang et al., 2016) running in cache-friendly Rust loops.
pub(crate) fn csf_cloth_simulate_impl(
    terrain_filled: &[f64],
    nx: usize,
    ny: usize,
    z_inv_max: f64,
    rigidness: i32,
    iterations: i32,
    gravity_step_m: f64,
    out_dtm: &mut [f64],
) {
    let n = nx * ny;
    let mut cloth = vec![z_inv_max + 0.25; n];
    let mut next_cloth = vec![0.0f64; n];
    let mut iter_start = vec![0.0f64; n];
    let passes = rigidness.clamp(1, 3) as usize;
    let max_iter = iterations.max(20) as usize;

    for _ in 0..max_iter {
        iter_start.copy_from_slice(&cloth);
        for i in 0..n {
            let tf = terrain_filled[i];
            if cloth[i] > tf {
                cloth[i] = (cloth[i] - gravity_step_m).max(tf);
            } else {
                cloth[i] = cloth[i].max(tf);
            }
        }

        for _ in 0..passes {
            for y in 0..ny {
                let yu = y.saturating_sub(1);
                let yd = (y + 1).min(ny - 1);
                for x in 0..nx {
                    let i = y * nx + x;
                    let c = cloth[i];
                    let tf = terrain_filled[i];
                    if c <= tf + 1e-6 {
                        next_cloth[i] = c;
                        continue;
                    }
                    let xl = x.saturating_sub(1);
                    let xr = (x + 1).min(nx - 1);
                    let avg = 0.25
                        * (cloth[yu * nx + x]
                            + cloth[yd * nx + x]
                            + cloth[y * nx + xl]
                            + cloth[y * nx + xr]);
                    next_cloth[i] = (c + 0.5 * (avg - c)).max(tf);
                }
            }
            cloth.copy_from_slice(&next_cloth);
        }

        let mut max_delta = 0.0f64;
        for i in 0..n {
            let d = (cloth[i] - iter_start[i]).abs();
            if d > max_delta {
                max_delta = d;
            }
        }
        if max_delta < 1e-4 {
            break;
        }
    }

    for i in 0..n {
        out_dtm[i] = -cloth[i];
    }
}
