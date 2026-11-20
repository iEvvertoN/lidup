use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::thread;

#[derive(Copy, Clone)]
struct FloodCell {
    z: f64,
    idx: usize,
}

impl PartialEq for FloodCell {
    fn eq(&self, other: &Self) -> bool {
        self.z == other.z && self.idx == other.idx
    }
}
impl Eq for FloodCell {}

impl PartialOrd for FloodCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloodCell {
    // Min-heap ordering
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .z
            .partial_cmp(&self.z)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

/// Barnes Priority-Flood depression filling (exact water-level surface for ponding depth).
/// `z_grid`: ny * nx elevation grid (NaN for missing outside cells).
/// `valid_u8`: ny * nx mask (1 valid surface cell, 0 void).
/// Outputs `out_water_depth_m` (>= 0 on ponded cells).
pub(crate) fn priority_flood_ponding_impl(
    z_grid: &[f64],
    valid_u8: &[u8],
    nx: usize,
    ny: usize,
    out_water_depth: &mut [f64],
) {
    let n = nx * ny;
    out_water_depth.fill(0.0);
    if nx < 3 || ny < 3 {
        return;
    }

    let mut visited = vec![0u8; n];
    let mut heap = BinaryHeap::with_capacity(n / 4);

    // Seed boundary valid cells and cells bordering missing (NaN/void) domain edges
    for y in 0..ny {
        for x in 0..nx {
            let idx = y * nx + x;
            if valid_u8[idx] == 0 || !z_grid[idx].is_finite() {
                visited[idx] = 1;
                continue;
            }
            let is_border = x == 0
                || y == 0
                || x + 1 == nx
                || y + 1 == ny
                || valid_u8[(y - 1) * nx + x] == 0
                || valid_u8[(y + 1) * nx + x] == 0
                || valid_u8[y * nx + (x - 1)] == 0
                || valid_u8[y * nx + (x + 1)] == 0;

            if is_border {
                visited[idx] = 1;
                heap.push(FloodCell {
                    z: z_grid[idx],
                    idx,
                });
            }
        }
    }

    const DIRS: [(isize, isize); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    while let Some(cell) = heap.pop() {
        let cx = cell.idx % nx;
        let cy = cell.idx / nx;
        for (dx, dy) in DIRS {
            let nx_i = cx as isize + dx;
            let ny_i = cy as isize + dy;
            if nx_i < 0 || ny_i < 0 || nx_i >= nx as isize || ny_i >= ny as isize {
                continue;
            }
            let n_idx = (ny_i as usize) * nx + (nx_i as usize);
            if visited[n_idx] != 0 {
                continue;
            }
            visited[n_idx] = 1;
            let raw_z = z_grid[n_idx];
            if !raw_z.is_finite() {
                continue;
            }
            let spill_z = cell.z.max(raw_z);
            out_water_depth[n_idx] = (spill_z - raw_z).max(0.0);
            heap.push(FloodCell {
                z: spill_z,
                idx: n_idx,
            });
        }
    }
}

/// Checks oriented rectangular footprint clearance across rotated angles [0..180 deg]
/// on an obstacle occupancy grid, plus 360-degree 2D visibility raycast from `sensor_xy`.
/// `occ_u8`: ny * nx (1 = obstacle, 0 = free), `domain_u8`: ny * nx (1 = inside room floor).
/// Outputs `out_passable_u8`: (1 = box fits centered here in >= 1 orientation, 0 = blocked)
/// Outputs `out_visible_u8`: (1 = visible from sensor_ix/iy, 0 = occluded/shadowed).
pub(crate) fn clearance_and_viewshed_impl(
    occ_u8: &[u8],
    domain_u8: &[u8],
    nx: usize,
    ny: usize,
    cell_m: f64,
    half_w_m: f64,
    half_d_m: f64,
    sensor_ix: isize,
    sensor_iy: isize,
    out_passable: &mut [u8],
    out_visible: &mut [u8],
) {
    out_passable.fill(0);
    out_visible.fill(0);

    // Precompute relative integer cell offsets for 12 angles in [0, 180)
    let mut footprints: Vec<Vec<(isize, isize)>> = Vec::with_capacity(12);
    let max_r_cells = (((half_w_m * half_w_m + half_d_m * half_d_m).sqrt()) / cell_m).ceil() as isize;
    for a_idx in 0..12 {
        let theta = (a_idx as f64) * std::f64::consts::PI / 12.0;
        let ct = theta.cos();
        let st = theta.sin();
        let mut pts = Vec::new();
        for dy in -max_r_cells..=max_r_cells {
            for dx in -max_r_cells..=max_r_cells {
                let lx = (dx as f64) * cell_m;
                let ly = (dy as f64) * cell_m;
                let rx = lx * ct + ly * st;
                let ry = -lx * st + ly * ct;
                if rx.abs() <= half_w_m && ry.abs() <= half_d_m {
                    pts.push((dx, dy));
                }
            }
        }
        if pts.is_empty() {
            pts.push((0, 0));
        }
        footprints.push(pts);
    }

    let n_threads = thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(4)
        .min(ny.max(1));
    let chunk_rows = (ny + n_threads - 1) / n_threads.max(1);
    let fp_ref = &footprints;

    thread::scope(|s| {
        for (c_idx, pass_chunk) in out_passable.chunks_mut(chunk_rows * nx).enumerate() {
            let y_start = c_idx * chunk_rows;
            s.spawn(move || {
                let rows = pass_chunk.len() / nx;
                for ly in 0..rows {
                    let y = y_start + ly;
                    for x in 0..nx {
                        let idx = y * nx + x;
                        if domain_u8[idx] == 0 || occ_u8[idx] != 0 {
                            continue;
                        }
                        let mut fits = false;
                        'angle_loop: for fp in fp_ref {
                            for &(dx, dy) in fp {
                                let xx = x as isize + dx;
                                let yy = y as isize + dy;
                                if xx < 0 || yy < 0 || xx >= nx as isize || yy >= ny as isize {
                                    continue 'angle_loop;
                                }
                                let ni = (yy as usize) * nx + (xx as usize);
                                if domain_u8[ni] == 0 || occ_u8[ni] != 0 {
                                    continue 'angle_loop;
                                }
                            }
                            fits = true;
                            break;
                        }
                        if fits {
                            pass_chunk[ly * nx + x] = 1;
                        }
                    }
                }
            });
        }
    });

    // 2D raycast viewshed from sensor cell across all valid domain cells
    if sensor_ix >= 0 && sensor_iy >= 0 && sensor_ix < nx as isize && sensor_iy < ny as isize {
        let sx_f = sensor_ix as f64;
        let sy_f = sensor_iy as f64;
        for y in 0..ny {
            for x in 0..nx {
                let idx = y * nx + x;
                if domain_u8[idx] == 0 {
                    continue;
                }
                let dx = (x as f64) - sx_f;
                let dy = (y as f64) - sy_f;
                let steps = dx.abs().max(dy.abs()).ceil() as usize;
                if steps == 0 {
                    out_visible[idx] = 1;
                    continue;
                }
                let mut blocked = false;
                for st in 1..steps {
                    let t = (st as f64) / (steps as f64);
                    let rx = (sx_f + dx * t).round() as usize;
                    let ry = (sy_f + dy * t).round() as usize;
                    if occ_u8[ry * nx + rx] != 0 {
                        blocked = true;
                        break;
                    }
                }
                if !blocked {
                    out_visible[idx] = 1;
                }
            }
        }
    }
}
