#ifndef LIDUP_CORE_H
#define LIDUP_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Returns ABI version of lidup_core (currently 2).
uint32_t lidup_core_abi_version(void);

/// Converts Float32 depth buffer (meters, arbitrary byte stride) to UInt16 millimeters.
int32_t lidup_depth_float32_to_mm_u16(
    const uint8_t *src_bytes,
    size_t src_byte_count,
    int32_t src_stride_bytes,
    int32_t width,
    int32_t height,
    uint16_t *out_mm
);

/// Computes mean depth (meters) in the central 5x5 region of a Float32 depth buffer.
/// Returns negative value if no valid samples are present.
float lidup_center_depth_m_f32(
    const uint8_t *src_bytes,
    size_t src_byte_count,
    int32_t src_stride_bytes,
    int32_t width,
    int32_t height
);

/// Computes Laplacian variance sharpness score on an 8-bit luma plane.
float lidup_sharpness_laplacian_u8(
    const uint8_t *y_bytes,
    size_t y_byte_count,
    int32_t stride_bytes,
    int32_t width,
    int32_t height,
    int32_t sample_step
);

/// Computes Laplacian variance sharpness score on a 32BGRA buffer (channel 1 = G).
float lidup_sharpness_laplacian_bgra_u8(
    const uint8_t *bgra_bytes,
    size_t bgra_byte_count,
    int32_t stride_bytes,
    int32_t width,
    int32_t height,
    int32_t sample_step
);

/// Computes inverse-variance weighted stationary GNSS average.
/// Returns valid sample count (>0 on success), or 0 if no valid samples (<100m h_acc).
int32_t lidup_weighted_gnss_average(
    const double *lat,
    const double *lon,
    const double *alt,
    const double *h_acc,
    const double *v_acc,
    int32_t count,
    double *out_lat,
    double *out_lon,
    double *out_alt,
    double *out_h_acc_mean,
    double *out_v_acc_mean
);

/// Returns 1 if column-major 4x4 SE(3) transforms moved >= min_dist OR rotated >= min_angle_deg, else 0.
int32_t lidup_se3_moved_enough(
    const float *prev_col_major_16,
    const float *curr_col_major_16,
    float min_dist_m,
    float min_angle_deg
);

/// RGB-guided bilateral depth upsampling + bimodal edge unmixing.
int32_t lidup_upsample_depth_guided_f32(
    const float *depth_m_f32,
    const uint8_t *valid_lo_u8,
    const uint8_t *conf_u8,
    int32_t has_conf,
    const float *rgb_lo_f32,
    const float *rgb_hi_f32,
    int32_t d_w,
    int32_t d_h,
    int32_t u_w,
    int32_t u_h,
    int32_t scale,
    float edge_jump_m,
    float *out_depth_f32,
    uint8_t *out_mask_u8
);

/// Edge jump filtering, backprojection, surface normal incidence pruning, and point packing.
int32_t lidup_extract_depth_points_f64(
    const double *depth_m_f64,
    const uint8_t *in_mask_u8,
    int32_t width,
    int32_t height,
    double fx,
    double fy,
    double cx,
    double cy,
    double edge_jump_m,
    double cos_thresh,
    uint8_t *out_mask_u8,
    double *out_pts_f64,
    double *out_weights_f64,
    int64_t *out_v_idx,
    int64_t *out_u_idx,
    int32_t *out_count
);

/// Multi-threaded bilinear resize from packed RGB uint8 to float32 [0, 1].
int32_t lidup_resize_rgb_u8_to_f32(
    const uint8_t *src_rgb_u8,
    int32_t src_w,
    int32_t src_h,
    int32_t dst_w,
    int32_t dst_h,
    float *out_rgb_f32
);

/// Multi-threaded RGB-to-depth camera pose warp.
int32_t lidup_warp_rgb_to_depth_f32(
    const float *rgb_hi_f32,
    const float *d_hi_filled_f32,
    const double *rel_mat4x4_f64,
    double fx,
    double fy,
    double cx,
    double cy,
    int32_t width,
    int32_t height,
    float *out_rgb_f32
);

/// Projects N 3D points into one 12 MP uint8 frame in parallel with zero float64 image allocation.
int32_t lidup_colorize_hires_frame_u8(
    const double *pts_arkit_f64,
    const double *norms_arkit_f64,
    int32_t n_pts,
    const double *pose_rgb_4x4_f64,
    const uint8_t *hi_rgb_u8,
    int32_t hi_w,
    int32_t hi_h,
    const float *depth_f32,
    int32_t d_w,
    int32_t d_h,
    double fx_hi,
    double fy_hi,
    double cx_hi,
    double cy_hi,
    double max_depth_m,
    double occ_tol_m,
    double sharp_imu_w,
    double *best_score_f64,
    double *out_colors_f64
);

/// Multi-view stereo photometric depth refinement sweep (parallelized across dz steps).
int32_t lidup_mvs_sweep_f32(
    const float *d0_f32,
    const uint8_t *valid_u8,
    const float *ref_nhp_f32,
    int32_t d_w,
    int32_t d_h,
    float fx,
    float fy,
    float cx,
    float cy,
    int32_t n_neighbors,
    const float *const *nb_nhp_ptrs,
    const float *const *nb_depth_ptrs,
    const float *nb_t_mats_f32,
    const int32_t *nb_dims_i32,
    const float *nb_intr_f32,
    float max_shift_m,
    int32_t n_steps,
    float *out_dz_raw_f32,
    float *out_best_score_f32,
    float *out_best_vis_f32
);

/// Cloth Simulation Filter (Zhang et al., 2016) terrain surface solver.
int32_t lidup_csf_cloth_filter_f64(
    const double *terrain_filled_f64,
    int32_t nx,
    int32_t ny,
    double z_inv_max,
    int32_t rigidness,
    int32_t iterations,
    double gravity_step_m,
    double *out_dtm_f64
);

#ifdef __cplusplus
}
#endif

#endif /* LIDUP_CORE_H */
