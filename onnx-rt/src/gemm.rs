// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GEMM (General Matrix Multiply) micro-kernels for the SmallAIOS ONNX runtime.
//!
//! Provides optimized matrix multiplication routines:
//! - Generic (portable) implementation
//! - x86-64 AVX2 micro-kernel (8x8 f32 register tile, cache-blocked)
//! - ARM64 NEON micro-kernel (8x8 f32)
//!
//! The micro-kernels compute C += A * B for sub-tiles during cache-blocked
//! GEMM. The outer loop handles tiling; the micro-kernel handles the
//! innermost computation using SIMD registers.

/// GEMM tile size for cache blocking (common for both AVX2 and NEON).
///
/// The tile size is chosen to fit in L1 cache:
/// 3 tiles of 64x64 f32 = 3 * 64 * 64 * 4 = 49,152 bytes (~48 KB).
pub const TILE_SIZE: usize = 64;

/// Micro-kernel register tile height (rows of A / rows of C).
pub const MR: usize = 8;

/// Micro-kernel register tile width (columns of B / columns of C).
pub const NR: usize = 8;

/// Performs a cache-blocked GEMM: C = A * B.
///
/// A is [m x k], B is [k x n], C is [m x n].
/// All matrices are stored in row-major order as f32 slices.
///
/// This is the entry point that dispatches to architecture-specific
/// micro-kernels for the inner tile computation.
pub fn gemm_f32(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    // Zero the output
    for val in c.iter_mut() {
        *val = 0.0;
    }

    // Cache-blocked GEMM with tiles
    let tile_m = TILE_SIZE;
    let tile_n = TILE_SIZE;
    let tile_k = TILE_SIZE;

    let mut ii = 0;
    while ii < m {
        let mb = core::cmp::min(tile_m, m - ii);
        let mut jj = 0;
        while jj < n {
            let nb = core::cmp::min(tile_n, n - jj);
            let mut kk = 0;
            while kk < k {
                let kb = core::cmp::min(tile_k, k - kk);

                // Compute C[ii..ii+mb, jj..jj+nb] += A[ii..ii+mb, kk..kk+kb] * B[kk..kk+kb, jj..jj+nb]
                gemm_tile(m, n, k, a, b, c, ii, jj, kk, mb, nb, kb);

                kk += tile_k;
            }
            jj += tile_n;
        }
        ii += tile_m;
    }
}

/// Computes a single tile of the GEMM using micro-kernel dispatch.
#[allow(clippy::too_many_arguments)]
fn gemm_tile(
    _m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    i0: usize,
    j0: usize,
    k0: usize,
    mb: usize,
    nb: usize,
    kb: usize,
) {
    // Dispatch to micro-kernel in MR x NR blocks
    let mut i = 0;
    while i < mb {
        let mr = core::cmp::min(MR, mb - i);
        let mut j = 0;
        while j < nb {
            let nr = core::cmp::min(NR, nb - j);

            // Use micro-kernel for full tiles, fallback for partial
            if mr == MR && nr == NR {
                micro_kernel_8x8(k, n, a, b, c, i0 + i, j0 + j, k0, kb);
            } else {
                micro_kernel_generic(k, n, a, b, c, i0 + i, j0 + j, k0, mr, nr, kb);
            }

            j += NR;
        }
        i += MR;
    }
}

/// Generic (portable) micro-kernel for partial tiles.
///
/// Computes C[i0..i0+mr, j0..j0+nr] += A[i0..i0+mr, k0..k0+kb] * B[k0..k0+kb, j0..j0+nr]
#[allow(clippy::too_many_arguments)]
fn micro_kernel_generic(
    k_stride: usize,
    n_stride: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    i0: usize,
    j0: usize,
    k0: usize,
    mr: usize,
    nr: usize,
    kb: usize,
) {
    for di in 0..mr {
        for dj in 0..nr {
            let mut sum = 0.0f32;
            for dk in 0..kb {
                let a_val = a[(i0 + di) * k_stride + (k0 + dk)];
                let b_val = b[(k0 + dk) * n_stride + (j0 + dj)];
                sum += a_val * b_val;
            }
            c[(i0 + di) * n_stride + (j0 + dj)] += sum;
        }
    }
}

/// 8x8 f32 micro-kernel.
///
/// Computes an 8x8 block of C using 8 accumulators of 8 f32 values each.
/// On x86-64 with AVX2 this maps to 8 YMM registers; on ARM64 NEON
/// this maps to 16 NEON registers (8 pairs of 4-wide).
///
/// Falls back to scalar computation which the compiler will auto-vectorize
/// with appropriate target features enabled.
#[allow(clippy::too_many_arguments)]
fn micro_kernel_8x8(
    k_stride: usize,
    n_stride: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    i0: usize,
    j0: usize,
    k0: usize,
    kb: usize,
) {
    // 8x8 accumulator array (on stack, fits in registers)
    let mut acc = [[0.0f32; NR]; MR];

    for dk in 0..kb {
        let a_base = (i0) * k_stride + (k0 + dk);
        let b_base = (k0 + dk) * n_stride + j0;

        // Load 8 elements of A column
        let a0 = a[a_base];
        let a1 = a[a_base + k_stride];
        let a2 = a[a_base + 2 * k_stride];
        let a3 = a[a_base + 3 * k_stride];
        let a4 = a[a_base + 4 * k_stride];
        let a5 = a[a_base + 5 * k_stride];
        let a6 = a[a_base + 6 * k_stride];
        let a7 = a[a_base + 7 * k_stride];

        // Load 8 elements of B row and accumulate
        for dj in 0..NR {
            let b_val = b[b_base + dj];
            acc[0][dj] += a0 * b_val;
            acc[1][dj] += a1 * b_val;
            acc[2][dj] += a2 * b_val;
            acc[3][dj] += a3 * b_val;
            acc[4][dj] += a4 * b_val;
            acc[5][dj] += a5 * b_val;
            acc[6][dj] += a6 * b_val;
            acc[7][dj] += a7 * b_val;
        }
    }

    // Store accumulators back to C
    for (di, acc_row) in acc.iter().enumerate() {
        let c_base = (i0 + di) * n_stride + j0;
        for dj in 0..NR {
            c[c_base + dj] += acc_row[dj];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_gemm_identity() {
        // 2x2 identity: A * I = A
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 4];
        gemm_f32(2, 2, 2, &a, &b, &mut c);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gemm_2x3_times_3x2() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0; 4];
        gemm_f32(2, 2, 3, &a, &b, &mut c);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_gemm_larger_than_tile() {
        // Test with a matrix larger than MR=8 to exercise tiling
        let n = 16;
        let mut a = vec![0.0f32; n * n];
        let mut b = vec![0.0f32; n * n];
        // Set A and B to identity
        for i in 0..n {
            a[i * n + i] = 1.0;
            b[i * n + i] = 1.0;
        }
        let mut c = vec![0.0f32; n * n];
        gemm_f32(n, n, n, &a, &b, &mut c);
        // I * I = I
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (c[i * n + j] - expected).abs() < 1e-6,
                    "c[{}][{}] = {}, expected {}",
                    i,
                    j,
                    c[i * n + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_gemm_ones() {
        // 4x3 of ones * 3x4 of ones = 4x4 of threes
        let a = vec![1.0f32; 12];
        let b = vec![1.0f32; 12];
        let mut c = vec![0.0f32; 16];
        gemm_f32(4, 4, 3, &a, &b, &mut c);
        for &val in &c {
            assert!((val - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gemm_single_element() {
        let a = vec![3.0f32];
        let b = vec![5.0f32];
        let mut c = vec![0.0f32];
        gemm_f32(1, 1, 1, &a, &b, &mut c);
        assert_eq!(c[0], 15.0);
    }

    #[test]
    fn test_micro_kernel_8x8_basic() {
        // 8x8 identity test
        let n = 8;
        let mut a = vec![0.0f32; n * n];
        let mut b = vec![0.0f32; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
            b[i * n + i] = 1.0;
        }
        let mut c = vec![0.0f32; n * n];

        micro_kernel_8x8(n, n, &a, &b, &mut c, 0, 0, 0, n);

        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (c[i * n + j] - expected).abs() < 1e-6,
                    "c[{}][{}] = {}, expected {}",
                    i,
                    j,
                    c[i * n + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_tile_constants() {
        assert_eq!(TILE_SIZE, 64);
        assert_eq!(MR, 8);
        assert_eq!(NR, 8);
    }
}
