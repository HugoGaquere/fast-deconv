#include <fast_deconv/common/multi_frequency.hpp>

#include "fast_deconv/linalg/pseudo_inverse.hpp"

namespace fast_deconv::kernel {

// Compute A[f, o] = xdes[f, o] * sqrt(jones_norm[f, peak_row, peak_col]) * sqrt(weights[f])
// jones_norm row-major: [n_freq, nrow, ncol]
// xdes row-major:       [n_freq, n_order]
// weights row-major:    [n_freq]
// A row-major:          [n_freq, n_order]
__global__ void compute_spectral_matrix_kernel(float* SAX, float* A, const float* xdes,
                                               const float* jones_norm, const float* weights,
                                               int n_freq, int n_order, int jn_freq_stride,
                                               int jn_peak_offset)
{
  const int n = n_freq * n_order;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
    const int f = idx / n_order;
    float sqrt_jn = sqrtf(jones_norm[f * jn_freq_stride + jn_peak_offset]);
    float sax = xdes[idx] * sqrt_jn;
    SAX[idx] = sax;
    A[idx] = sax * sqrtf(weights[f]);
  }
}

// Fused kernel: compute spectral coefficients at peak location
//   Step 1: wy[f] = sqrt_w[f] * dirty[f, peak_row, peak_col]
//   Step 2: compact[o] = sum_f A_pinv(o, f) * wy[f]   (one thread per order, loop over freq)
//   Step 3: per_chan[f] = sum_o A(f, o) * compact[o]    (one thread per freq, loop over order)
//
// A_pinv: col-major [n_order, n_freq]  (from compute_pseudo_inverse)
// A:      row-major [n_freq, n_order]  (from compute_spectral_matrix_kernel)
constexpr int SPECTRAL_BLOCK_SIZE = 128;
__global__ void compute_spectral_coeffs_kernel(float* compact_out, float* per_chan_out,
                                               const float* dirty, const float* weights,
                                               const float* A_pinv, const float* SAX, int n_freq,
                                               int n_order, int dirty_freq_stride,
                                               int dirty_peak_offset)
{
  extern __shared__ float smem[];
  float* s_wy = smem;                // [n_freq]
  float* s_compact = s_wy + n_freq;  // [n_order]

  const int tid = threadIdx.x;

  // Step 1: build weighted vector at peak pixel
  for (int f = tid; f < n_freq; f += SPECTRAL_BLOCK_SIZE) {
    s_wy[f] = sqrtf(weights[f]) * dirty[f * dirty_freq_stride + dirty_peak_offset];
  }
  __syncthreads();

  // Step 2: compact[o] = A_pinv[o, :] . wy
  // A_pinv col-major [n_order, n_freq]: element (o, f) at [f * n_order + o]
  // One thread per order, each loops over all frequencies
  if (tid < n_order) {
    float sum = 0.0f;
    for (int f = 0; f < n_freq; f++) {
      sum += A_pinv[f * n_order + tid] * s_wy[f];
    }
    s_compact[tid] = sum;
    compact_out[tid] = sum;
  }
  __syncthreads();

  // Step 3: per_chan[f] = SAX[f, :] . compact
  // SAX = sqrt(jn) * xdes, row-major [n_freq, n_order]: element (f, o) at [f * n_order + o]
  // One thread per freq, each loops over all orders
  for (int f = tid; f < n_freq; f += SPECTRAL_BLOCK_SIZE) {
    float sum = 0.0f;
    for (int o = 0; o < n_order; o++) {
      sum += SAX[f * n_order + o] * s_compact[o];
    }
    per_chan_out[f] = sum;
  }
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::multi_frequency {

void fit_coefficients(const core::resources& resources, const core::stream_resources& stream_res,
                      const core::device_span3d<float> residual,
                      const core::device_span3d<float> jones_norm,
                      const core::device_vect<float> weights_freq,
                      const core::device_span2d<float> xdes, const std::pair<int, int> peak_coords,
                      core::device_vect<float> compact_coeffs_out,
                      core::device_vect<float> coeffs_per_chan_out)
{
  const int n_freq = xdes.extent(0);
  const int n_order = xdes.extent(1);
  const auto [peak_row, peak_col] = peak_coords;

  float* d_SAX = resources.alloc_async<float>(n_freq * n_order, stream_res);
  float* d_A = resources.alloc_async<float>(n_freq * n_order, stream_res);
  float* d_A_pinv = resources.alloc_async<float>(n_order * n_freq, stream_res);

  //   1. SAX = sqrt(jones_norm[f, peak]) * xdes[f, o],  A = sqrt(w[f]) * SAX
  const int jn_freq_stride = jones_norm.extent(1) * jones_norm.extent(2);
  const int jn_peak_offset = peak_row * jones_norm.extent(2) + peak_col;
  kernel::compute_spectral_matrix_kernel<<<1, n_freq * n_order, 0, stream_res.cuda_stream>>>(
      d_SAX, d_A, xdes.data_handle(), jones_norm.data_handle(), weights_freq.data_handle(), n_freq,
      n_order, jn_freq_stride, jn_peak_offset);

  //   2. A_pinv = inv(A^T A) @ A^T
  linalg::compute_pseudo_inverse(resources, stream_res, d_A, d_A_pinv, n_freq, n_order);

  //   3. wy = sqrt_w * dirty[f, peak]  →  compact = A_pinv @ wy  →  per_chan = SAX @ compact
  const int dirty_freq_stride = residual.extent(1) * residual.extent(2);
  const int dirty_peak_offset = peak_row * residual.extent(2) + peak_col;
  size_t smem_bytes = (n_freq + n_order) * sizeof(float);
  kernel::compute_spectral_coeffs_kernel<<<1, kernel::SPECTRAL_BLOCK_SIZE, smem_bytes,
                                           stream_res.cuda_stream>>>(
      compact_coeffs_out.data_handle(), coeffs_per_chan_out.data_handle(), residual.data_handle(),
      weights_freq.data_handle(), d_A_pinv, d_SAX, n_freq, n_order, dirty_freq_stride,
      dirty_peak_offset);


  // Step 4: subtract PSF from dirty
  // overlap_region ovr = compute_overlap_region(peak_row, peak_col, nrow, ncol, psf_nrow,
  // psf_ncol); const int total = n_freq * ovr.w * ovr.h; FD_LOG_DEBUG(
  //     "subtract_component: peak=({},{}) n_freq={} n_order={} dirty={}x{} psf={}x{} roi={}x{} "
  //     "total={} blocks={}",
  //     peak_row, peak_col, n_freq, n_order, nrow, ncol, psf_nrow, psf_ncol, ovr.w, ovr.h, total,
  //     CEIL_DIV(total, 256));
  // spectral_psf_subtract_kernel<<<CEIL_DIV(total, 256), 256, 0, cuda_stream>>>(
  //     dirty, psf, d_per_chan, ovr, gain, n_freq, nrow * ncol, psf_nrow * psf_ncol);

  // Free temporaries
  // resources.free_async(d_per_chan, stream_res);
  resources.free_async(d_A_pinv, stream_res);
  resources.free_async(d_A, stream_res);
  resources.free_async(d_SAX, stream_res);
}

}  // namespace fast_deconv::multi_frequency