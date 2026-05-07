#pragma once
#include <cufft.h>

#include <iostream>
#include <utility>
#include <vector>

#ifndef CUFFT_CALL
#define CUFFT_CALL(call)                                               \
  {                                                                    \
    auto status = static_cast<cufftResult>(call);                      \
    if (status != CUFFT_SUCCESS)                                       \
      fprintf(stderr,                                                  \
              "ERROR: CUFFT call \"%s\" in line %d of file %s failed " \
              "with "                                                  \
              "code (%d).\n",                                          \
              #call, __LINE__, __FILE__, status);                      \
  }
#endif  // CUFFT_CALL

using complex_type = cufftComplex;

namespace fast_deconv::linalg {

/**
 * @brief Generic cuFFT convolution context.
 *
 * Builds 2D R2C/C2R FFT plans over a padded grid (one R2C, N C2R) and exposes
 * the maximum workspace size required across all plans via @ref required_work_size.
 *
 * The caller is responsible for:
 *   1. Allocating @ref required_work_size bytes from their pool of choice.
 *   2. Calling @ref set_work_area to bind that buffer to every plan.
 *   3. Calling @ref set_stream once per launch (or per stream change).
 *
 * Plans are destroyed with the context.
 */
struct convolve_ctx {
  int input_nrow = 0, input_ncol = 0;       // unpadded input size
  int padding_nrow = 0, padding_ncol = 0;   // input start offset within padded buffer
                                            // (= (padded - input) / 2; far side gets one extra
                                            // zero pixel when the difference is odd)
  int padded_nrow = 0, padded_ncol = 0;     // padded spatial size (rounded up to next 7-smooth)
  int freq_nrow = 0, freq_ncol = 0;         // half-complex frequency size
  int forward_batch = 0;                    // batch size for the R2C plan
  int backward_batch = 0;                   // batch size for the C2R plan(s)
  std::vector<cufftHandle> plans_forward;   // forward (R2C) plans
  std::vector<cufftHandle> plans_backward;  // backward (C2R) plans
  size_t work_size = 0;                     // max workspace size across all plans (bytes)
  void* work_area = nullptr;                // shared cuFFT workspace (caller-managed)

  convolve_ctx() = default;

  /**
   * @brief Create plans for a padded 2D grid and disable cuFFT auto-allocation.
   *
   * Forward and backward plans can have different batch sizes, e.g. forward=1
   * to FFT a single image once, backward=N to IFFT N filtered spectra in a
   * single batched call. The shared workspace is *not* allocated here; the
   * caller must allocate @c required_work_size bytes and pass them to
   * @ref set_work_area before executing any plan.
   *
   * @param input_nrow        Unpadded input rows.
   * @param input_ncol        Unpadded input cols.
   * @param forward_batch     Batch size for the R2C plan.
   * @param backward_batch    Batch size shared by all C2R plans.
   * @param n_backward_plans  Number of C2R plans to create (e.g. 1 for conv, 2 for conv + conv^2).
   * @param padding           FFT padding factor (e.g. 1.5).
   */
  convolve_ctx(int input_nrow, int input_ncol, int forward_batch, int backward_batch, int n_backward_plans,
               float padding);

  convolve_ctx(const convolve_ctx&) = delete;
  convolve_ctx& operator=(const convolve_ctx&) = delete;
  convolve_ctx(convolve_ctx&&) noexcept = default;
  convolve_ctx& operator=(convolve_ctx&&) noexcept = default;

  ~convolve_ctx()
  {
    for (auto p : plans_forward) CUFFT_CALL(cufftDestroy(p));
    for (auto p : plans_backward) CUFFT_CALL(cufftDestroy(p));
  }

  /// Required size in bytes of the shared workspace buffer.
  size_t required_work_size() const { return work_size; }

  /// Bind a single shared workspace buffer to every plan in this context.
  void set_work_area(void* area)
  {
    work_area = area;
    for (auto p : plans_forward) CUFFT_CALL(cufftSetWorkArea(p, area));
    for (auto p : plans_backward) CUFFT_CALL(cufftSetWorkArea(p, area));
  }

  /// Bind a CUDA stream to every plan in this context.
  void set_stream(cudaStream_t stream) const
  {
    for (auto p : plans_forward) CUFFT_CALL(cufftSetStream(p, stream));
    for (auto p : plans_backward) CUFFT_CALL(cufftSetStream(p, stream));
  }
};

// Pads and ifftshifts a 2D image in one pass.
// Input:  (nx, ny) real, origin at center
// Output: (px, py) real, origin at (0,0), zero-padded
// Output is zeroed before launch.
void pad_ifftshift(float* input, float* output, int nx, int ny, int px, int py, int npad_x, int npad_y,
                   cudaStream_t stream);

// Pads and ifftshifts a batched 2D image in one pass.
// Input:  (n_batch, nx, ny) real, origin at center
// Output: (n_batch, px, py) real, origin at (0,0), zero-padded
// Output is zeroed before launch.
void pad_ifftshift_batched(float* input, float* output, int nx, int ny, int px, int py, int npad_x, int npad_y,
                           int n_batch, cudaStream_t stream);

// Fftshifts and crops a batched 2D image in one pass.
// Input:  (n_batch, px, py) real, origin at (0,0) (FFT output)
// Output: (n_batch, nx, ny) real, origin at center, cropped
void fftshift_crop(float* input, float* output, int nx, int ny, int px, int py, int npad_x, int npad_y, int n_batch,
                   cudaStream_t stream);

// Compute padding amounts (rows, cols) for a target padding factor.
std::pair<int, int> compute_padding(int npix_x, int npix_y, float padding);

// Smallest m >= n whose prime factors are all in {2, 3, 5, 7}. cuFFT runs
// Cooley-Tukey on these sizes with minimal workspace; non-smooth sizes (e.g.
// large prime factors) trigger Bluestein, which can blow up workspace by 10x+.
int next_fast_size(int n);

}  // namespace fast_deconv::linalg
