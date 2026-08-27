#pragma once
#include <cufft.h>

#include <cstddef>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>
#include <vector>

namespace fast_deconv::linalg {

using complex_type = cufftComplex;

/**
 * @brief cuFFT R2C/C2R convolution plans over a padded grid.
 *
 * Builds one R2C plan and @p n_backward_plans C2R plans with cuFFT
 * auto-allocation disabled, all bound to the lane's stream. The constructor
 * allocates nothing: the caller queries required_work_size() and provides a
 * workspace via bind_work_area() before any transform. The buffer may be
 * shared with other contexts on the same stream, since executions are then
 * serialized.
 *
 * Transforms are unnormalized, as cuFFT leaves them — callers fold
 * 1 / dims().padded_total() into their frequency-domain multiply.
 *
 * The exec_ctx and the work-area buffer must outlive the context.
 */
class convolve_ctx {
 public:
  /**
   * @brief Create the plans for a padded 2D grid.
   *
   * Forward and backward plans can have different batch sizes, e.g. forward=1
   * to FFT a single image once, backward=N to IFFT N filtered spectra in a
   * single batched call.
   *
   * @param ctx               Execution lane the plans run on. Must outlive this ctx.
   * @param input_nrow        Unpadded input rows.
   * @param input_ncol        Unpadded input cols.
   * @param forward_batch     Batch size for the R2C plan.
   * @param backward_batch    Batch size shared by all C2R plans.
   * @param n_backward_plans  Number of C2R plans (e.g. 1 for conv, 2 for conv + conv^2).
   * @param padding           FFT padding factor (e.g. 1.5).
   */
  convolve_ctx(const core::exec_ctx& ctx, int input_nrow, int input_ncol, int forward_batch, int backward_batch,
               int n_backward_plans, float padding);

  convolve_ctx(const convolve_ctx&) = delete;
  convolve_ctx& operator=(const convolve_ctx&) = delete;
  convolve_ctx(convolve_ctx&&) = delete;
  convolve_ctx& operator=(convolve_ctx&&) = delete;

  ~convolve_ctx();

  /// Forward R2C over forward_batch() padded slices.
  void forward_async(float* input, complex_type* output) const;

  /// Backward C2R over backward_batch() spectra, through plan @p plan_idx.
  void backward_async(complex_type* input, float* output, int plan_idx = 0) const;

  /// Bind a caller-owned workspace (>= required_work_size() bytes) to every
  /// plan. The context does not take ownership; the buffer must stay alive
  /// until the last transform.
  void bind_work_area(void* work_area);

  /// Required size in bytes of the shared workspace buffer.
  std::size_t required_work_size() const { return work_size_; }

  const fft_dims& dims() const { return dims_; }
  int forward_batch() const { return forward_batch_; }
  int backward_batch() const { return backward_batch_; }
  const core::exec_ctx& ctx() const { return ctx_; }

 private:
  const core::exec_ctx& ctx_;
  fft_dims dims_;
  int forward_batch_ = 0;
  int backward_batch_ = 0;
  std::vector<cufftHandle> plans_forward_;
  std::vector<cufftHandle> plans_backward_;
  std::size_t work_size_ = 0;
};

}  // namespace fast_deconv::linalg
