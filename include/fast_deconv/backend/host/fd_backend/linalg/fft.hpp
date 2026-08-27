#pragma once
#include <complex>
#include <cstddef>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>

namespace fast_deconv::linalg {

using complex_type = std::complex<float>;

/// Same shape as the cuFFT context so call sites are identical. A host FFT
/// library allocates its own scratch, so the work-area API is inert here.
class convolve_ctx {
 public:
  convolve_ctx(const core::exec_ctx& ctx, int input_nrow, int input_ncol, int forward_batch, int backward_batch,
               int /*n_backward_plans*/, float padding)
      : ctx_(ctx),
        dims_(input_nrow, input_ncol, padding),
        forward_batch_(forward_batch),
        backward_batch_(backward_batch)
  {
  }

  convolve_ctx(const convolve_ctx&) = delete;
  convolve_ctx& operator=(const convolve_ctx&) = delete;
  convolve_ctx(convolve_ctx&&) = delete;
  convolve_ctx& operator=(convolve_ctx&&) = delete;

  void forward_async(float* input, complex_type* output) const;
  void backward_async(complex_type* input, float* output, int plan_idx = 0) const;

  void bind_work_area(void* /*work_area*/) {}
  std::size_t required_work_size() const { return 0; }

  const fft_dims& dims() const { return dims_; }
  int forward_batch() const { return forward_batch_; }
  int backward_batch() const { return backward_batch_; }
  const core::exec_ctx& ctx() const { return ctx_; }

 private:
  const core::exec_ctx& ctx_;
  fft_dims dims_;
  int forward_batch_ = 0;
  int backward_batch_ = 0;
};

}  // namespace fast_deconv::linalg
