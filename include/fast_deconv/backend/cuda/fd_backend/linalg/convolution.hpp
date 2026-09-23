#pragma once
#include <cufft.h>

#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>
#include <fd_backend/linalg/fft.hpp>

namespace fast_deconv::linalg {

/**
 * Gaussian convolution of a batch of same-sized images through a padded FFT.
 * The cuFFT plans and the scratch are built once and run on the lane given at
 * construction, which must outlive this. No input is kept between calls: a
 * caller that convolves one input with several sigmas keeps the spectrum from
 * forward() and passes it to convolve_spectrum() for each sigma.
 */
class convolution_ctx {
 public:
  convolution_ctx(const core::exec_ctx& ctx, int nrow, int ncol, float padding, int batch);
  ~convolution_ctx();

  convolution_ctx(const convolution_ctx&) = delete;
  convolution_ctx& operator=(const convolution_ctx&) = delete;
  convolution_ctx(convolution_ctx&&) = delete;
  convolution_ctx& operator=(convolution_ctx&&) = delete;

  /// Pad + ifftshift + R2C of @p input, (batch, nrow, ncol), into @p spectrum, batch * dims().freq_total().
  void forward(core::span3d<const float> input, core::span1d<complex_type> spectrum) const;

  /// Multiply @p spectrum by Gaussian(@p sigma), C2R, fftshift + crop into @p out. @p spectrum is left intact.
  void convolve_spectrum(core::span1d<const complex_type> spectrum, float sigma, core::span3d<float> out) const;

  /// forward() then convolve_spectrum(), through a temporary spectrum. Needs batch() == 1.
  void convolve_with_gaussian(core::span2d<const float> input, float sigma, core::span2d<float> out) const;

  /// @p input convolved once (@p out_conv) and twice (@p out_conv2) with Gaussian(@p sigma).
  void convolve_with_gaussian_once_and_twice(core::span3d<const float> input, float sigma, core::span3d<float> out_conv,
                                             core::span3d<float> out_conv2) const;

  const fft_dims& dims() const { return dims_; }
  int batch() const { return batch_; }

 private:
  void forward_(float* input, complex_type* output) const;   // R2C over the batch
  void backward_(complex_type* input, float* output) const;  // unnormalized C2R over the batch

  const core::exec_ctx& ctx_;
  fft_dims dims_;
  int batch_;
  core::owned_ptr<float> padded_;          // batch * padded_total: forward input, then inverse output
  core::owned_ptr<complex_type> product_;  // batch * freq_total: spectrum times the Gaussian
  cufftHandle r2c_ = 0;
  cufftHandle c2r_ = 0;
};

}  // namespace fast_deconv::linalg
