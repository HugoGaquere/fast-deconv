#pragma once
#include <cufft.h>

#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>
#include <fd_backend/linalg/fft.hpp>

namespace fast_deconv::linalg {

/**
 * Linear (zero outside) convolution of a batch of same-sized images with the band-limited Gaussian
 * H(f) = exp(-2 pi^2 sigma^2 f^2). A caller that blurs one input with several sigmas runs forward()
 * once and convolve() per sigma. On this backend the convolution is a padded 2D FFT: the cuFFT plans
 * and the scratch are built once and run on the lane given at construction, which must outlive this.
 */
class gaussian_convolution_ctx {
 public:
  /// On this backend, the half-complex spectrum of the padded input: batch * freq_total.
  using spectrum = core::cont1d<complex_type>;

  /// @p gap: zero padding in pixels per axis, P = next_fast_size(n + gap); wrap-free for every sigma with
  /// gaussian_reach(sigma) <= gap.
  gaussian_convolution_ctx(const core::exec_ctx& ctx, int batch, int nrow, int ncol, int gap);
  ~gaussian_convolution_ctx();

  gaussian_convolution_ctx(const gaussian_convolution_ctx&) = delete;
  gaussian_convolution_ctx& operator=(const gaussian_convolution_ctx&) = delete;
  gaussian_convolution_ctx(gaussian_convolution_ctx&&) = delete;
  gaussian_convolution_ctx& operator=(gaussian_convolution_ctx&&) = delete;

  /// A spectrum sized for this context, for forward().
  spectrum make_spectrum() const;

  /// Pad (input at the top-left) + R2C of @p input, (batch, nrow, ncol), into @p out. @p input can change once this
  /// returns.
  void forward(core::span3d<const float> input, spectrum& out) const;

  /// Multiply @p in by Gaussian(@p sigma), C2R, crop the top-left into @p out. @p in is left intact.
  void convolve(const spectrum& in, float sigma, core::span3d<float> out) const;

  /// forward() then convolve(), through a temporary spectrum, for single-sigma uses.
  void convolve(core::span3d<const float> input, float sigma, core::span3d<float> out) const;

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
