#pragma once
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>
#include <fd_backend/linalg/fft.hpp>

namespace fast_deconv::linalg {

/**
 * Gaussian convolution of a batch of same-sized images, separable through
 * per-thread tiles (no 2D FFT). It keeps the FFT backend's interface: a caller
 * that convolves one input with several sigmas keeps the "spectrum" from
 * forward() and passes it to convolve_spectrum() for each sigma. The lane
 * given at construction must outlive this.
 */
class convolution_ctx {
 public:
  convolution_ctx(const core::exec_ctx& ctx, int nrow, int ncol, float padding, int batch);
  ~convolution_ctx() = default;

  convolution_ctx(const convolution_ctx&) = delete;
  convolution_ctx& operator=(const convolution_ctx&) = delete;
  convolution_ctx(convolution_ctx&&) = delete;
  convolution_ctx& operator=(convolution_ctx&&) = delete;

  /// Copies @p input, (batch, nrow, ncol), into @p spectrum, batch * dims().freq_total(): on this
  /// backend the buffer holds the input as floats, not a spectrum.
  void forward(core::span3d<const float> input, core::span1d<complex_type> spectrum) const;

  /// Separable Gaussian(@p sigma) convolution of forward()'s output into @p out: rows then columns, each
  /// line zero-padded in a per-thread tile and convolved by ducc0::convolve_axis. The padding is per sigma,
  /// from the kernel's reach, not dims()'s padding factor. @p spectrum is left intact.
  void convolve_spectrum(core::span1d<const complex_type> spectrum, float sigma, core::span3d<float> out) const;

  /// forward() then convolve_spectrum(), through a temporary spectrum. Needs batch() == 1.
  void convolve_with_gaussian(core::span2d<const float> input, float sigma, core::span2d<float> out) const;

  const fft_dims& dims() const { return dims_; }
  int batch() const { return batch_; }

 private:
  const core::exec_ctx& ctx_;
  fft_dims dims_;
  int batch_;
};

}  // namespace fast_deconv::linalg
