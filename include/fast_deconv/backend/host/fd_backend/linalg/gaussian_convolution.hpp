#pragma once
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>

namespace fast_deconv::linalg {

/**
 * Linear (zero outside) convolution of a batch of same-sized images with the band-limited Gaussian
 * H(f) = exp(-2 pi^2 sigma^2 f^2). A caller that blurs one input with several sigmas runs forward()
 * once and convolve() per sigma. On this backend the convolution is separable, rows then columns,
 * through per-thread tiles: no 2D FFT. The lane given at construction must outlive this.
 */
class gaussian_convolution_ctx {
 public:
  /// On this backend, a copy of the input: (batch, nrow, ncol).
  using spectrum = core::cont3d<float>;

  /// @p gap: zero padding in pixels per axis; wrap-free for every sigma with gaussian_reach(sigma) <= gap.
  /// Each line is padded to min(gap, gaussian_reach(sigma)), so small sigmas keep short lines.
  gaussian_convolution_ctx(const core::exec_ctx& ctx, int batch, int nrow, int ncol, int gap);
  ~gaussian_convolution_ctx() = default;

  gaussian_convolution_ctx(const gaussian_convolution_ctx&) = delete;
  gaussian_convolution_ctx& operator=(const gaussian_convolution_ctx&) = delete;
  gaussian_convolution_ctx(gaussian_convolution_ctx&&) = delete;
  gaussian_convolution_ctx& operator=(gaussian_convolution_ctx&&) = delete;

  /// A spectrum sized for this context, for forward().
  spectrum make_spectrum() const;

  /// Transforms @p input, (batch, nrow, ncol), into @p out. @p input can change once this returns.
  void forward(core::span3d<const float> input, spectrum& out) const;

  /// Gaussian(@p sigma) convolution of forward()'s output into @p out. @p in is left intact.
  void convolve(const spectrum& in, float sigma, core::span3d<float> out) const;

  /// Gaussian(@p sigma) convolution of @p input into @p out, for single-sigma uses.
  void convolve(core::span3d<const float> input, float sigma, core::span3d<float> out) const;

  int batch() const { return batch_; }

 private:
  // The tiled convolution of (batch, nrow, ncol) floats at @p in into @p out.
  void convolve_tiled_(const float* in, float sigma, core::span3d<float> out) const;

  const core::exec_ctx& ctx_;
  int batch_;
  int nrow_;
  int ncol_;
  int gap_;
};

}  // namespace fast_deconv::linalg
