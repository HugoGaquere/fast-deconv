#pragma once
#include <cufft.h>

#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>
#include <fd_backend/linalg/fft.hpp>
#include <vector>

namespace fast_deconv::linalg {

/**
 * @brief Linear (zero outside) convolution of a batch of same-sized images with the band-limited Gaussian
 *        H(f) = exp(-2 pi^2 sigma^2 f^2).
 *
 * A caller that blurs one input with several sigmas runs forward() once and convolve() per sigma. On this
 * backend the convolution is a padded 2D FFT: the cuFFT plans and the scratch are built once and run on
 * the context given at construction, which must outlive this.
 */
class gaussian_convolution_ctx {
 public:
  /// On this backend, the half-complex spectrum of the padded input: batch * freq_total.
  using spectrum = core::cont1d<complex_type>;

  /**
   * @brief Builds the cuFFT plans and the scratch for one image shape.
   *
   * @param[in] ctx    Context the plans and scratch live on. Must outlive this.
   * @param[in] batch  Images per call.
   * @param[in] nrow   Image rows.
   * @param[in] ncol   Image columns.
   * @param[in] sigmas Gaussian sigmas convolve() will be called with, in pixels.
   * @param[in] factor Multiplier applied to every sigma.
   *
   * Pads to P = next_fast_size(n + max_gaussian_reach(sigmas, factor)) per axis: wrap-free for those sigmas.
   */
  gaussian_convolution_ctx(const core::exec_ctx& ctx, std::int64_t batch, std::int64_t nrow, std::int64_t ncol,
                           const std::vector<float>& sigmas, float factor = 1.0f);
  ~gaussian_convolution_ctx();

  gaussian_convolution_ctx(const gaussian_convolution_ctx&) = delete;
  gaussian_convolution_ctx& operator=(const gaussian_convolution_ctx&) = delete;
  gaussian_convolution_ctx(gaussian_convolution_ctx&&) = delete;
  gaussian_convolution_ctx& operator=(gaussian_convolution_ctx&&) = delete;

  /// A spectrum sized for this context, for forward().
  spectrum make_spectrum() const;

  /**
   * @brief Pad (input at the top-left) + R2C of @p input into @p out.
   *
   * @param[in]  input  Images, (batch, nrow, ncol); can change once this returns.
   * @param[out] out    Spectrum from make_spectrum().
   */
  void forward(core::span3d<const float> input, spectrum& out) const;

  /// Multiply @p in by Gaussian(@p sigma), C2R, crop the top-left into @p out. @p in is left intact.
  void convolve(const spectrum& in, float sigma, core::span3d<float> out) const;

  /// Pad, R2C, multiply in place, C2R, crop: no spectrum kept, for single-sigma uses.
  void convolve(core::span3d<const float> input, float sigma, core::span3d<float> out) const;

  std::int64_t batch() const { return batch_; }

 private:
  void forward_(float* input, complex_type* output) const;   // R2C over the batch
  void backward_(complex_type* input, float* output) const;  // unnormalized C2R over the batch

  const core::exec_ctx& ctx_;
  fft_dims dims_;
  std::int64_t batch_;
  core::owned_ptr<float> padded_;          // batch * padded_total: forward input, then inverse output
  core::owned_ptr<complex_type> product_;  // batch * freq_total: spectrum times the Gaussian
  cufftHandle r2c_ = 0;
  cufftHandle c2r_ = 0;
};

}  // namespace fast_deconv::linalg
