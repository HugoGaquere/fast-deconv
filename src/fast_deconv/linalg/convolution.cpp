#include <cassert>
#include <cstddef>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/convolution.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <numbers>

namespace fast_deconv::linalg {

void convolution_ctx::convolve_with_gaussian(core::span2d<const float> input, float sigma,
                                             core::span2d<float> out) const
{
  FD_PROFILE_FN();
  assert(batch_ == 1);
  auto spectrum = ctx_.alloc_mdcontainer_async<complex_type>(dims_.freq_total());
  forward(core::span3d<const float>(input.data_handle(), 1, input.extent(0), input.extent(1)), spectrum);
  convolve_spectrum(spectrum, sigma, core::span3d<float>(out.data_handle(), 1, out.extent(0), out.extent(1)));
}

void convolution_ctx::convolve_with_gaussian_once_and_twice(core::span3d<const float> input, float sigma,
                                                            core::span3d<float> out_conv,
                                                            core::span3d<float> out_conv2) const
{
  FD_PROFILE_FN();
  assert(out_conv.is_exhaustive() && out_conv.extent(0) == batch_);
  assert(out_conv2.is_exhaustive() && out_conv2.extent(0) == batch_);

  // Previous version: one R2C, both products in one multiply pass, two inverses (FFT route on both backends).
  // const std::size_t freq_count = static_cast<std::size_t>(batch_) * dims_.freq_total();
  // auto spectrum = ctx_.alloc_mdcontainer_async<complex_type>(freq_count);
  // auto product2 = ctx_.alloc_ptr_async<complex_type>(freq_count);
  // // Its own R2C: on the host backend forward() keeps the padded input instead of a spectrum.
  // pad_ifftshift_batched_async(ctx_, dims_, input.data_handle(), padded_.get(), batch_);
  // forward_(padded_.get(), spectrum.data_handle());
  //
  // // One pass writes both products; each inverse then consumes its own.
  // multiply_with_gaussian_once_and_twice(ctx_, dims_, batch_, spectrum.data_handle(), product_.get(), product2.get(),
  //                                       sigma);
  // backward_(product_.get(), padded_.get());
  // fftshift_crop_async(ctx_, dims_, padded_.get(), out_conv.data_handle(), batch_);
  // backward_(product2.get(), padded_.get());
  // fftshift_crop_async(ctx_, dims_, padded_.get(), out_conv2.data_handle(), batch_);

  // G(sigma) twice is G(sigma * sqrt(2)): H(sigma)^2 = H(sigma * sqrt(2)). On the host, forward() and
  // convolve_spectrum() are the tiled convolution; on CUDA, one R2C and one multiply + C2R per sigma.
  auto spectrum = ctx_.alloc_mdcontainer_async<complex_type>(static_cast<std::size_t>(batch_) * dims_.freq_total());
  forward(input, spectrum);
  convolve_spectrum(spectrum, sigma, out_conv);
  convolve_spectrum(spectrum, sigma * std::numbers::sqrt2_v<float>, out_conv2);
}

}  // namespace fast_deconv::linalg
