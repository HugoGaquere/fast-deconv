#include <cassert>
#include <cstddef>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/convolution.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/linalg.hpp>

namespace fast_deconv::linalg {

void convolution_ctx::forward(core::span3d<const float> input, core::span1d<complex_type> spectrum) const
{
  FD_PROFILE_FN();
  assert(input.is_exhaustive() && input.extent(0) == batch_);
  assert(input.extent(1) == dims_.input_nrow && input.extent(2) == dims_.input_ncol);
  assert(spectrum.size() == static_cast<std::size_t>(batch_) * dims_.freq_total());

  pad_ifftshift_batched_async(ctx_, dims_, input.data_handle(), padded_.get(), batch_);
  forward_(padded_.get(), spectrum.data_handle());
}

void convolution_ctx::convolve_spectrum(core::span1d<const complex_type> spectrum, float sigma,
                                        core::span3d<float> out) const
{
  FD_PROFILE_FN();
  assert(spectrum.size() == static_cast<std::size_t>(batch_) * dims_.freq_total());
  assert(out.is_exhaustive() && out.extent(0) == batch_);
  assert(out.extent(1) == dims_.input_nrow && out.extent(2) == dims_.input_ncol);

  // The multiply writes product_, so the caller's spectrum survives for the next sigma.
  multiply_with_gaussian(ctx_, dims_, batch_, spectrum.data_handle(), product_.get(), sigma);
  backward_(product_.get(), padded_.get());
  fftshift_crop_async(ctx_, dims_, padded_.get(), out.data_handle(), batch_);
}

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

  const std::size_t freq_count = static_cast<std::size_t>(batch_) * dims_.freq_total();
  auto spectrum = ctx_.alloc_mdcontainer_async<complex_type>(freq_count);
  auto product2 = ctx_.alloc_ptr_async<complex_type>(freq_count);
  forward(input, spectrum);

  // One pass writes both products; each inverse then consumes its own.
  multiply_with_gaussian_once_and_twice(ctx_, dims_, batch_, spectrum.data_handle(), product_.get(), product2.get(),
                                        sigma);
  backward_(product_.get(), padded_.get());
  fftshift_crop_async(ctx_, dims_, padded_.get(), out_conv.data_handle(), batch_);
  backward_(product2.get(), padded_.get());
  fftshift_crop_async(ctx_, dims_, padded_.get(), out_conv2.data_handle(), batch_);
}

}  // namespace fast_deconv::linalg
