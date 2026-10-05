#include <cassert>
#include <cstddef>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/convolution.hpp>

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

}  // namespace fast_deconv::linalg
