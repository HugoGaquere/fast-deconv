#include <fast_deconv/algorithm/scales.hpp>
#include <stdexcept>
#include <string>

namespace fast_deconv::scale {

namespace {
[[noreturn]] void not_implemented(const char* what)
{
  throw std::runtime_error(std::string(what) + ": host backend not implemented yet");
}
}  // namespace

void make_gaussian_kernels_async(const core::exec_ctx& /*ctx*/, core::span1d<float> /*sigmas*/,
                                 int /*scale_ncol_full*/, core::span3d<float> /*scales*/)
{
  not_implemented("scale::make_gaussian_kernels_async");
}

void convolve_with_scales(const linalg::convolve_ctx& /*conv*/, core::span2d<float> /*dirty*/,
                          core::span3d<float> /*scales*/, core::span3d<float> /*out_scaled_dirty*/)
{
  not_implemented("scale::convolve_with_scales");
}

int scale_selection(const core::exec_ctx& /*ctx*/, core::span3d<float> /*scaled_dirty*/,
                    core::host_vect<float> /*bias*/, const std::vector<int>& /*retired_scales*/)
{
  not_implemented("scale::scale_selection");
}

void convolve_psfs_with_scale_async(const linalg::convolve_ctx& /*conv*/, core::span4d<float> /*psfs*/,
                                    core::span1d<float> /*d_sigma*/, int /*scale_idx*/,
                                    core::span1d<float> /*weights*/, core::span4d<float> /*out_conv_psf*/,
                                    core::span3d<float> /*out_conv2_mean*/)
{
  not_implemented("scale::convolve_psfs_with_scale_async");
}

void convolve_psfs_with_scales_async(const linalg::convolve_ctx& /*conv*/, core::span4d<float> /*psfs*/,
                                     core::span1d<float> /*d_sigmas*/, core::span1d<float> /*weights*/,
                                     core::span5d<float> /*out_conv_psf*/, core::span4d<float> /*out_conv2_mean*/)
{
  not_implemented("scale::convolve_psfs_with_scales_async");
}

}  // namespace fast_deconv::scale
