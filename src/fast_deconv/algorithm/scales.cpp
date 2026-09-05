#include <emu/submdspan.hpp>
#include <fast_deconv/algorithm/scales.hpp>

namespace fast_deconv::scale {

void convolve_psfs_with_scale_async(const linalg::convolve_ctx& conv, core::span4d<float> psfs,
                                    core::span1d<float> d_sigma, int scale_idx, core::span1d<const float> weights,
                                    core::span4d<float> out_conv_psf, core::span3d<float> out_conv2_mean)
{
  const int n_facets = psfs.extent(0);
  psf_convolve_scratch scratch(conv.ctx(), conv.dims(), psfs.extent(1));

  for (int f = 0; f < n_facets; f++)
    convolve_psf_with_scale_async(conv, emu::submdspan(psfs, f), d_sigma, scale_idx, weights, scratch,
                                  emu::submdspan(out_conv_psf, f), emu::submdspan(out_conv2_mean, f));
}

void convolve_psfs_with_scales_async(const linalg::convolve_ctx& conv, core::span4d<float> psfs,
                                     core::span1d<float> d_sigmas, core::span1d<const float> weights,
                                     core::span5d<float> out_conv_psf, core::span4d<float> out_conv2_mean)
{
  const int n_scales = d_sigmas.size();
  for (int i = 0; i < n_scales; i++) {
    core::span1d<float> sigma_view(d_sigmas.data_handle() + i, 1);
    convolve_psfs_with_scale_async(conv, psfs, sigma_view, i, weights, emu::submdspan(out_conv_psf, i),
                                   emu::submdspan(out_conv2_mean, i));
  }
}

}  // namespace fast_deconv::scale
