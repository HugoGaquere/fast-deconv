#include "fast_deconv/core/span_types.hpp"

#include <fast_deconv/algorithm/wscms.hpp>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/mdarray.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

int run_wscms_example()
{
  uint n_facets     = 1;
  uint n_scales     = 2;
  uint n_pol        = 1;
  uint n_freq       = 1;
  uint dirty_width  = 10000;
  uint dirty_height = 10000;
  uint psf_width    = 9000;
  uint psf_height   = 9000;

  auto dirty =
    fast_deconv::core::make_managed_mdarray<float>(n_freq, n_pol, dirty_width, dirty_height);
  auto scaled_dirty =
    fast_deconv::core::make_managed_mdarray<float>(n_freq, n_pol, dirty_width, dirty_height);
  auto psfs = fast_deconv::core::make_managed_mdarray<float>(
    n_facets, n_scales, n_freq, n_pol, psf_width, psf_height);
  auto jones_norm =
    fast_deconv::core::make_managed_mdarray<float>(n_freq, n_pol, dirty_width, dirty_height);
  auto mask  = fast_deconv::core::make_managed_mdarray<bool>(dirty_width, dirty_height);
  auto gains = fast_deconv::core::make_managed_mdarray<float>(n_facets, n_scales);

  auto fill_with_random = [](auto arr) -> auto {
    for (int i = 0; i < arr.size(); i++) {
      arr.data_handle()[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
  };

  auto fill_mask = [](auto arr) -> auto {
    for (int i = 0; i < arr.size(); i++) {
      arr.data_handle()[i] = true;
    }
  };

  fill_with_random(dirty.view());
  fill_with_random(scaled_dirty.view());
  fill_with_random(psfs.view());
  fill_with_random(gains.view());
  fill_mask(mask.view());

  fast_deconv::algorithm::wscms::MinorCycleContext ctx{
    .psfs       = psfs.view(),
    .jones_norm = jones_norm.view(),
    .gains      = gains.view(),
    .mask       = mask.view(),
  };

  fast_deconv::core::stream_resources resources;
  uint scale_idx = 1;

  std::printf("Running WSCMS minor cycle\n");

  // fast_deconv::algorithm::wscms::wscms_minor_cycle(dirty.view(),
  //                                                  scaled_dirty.view(),
  //                                                  scale_idx,
  //                                                  ctx,
  //                                                  resources);

  std::printf("\n");

  return 0;
}
