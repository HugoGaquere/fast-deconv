#include "fast_deconv/core/span_types.hpp"

#include <fast_deconv/algorithm/wscms.hpp>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/mdarray.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/argmax.cuh>

#include <chrono>

template <typename T>
void print_mdspan(const fast_deconv::core::span_2d<T>& M)
{
  using cuda::std::layout_left;
  using cuda::std::layout_right;

  const size_t rows = M.extent(0);
  const size_t cols = M.extent(1);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      printf("%8.3f ", static_cast<double>(M(i, j)));
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char** argv)
{
  std::printf("Hello gpu world\n");

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

  auto matrix = fast_deconv::core::make_managed_mdarray<float>(5, 5);
  fill_with_random(matrix.view());
  print_mdspan<float>(matrix.view());

  fast_deconv::algorithm::wscms::Params params{
    .peak_factor      = 0.2,
    .max_iter         = 1000,
    .cell_size_radian = {0.2, 0.2},
  };

  fast_deconv::algorithm::wscms::Facets facets{
    .edges   = {{0.2, 0.5, 0.6}},
    .centers = {{0.2, 0.2}},
  };

  fast_deconv::core::stream_resources resources;
  uint scale_idx = 1;

  std::printf("Running WSCMS minor cycle\n");
  auto start = std::chrono::high_resolution_clock::now();

  fast_deconv::algorithm::wscms::wscms_minor_cycle(dirty.view(),
                                                   scaled_dirty.view(),
                                                   mask.view(),
                                                   psfs.view(),
                                                   jones_norm.view(),
                                                   gains.view(),
                                                   scale_idx,
                                                   facets,
                                                   params,
                                                   resources);

  auto stop     = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  printf("Latency: %ld microsec\n", duration.count());

  std::printf("\n");

  return 0;
}
