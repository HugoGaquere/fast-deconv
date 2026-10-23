#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fast_deconv/common/detail/mask_steps.hpp>

namespace fast_deconv::common::detail {

void stamp_peaks(const core::exec_ctx& /*ctx*/, const std::vector<index2d>& coords, const std::vector<int>& scales,
                 core::span3d<bool> mask_per_scale)
{
  std::fill_n(mask_per_scale.data_handle(), mask_per_scale.size(), false);
  for (std::size_t i = 0; i < coords.size(); i++)
    mask_per_scale(scales.at(i), coords.at(i).row, coords.at(i).col) = true;
}

void threshold_above(const core::exec_ctx& /*ctx*/, core::span2d<const float> in, float threshold,
                     core::span2d<bool> out)
{
  const float* src = in.data_handle();
  bool* dst = out.data_handle();
  const std::int64_t n = in.mapping().required_span_size();
  for (std::int64_t k = 0; k < n; k++) dst[k] = src[k] > threshold;
}

void finalize_mask(const core::exec_ctx& /*ctx*/, core::span2d<const bool> dilated, core::span2d<const bool> external,
                   core::span2d<bool> slice)
{
  const bool* d = dilated.data_handle();
  const bool* e = external.data_handle();
  bool* s = slice.data_handle();
  const std::int64_t n = slice.mapping().required_span_size();
  for (std::int64_t k = 0; k < n; k++) s[k] = !d[k] || e[k];
}

}  // namespace fast_deconv::common::detail
