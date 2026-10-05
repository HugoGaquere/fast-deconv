#include <algorithm>
#include <emu/submdspan.hpp>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/gaussian_convolution.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <stdexcept>

#include "fast_deconv/matrix/peak.hpp"

namespace fast_deconv::scale {

scale_result select_best_scale(const core::exec_ctx& exec_ctx, const linalg::gaussian_convolution_ctx& conv_ctx,
                               core::span2d<const float> dirty, const std::vector<float>& sigmas,
                               core::host_span1d<float> bias, const std::vector<int>& retired, core::span3d<bool> mask,
                               bool absolute)
{
  FD_PROFILE_FN();

  auto current = exec_ctx.alloc_mdcontainer_async<float>(dirty.extent(0), dirty.extent(1));
  auto best = exec_ctx.alloc_mdcontainer_async<float>(dirty.extent(0), dirty.extent(1));

  // No mask means no masking; a single-plane mask is shared by every scale.
  const auto criterion_for = [&](int s) {
    const bool* scale_mask = mask.data_handle() == nullptr ? nullptr
                             : mask.extent(0) == 1         ? mask.data_handle()
                                                           : emu::submdspan(mask, s).data_handle();
    return matrix::peak_criterion{.mask = scale_mask, .absolute = absolute};
  };
  const auto is_retired = [&](int s) { return std::ranges::find(retired, s) != retired.end(); };

  int best_scale = -1;
  matrix::peak best_peak{};
  float best_biased = -std::numeric_limits<float>::infinity();
  // Strictly greater, so the lowest scale wins a tie and a NaN peak loses.
  const auto is_better = [&](int s, matrix::peak p) {
    const float biased = p.value * bias(s);
    if (!(biased > best_biased)) return false;
    best_biased = biased;
    best_scale = s;
    best_peak = p;
    return true;
  };

  // Scale 0 is the identity: search dirty in place, copy only if it wins.
  if (!is_retired(0)) is_better(0, matrix::find_peak(exec_ctx, dirty, criterion_for(0)));

  // One forward transform, shared by every scale's convolution.
  auto spectrum = conv_ctx.make_spectrum();
  if (sigmas.size() > 1)
    conv_ctx.forward(core::span3d<const float>(dirty.data_handle(), 1, dirty.extent(0), dirty.extent(1)), spectrum);

  for (int s = 1; s < static_cast<int>(sigmas.size()); s++) {
    if (is_retired(s)) continue;
    // Rebuilt each time: the swap below moves current onto the other buffer.
    conv_ctx.convolve(spectrum, sigmas.at(s),
                      core::span3d<float>(current.data_handle(), 1, current.extent(0), current.extent(1)));
    if (is_better(s, matrix::find_peak(exec_ctx, current, criterion_for(s)))) std::swap(current, best);
  }

  if (best_scale < 0) throw std::invalid_argument("select_best_scale: every scale is retired");

  if (best_scale == 0) exec_ctx.copy(best, dirty);

  return {.scale = best_scale,
          .scaled_residual = std::move(best),
          .peak = best_peak,
          .criterion = criterion_for(best_scale)};
}

}  // namespace fast_deconv::scale
