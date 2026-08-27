#include <fast_deconv/common/mask.hpp>
#include <stdexcept>

namespace fast_deconv::common {

namespace {
[[noreturn]] void not_implemented() { throw std::runtime_error("common::mask: host backend not implemented yet"); }
}  // namespace

void mask_and_abs_async(const core::exec_ctx& /*ctx*/, core::span2d<float> /*data*/, core::span2d<bool> /*mask*/,
                        float /*fill_value*/, bool /*abs*/)
{
  not_implemented();
}

void mask_and_abs_async(const core::exec_ctx& /*ctx*/, core::span3d<float> /*data*/, core::span2d<bool> /*mask*/,
                        float /*fill_value*/, bool /*abs*/)
{
  not_implemented();
}

void mask_and_abs_async(const core::exec_ctx& /*ctx*/, core::span3d<float> /*data*/, core::span3d<bool> /*mask*/,
                        float /*fill_value*/, bool /*abs*/)
{
  not_implemented();
}

void mask_less_than_threshold(const core::exec_ctx& /*ctx*/, core::span2d<float> /*data*/, float /*threshold*/,
                              float /*fill_value*/)
{
  not_implemented();
}

void build_auto_mask(const core::exec_ctx& /*stream*/, const std::vector<std::pair<int, int>>& /*coords*/,
                     const std::vector<int>& /*scales*/, core::span3d<float> /*central_facet_psfs*/,
                     core::span1d<float> /*weights_freq*/, core::span1d<float> /*scale_sigmas*/, float /*fft_padding*/,
                     core::span2d<bool> /*external_mask*/, core::span3d<bool> /*mask_per_scale*/)
{
  not_implemented();
}

}  // namespace fast_deconv::common
