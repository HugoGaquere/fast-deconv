#include <fast_deconv/linalg/linalg.hpp>
#include <stdexcept>

namespace fast_deconv::linalg {

namespace {
[[noreturn]] void not_implemented()
{
  throw std::runtime_error("linalg::weighted_sum_async: host backend not implemented yet");
}
}  // namespace

void weighted_sum_async(const core::exec_ctx& /*ctx*/, const float* /*A*/, const float* /*weights*/, float* /*out*/,
                        int /*w*/, int /*n*/)
{
  not_implemented();
}

void weighted_sum_async(const core::exec_ctx& /*ctx*/, const core::span3d<float> /*A*/,
                        const core::span1d<float> /*weights*/, core::span2d<float> /*out*/)
{
  not_implemented();
}

}  // namespace fast_deconv::linalg
