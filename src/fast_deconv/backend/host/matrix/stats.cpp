#include <fast_deconv/matrix/stats.hpp>
#include <stdexcept>

namespace fast_deconv::matrix {

stats_ctx::stats_ctx(const core::exec_ctx& ctx, std::size_t n_elements, bool use_abs)
    : ctx_(ctx), n_elements_(n_elements), use_abs_(use_abs)
{
}

void stats_ctx::run_async(core::span2d<float> /*data*/, core::span2d<bool> /*mask*/)
{
  throw std::runtime_error("matrix::stats: host backend not implemented yet");
}

stats_result stats_ctx::run(core::span2d<float> /*data*/, core::span2d<bool> /*mask*/)
{
  throw std::runtime_error("matrix::stats: host backend not implemented yet");
}

}  // namespace fast_deconv::matrix
