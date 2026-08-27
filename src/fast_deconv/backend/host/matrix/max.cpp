#include <fast_deconv/matrix/max.hpp>
#include <stdexcept>

namespace fast_deconv::matrix {

float max(const core::exec_ctx& ctx, core::span2d<float> data, core::span2d<bool> mask, bool use_abs)
{
  throw std::runtime_error("backend not implemented yet");
}

}  // namespace fast_deconv::matrix
