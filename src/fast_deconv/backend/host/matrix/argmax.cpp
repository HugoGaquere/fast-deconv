#include <fast_deconv/matrix/argmax.hpp>
#include <stdexcept>

namespace fast_deconv::matrix {

void argmax_ctx::run_async(core::span2d<float> /*data*/)
{
  throw std::runtime_error("matrix::argmax: host backend not implemented yet");
}

peak argmax_ctx::run(core::span2d<float> /*data*/)
{
  throw std::runtime_error("matrix::argmax: host backend not implemented yet");
}

}  // namespace fast_deconv::matrix
