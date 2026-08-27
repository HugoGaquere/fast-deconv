#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <stdexcept>

namespace fast_deconv::matrix {

peak tiled_argmax_ctx::run(core::span2d<float> /*data*/)
{
  throw std::runtime_error("matrix::tiled_argmax: host backend not implemented yet");
}

peak tiled_argmax_ctx::run_incremental(core::span2d<float> /*data*/, int /*peak_row*/, int /*peak_col*/,
                                       int /*foot_height*/, int /*foot_width*/)
{
  throw std::runtime_error("matrix::tiled_argmax: host backend not implemented yet");
}

}  // namespace fast_deconv::matrix
