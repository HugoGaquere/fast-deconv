#include <fast_deconv/matrix/rms.hpp>
#include <stdexcept>

namespace fast_deconv::matrix {

float rms(const core::exec_ctx& /*ctx*/, core::span2d<float> /*data*/, core::span2d<bool> /*mask*/)
{
  throw std::runtime_error("matrix::rms: host backend not implemented yet");
}

}  // namespace fast_deconv::matrix
