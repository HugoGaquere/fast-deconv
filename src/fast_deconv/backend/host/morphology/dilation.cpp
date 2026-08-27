#include <fast_deconv/morphology/dilation.hpp>
#include <stdexcept>

namespace fast_deconv::morphology {

void binary_dilation(const core::exec_ctx& /*ctx*/, core::span2d<bool> /*data*/, core::span2d<bool> /*structure*/,
                     roi /*structure_roi*/, core::span2d<bool> /*out*/)
{
  throw std::runtime_error("morphology::binary_dilation: host backend not implemented yet");
}

}  // namespace fast_deconv::morphology
