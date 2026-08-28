#include <fast_deconv/morphology/roi.hpp>
#include <stdexcept>

namespace fast_deconv::morphology {

common::roi compute_mask_roi(const core::exec_ctx& ctx, core::span2d<bool> data)
{
  throw std::runtime_error("morphology::compute_mask_roi: host backend not implemented yet");
}

}  // namespace fast_deconv::morphology
