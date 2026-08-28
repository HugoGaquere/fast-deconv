#include <fast_deconv/common/gain.hpp>
#include <stdexcept>

namespace fast_deconv::common {

namespace {
[[noreturn]] void not_implemented() { throw std::runtime_error("common::gain: host backend not implemented yet"); }
}  // namespace

std::vector<float> compute_gain_batched(const core::exec_ctx& /*ctx*/, const core::span4d<float>& /*psfs*/,
                                        const core::span1d<const float>& /*weights_freq*/, float /*gamma*/)
{
  not_implemented();
}

std::vector<float> compute_all_gains_batched(const core::exec_ctx& /*ctx*/, const core::span5d<float>& /*psfs*/,
                                             const core::span1d<const float>& /*weights_freq*/, float /*gamma*/)
{
  not_implemented();
}

}  // namespace fast_deconv::common
