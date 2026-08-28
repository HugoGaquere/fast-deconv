#include <fast_deconv/common/clean.hpp>
#include <stdexcept>

namespace fast_deconv::common {

namespace {
[[noreturn]] void not_implemented() {
  throw std::runtime_error("common::clean: host backend not implemented yet");
}
} // namespace

void subtract_component_async(const core::exec_ctx &ctx,
                              core::span2d<float> residual,
                              core::span2d<float> psf, index2d peak_coords,
                              float gain) {
  not_implemented();
}

void subtract_component_async(const core::exec_ctx &ctx,
                              core::span3d<float> residual,
                              core::span3d<float> psf,
                              core::span1d<float> spectral_coeffs,
                              index2d peak_coords, float gain) {
  not_implemented();
}

} // namespace fast_deconv::common
