#include <fast_deconv/linalg/pseudo_inverse.hpp>
#include <stdexcept>

namespace fast_deconv::linalg {

void compute_pseudo_inverse(const core::exec_ctx& /*ctx*/, const float* /*d_A*/, float* /*d_A_pinv*/, int /*n_rows*/,
                            int /*n_cols*/)
{
  throw std::runtime_error("linalg::compute_pseudo_inverse: host backend not implemented yet");
}

}  // namespace fast_deconv::linalg
