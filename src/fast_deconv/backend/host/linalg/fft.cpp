#include <fast_deconv/linalg/fft.hpp>
#include <stdexcept>
#include <string>

namespace fast_deconv::linalg {

namespace {
[[noreturn]] void not_implemented(const char* what)
{
  throw std::runtime_error(std::string(what) + ": host backend not implemented yet");
}
}  // namespace

void pad_ifftshift_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output)
{
  not_implemented("linalg::pad_ifftshift_async");
}

void pad_ifftshift_batched_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output,
                                 int n_batch)
{
  not_implemented("linalg::pad_ifftshift_batched_async");
}

void fftshift_crop_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output,
                         int n_batch)
{
  not_implemented("linalg::fftshift_crop_async");
}

void convolve_ctx::forward_async(float* input, complex_type* output) const
{
  not_implemented("linalg::convolve_ctx::forward_async");
}

void convolve_ctx::backward_async(complex_type* input, float* output, int plan_idx) const
{
  not_implemented("linalg::convolve_ctx::backward_async");
}

}  // namespace fast_deconv::linalg
