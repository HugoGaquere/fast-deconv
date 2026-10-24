#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/gaussian_convolution.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/cufft_macros.hpp>

namespace fast_deconv::linalg {

namespace {

// Copies n_batch (nrow, ncol) blocks between stacks of planes whose rows hold src_ncol / dst_ncol floats and whose
// planes hold src_nrow / dst_nrow rows; one copy for the whole batch.
void copy_planes(const core::exec_ctx& ctx, const float* src, std::int64_t src_nrow, std::int64_t src_ncol, float* dst,
                 std::int64_t dst_nrow, std::int64_t dst_ncol, std::int64_t nrow, std::int64_t ncol,
                 std::int64_t n_batch)
{
  // Plane stride is pitch * ysize; the extent's width is in bytes for linear memory.
  cudaMemcpy3DParms p{};
  p.srcPtr = make_cudaPitchedPtr(const_cast<float*>(src), core::narrow<std::size_t>(src_ncol) * sizeof(float),
                                 core::narrow<std::size_t>(src_ncol), core::narrow<std::size_t>(src_nrow));
  p.dstPtr = make_cudaPitchedPtr(dst, core::narrow<std::size_t>(dst_ncol) * sizeof(float),
                                 core::narrow<std::size_t>(dst_ncol), core::narrow<std::size_t>(dst_nrow));
  p.extent = make_cudaExtent(core::narrow<std::size_t>(ncol) * sizeof(float), core::narrow<std::size_t>(nrow),
                             core::narrow<std::size_t>(n_batch));
  p.kind = cudaMemcpyDeviceToDevice;
  CHECK_CUDA(cudaMemcpy3DAsync(&p, ctx.cuda_stream));
}

}  // namespace

void pad(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output, std::int64_t n_batch)
{
  ctx.zero(output, dims.padded_total() * n_batch);
  copy_planes(ctx, input, dims.input_nrow, dims.input_ncol, output, dims.padded_nrow, dims.padded_ncol,
              dims.input_nrow, dims.input_ncol, n_batch);
}

void crop(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output, std::int64_t n_batch)
{
  copy_planes(ctx, input, dims.padded_nrow, dims.padded_ncol, output, dims.input_nrow, dims.input_ncol,
              dims.input_nrow, dims.input_ncol, n_batch);
}

gaussian_convolution_ctx::gaussian_convolution_ctx(const core::exec_ctx& ctx, std::int64_t batch, std::int64_t nrow,
                                                   std::int64_t ncol, const std::vector<float>& sigmas, float factor)
    : ctx_(ctx),
      dims_(nrow, ncol, max_gaussian_reach(sigmas, factor)),
      batch_(batch),
      padded_(ctx.alloc_ptr<float>(batch * dims_.padded_total())),
      product_(ctx.alloc_ptr<complex_type>(batch * dims_.freq_total()))
{
  // cuFFT allocates each plan's workspace itself. The 64-bit plan: batch * plane can exceed INT_MAX.
  std::array<long long, 2> fft_size{dims_.padded_nrow, dims_.padded_ncol};
  auto make_plan = [&](cufftHandle& plan, cufftType type) {
    CUFFT_CALL(cufftCreate(&plan));
    std::size_t work_size = 0;
    CUFFT_CALL(cufftMakePlanMany64(plan, 2, fft_size.data(), nullptr, 1, 0, nullptr, 1, 0, type, batch, &work_size));
    CUFFT_CALL(cufftSetStream(plan, ctx.cuda_stream));
  };
  make_plan(r2c_, CUFFT_R2C);
  make_plan(c2r_, CUFFT_C2R);
}

gaussian_convolution_ctx::~gaussian_convolution_ctx()
{
  CUFFT_CALL(cufftDestroy(r2c_));
  CUFFT_CALL(cufftDestroy(c2r_));
}

void gaussian_convolution_ctx::forward_(float* input, complex_type* output) const
{
  CUFFT_CALL(cufftExecR2C(r2c_, input, output));
}

void gaussian_convolution_ctx::backward_(complex_type* input, float* output) const
{
  CUFFT_CALL(cufftExecC2R(c2r_, input, output));
}

gaussian_convolution_ctx::spectrum gaussian_convolution_ctx::make_spectrum() const
{
  return ctx_.alloc_mdcontainer<complex_type>(batch_ * dims_.freq_total());
}

void gaussian_convolution_ctx::forward(core::span3d<const float> input, spectrum& out) const
{
  FD_PROFILE_FN();
  assert(input.is_exhaustive() && input.extent(0) == batch_);
  assert(input.extent(1) == dims_.input_nrow && input.extent(2) == dims_.input_ncol);
  assert(out.mapping().required_span_size() == batch_ * dims_.freq_total());

  pad(ctx_, dims_, input.data_handle(), padded_.get(), batch_);
  forward_(padded_.get(), out.data_handle());
}

void gaussian_convolution_ctx::convolve(const spectrum& in, float sigma, core::span3d<float> out) const
{
  FD_PROFILE_FN();
  assert(in.mapping().required_span_size() == batch_ * dims_.freq_total());
  assert(out.is_exhaustive() && out.extent(0) == batch_);
  assert(out.extent(1) == dims_.input_nrow && out.extent(2) == dims_.input_ncol);
  assert(gaussian_reach(sigma) <= std::min(dims_.padded_nrow - dims_.input_nrow, dims_.padded_ncol - dims_.input_ncol));

  // The multiply writes product_, so the caller's spectrum survives for the next sigma.
  multiply_with_gaussian(ctx_, dims_, batch_, in.data_handle(), product_.get(), sigma);
  backward_(product_.get(), padded_.get());
  crop(ctx_, dims_, padded_.get(), out.data_handle(), batch_);
}

// The multiply runs in place on product_: no spectrum survives for another sigma.
void gaussian_convolution_ctx::convolve(core::span3d<const float> input, float sigma, core::span3d<float> out) const
{
  FD_PROFILE_FN();
  assert(input.is_exhaustive() && input.extent(0) == batch_);
  assert(input.extent(1) == dims_.input_nrow && input.extent(2) == dims_.input_ncol);
  assert(out.is_exhaustive() && out.extent(0) == batch_);
  assert(out.extent(1) == dims_.input_nrow && out.extent(2) == dims_.input_ncol);
  assert(gaussian_reach(sigma) <= std::min(dims_.padded_nrow - dims_.input_nrow, dims_.padded_ncol - dims_.input_ncol));

  pad(ctx_, dims_, input.data_handle(), padded_.get(), batch_);
  forward_(padded_.get(), product_.get());
  multiply_with_gaussian(ctx_, dims_, batch_, product_.get(), product_.get(), sigma);
  backward_(product_.get(), padded_.get());
  crop(ctx_, dims_, padded_.get(), out.data_handle(), batch_);
}

}  // namespace fast_deconv::linalg
