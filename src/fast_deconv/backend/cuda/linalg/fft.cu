#include <algorithm>
#include <array>
#include <cassert>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/convolution.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/cufft_macros.hpp>

namespace fast_deconv::kernel {

// Pads and ifftshifts a 2D image in one pass.
// Input:  (nx, ny) real, origin at center
// Output: (px, py) real, origin at (0,0), zero-padded
__global__ void pad_ifftshift_kernel(const float* input, float* output, int nx, int ny, int px, int py, int npad_x,
                                     int npad_y)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nx || col >= ny) return;

  // Position in padded array (input centered)
  const int pad_row = row + npad_x;
  const int pad_col = col + npad_y;

  // ifftshift: shift by ceil(N/2) = (N+1)/2
  const int out_row = (pad_row + (px + 1) / 2) % px;
  const int out_col = (pad_col + (py + 1) / 2) % py;

  output[out_row * py + out_col] = input[row * ny + col];
}

// Pads and ifftshifts a batched 2D image in one pass.
__global__ void pad_ifftshift_batched_kernel(const float* input, float* output, int nx, int ny, int px, int py,
                                             int npad_x, int npad_y, int n_batch)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nx || col >= ny) return;

  const int pad_row = row + npad_x;
  const int pad_col = col + npad_y;

  const int out_row = (pad_row + (px + 1) / 2) % px;
  const int out_col = (pad_col + (py + 1) / 2) % py;

  const int in_stride = nx * ny;
  const int out_stride = px * py;

  const float* in = input + row * ny + col;
  float* out = output + out_row * py + out_col;
  for (int b = 0; b < n_batch; b++, in += in_stride, out += out_stride) *out = *in;
}

// Fftshifts and crops a batched 2D image in one pass.
__global__ void fftshift_crop_kernel(const float* input, float* output, int nx, int ny, int px, int py, int npad_x,
                                     int npad_y, int n_batch)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nx || col >= ny) return;

  const int src_row = (row + npad_x + (px + 1) / 2) % px;
  const int src_col = (col + npad_y + (py + 1) / 2) % py;

  const int out_stride = nx * ny;
  const int in_stride = px * py;

  const float* in = input + src_row * py + src_col;
  float* out = output + row * ny + col;
  for (int b = 0; b < n_batch; b++, in += in_stride, out += out_stride) *out = *in;
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::linalg {

void pad_ifftshift_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output)
{
  CHECK_CUDA(cudaMemsetAsync(output, 0, sizeof(float) * dims.padded_total(), ctx.cuda_stream));
  dim3 block_dim(16, 16);
  dim3 grid_dim(CEIL_DIV(dims.input_ncol, block_dim.x), CEIL_DIV(dims.input_nrow, block_dim.y));
  kernel::pad_ifftshift_kernel<<<grid_dim, block_dim, 0, ctx.cuda_stream>>>(
      input, output, dims.input_nrow, dims.input_ncol, dims.padded_nrow, dims.padded_ncol, dims.padding_nrow,
      dims.padding_ncol);
}

void pad_ifftshift_batched_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output,
                                 int n_batch)
{
  CHECK_CUDA(cudaMemsetAsync(output, 0, sizeof(float) * dims.padded_total() * n_batch, ctx.cuda_stream));
  dim3 block_dim(16, 16);
  dim3 grid_dim(CEIL_DIV(dims.input_ncol, block_dim.x), CEIL_DIV(dims.input_nrow, block_dim.y));
  kernel::pad_ifftshift_batched_kernel<<<grid_dim, block_dim, 0, ctx.cuda_stream>>>(
      input, output, dims.input_nrow, dims.input_ncol, dims.padded_nrow, dims.padded_ncol, dims.padding_nrow,
      dims.padding_ncol, n_batch);
}

void fftshift_crop_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output,
                         int n_batch)
{
  dim3 block_dim(16, 16);
  dim3 grid_dim(CEIL_DIV(dims.input_ncol, block_dim.x), CEIL_DIV(dims.input_nrow, block_dim.y));
  kernel::fftshift_crop_kernel<<<grid_dim, block_dim, 0, ctx.cuda_stream>>>(
      input, output, dims.input_nrow, dims.input_ncol, dims.padded_nrow, dims.padded_ncol, dims.padding_nrow,
      dims.padding_ncol, n_batch);
}

convolution_ctx::convolution_ctx(const core::exec_ctx& ctx, int nrow, int ncol, float padding, int batch)
    : ctx_(ctx),
      dims_(nrow, ncol, padding),
      batch_(batch),
      padded_(ctx.alloc_ptr_async<float>(static_cast<std::size_t>(batch) * dims_.padded_total())),
      product_(ctx.alloc_ptr_async<complex_type>(static_cast<std::size_t>(batch) * dims_.freq_total()))
{
  // cuFFT allocates each plan's workspace itself.
  std::array<int, 2> fft_size{dims_.padded_nrow, dims_.padded_ncol};
  auto make_plan = [&](cufftHandle& plan, cufftType type) {
    CUFFT_CALL(cufftCreate(&plan));
    std::size_t work_size = 0;
    CUFFT_CALL(cufftMakePlanMany(plan, 2, fft_size.data(), nullptr, 1, 0, nullptr, 1, 0, type, batch, &work_size));
    CUFFT_CALL(cufftSetStream(plan, ctx.cuda_stream));
  };
  make_plan(r2c_, CUFFT_R2C);
  make_plan(c2r_, CUFFT_C2R);
}

convolution_ctx::~convolution_ctx()
{
  CUFFT_CALL(cufftDestroy(r2c_));
  CUFFT_CALL(cufftDestroy(c2r_));
}

void convolution_ctx::forward_(float* input, complex_type* output) const
{
  CUFFT_CALL(cufftExecR2C(r2c_, input, output));
}

void convolution_ctx::backward_(complex_type* input, float* output) const
{
  CUFFT_CALL(cufftExecC2R(c2r_, input, output));
}

void convolution_ctx::forward(core::span3d<const float> input, core::span1d<complex_type> spectrum) const
{
  FD_PROFILE_FN();
  assert(input.is_exhaustive() && input.extent(0) == batch_);
  assert(input.extent(1) == dims_.input_nrow && input.extent(2) == dims_.input_ncol);
  assert(spectrum.size() == static_cast<std::size_t>(batch_) * dims_.freq_total());

  pad_ifftshift_batched_async(ctx_, dims_, input.data_handle(), padded_.get(), batch_);
  forward_(padded_.get(), spectrum.data_handle());
}

void convolution_ctx::convolve_spectrum(core::span1d<const complex_type> spectrum, float sigma,
                                        core::span3d<float> out) const
{
  FD_PROFILE_FN();
  assert(spectrum.size() == static_cast<std::size_t>(batch_) * dims_.freq_total());
  assert(out.is_exhaustive() && out.extent(0) == batch_);
  assert(out.extent(1) == dims_.input_nrow && out.extent(2) == dims_.input_ncol);

  // The multiply writes product_, so the caller's spectrum survives for the next sigma.
  multiply_with_gaussian(ctx_, dims_, batch_, spectrum.data_handle(), product_.get(), sigma);
  backward_(product_.get(), padded_.get());
  fftshift_crop_async(ctx_, dims_, padded_.get(), out.data_handle(), batch_);
}

}  // namespace fast_deconv::linalg
