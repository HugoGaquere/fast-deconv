#include <algorithm>
#include <array>
#include <fast_deconv/linalg/fft.hpp>
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

convolve_ctx::convolve_ctx(const core::exec_ctx& ctx, int input_nrow, int input_ncol, int forward_batch,
                           int backward_batch, int n_backward_plans, float padding)
    : ctx_(ctx),
      dims_(input_nrow, input_ncol, padding),
      forward_batch_(forward_batch),
      backward_batch_(backward_batch),
      plans_forward_(1),
      plans_backward_(n_backward_plans)
{
  // Disable cuFFT auto-allocation so all plans share a single caller-managed workspace.
  // Plans execute sequentially, so one buffer of max(plan_work_sizes) suffices.
  // See cuFFT §2.14 Caller Allocated Work Area.
  std::array<int, 2> fft_size{dims_.padded_nrow, dims_.padded_ncol};
  auto make_plan = [&](cufftHandle& plan, cufftType type, int batch) {
    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftSetAutoAllocation(plan, 0));
    size_t plan_work = 0;
    CUFFT_CALL(cufftMakePlanMany(plan, 2, fft_size.data(), nullptr, 1, 0, nullptr, 1, 0, type, batch, &plan_work));
    work_size_ = std::max(work_size_, plan_work);
    CUFFT_CALL(cufftSetStream(plan, ctx.cuda_stream));
  };

  make_plan(plans_forward_[0], CUFFT_R2C, forward_batch);
  for (auto& p : plans_backward_) make_plan(p, CUFFT_C2R, backward_batch);
}

convolve_ctx::~convolve_ctx()
{
  for (auto p : plans_forward_) CUFFT_CALL(cufftDestroy(p));
  for (auto p : plans_backward_) CUFFT_CALL(cufftDestroy(p));
}

void convolve_ctx::forward_async(float* input, complex_type* output) const
{
  CUFFT_CALL(cufftExecR2C(plans_forward_[0], input, output));
}

void convolve_ctx::backward_async(complex_type* input, float* output, int plan_idx) const
{
  CUFFT_CALL(cufftExecC2R(plans_backward_.at(plan_idx), input, output));
}

void convolve_ctx::bind_work_area(void* work_area)
{
  for (auto p : plans_forward_) CUFFT_CALL(cufftSetWorkArea(p, work_area));
  for (auto p : plans_backward_) CUFFT_CALL(cufftSetWorkArea(p, work_area));
}

}  // namespace fast_deconv::linalg
