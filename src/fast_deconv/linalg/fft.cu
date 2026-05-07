#include <algorithm>
#include <array>
#include <cmath>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::kernel {

// Pads and ifftshifts a 2D image in one pass.
// Input:  (nx, ny) real, origin at center
// Output: (px, py) real, origin at (0,0), zero-padded
__global__ void pad_ifftshift_kernel(const float* input, float* output, int nx, int ny, int px,
                                     int py, int npad_x, int npad_y)
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
__global__ void pad_ifftshift_batched_kernel(const float* input, float* output, int nx, int ny,
                                             int px, int py, int npad_x, int npad_y, int n_batch)
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
  const int in_idx = row * ny + col;
  const int out_idx = out_row * py + out_col;

  for (int b = 0; b < n_batch; b++) {
    output[b * out_stride + out_idx] = input[b * in_stride + in_idx];
  }
}

// Fftshifts and crops a batched 2D image in one pass.
__global__ void fftshift_crop_kernel(const float* input, float* output, int nx, int ny, int px,
                                     int py, int npad_x, int npad_y, int n_batch)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nx || col >= ny) return;

  const int src_row = (row + npad_x + (px + 1) / 2) % px;
  const int src_col = (col + npad_y + (py + 1) / 2) % py;

  const int out_stride = nx * ny;
  const int in_stride = px * py;
  const int out_idx = row * ny + col;
  const int in_idx = src_row * py + src_col;

  for (int b = 0; b < n_batch; b++) {
    output[b * out_stride + out_idx] = input[b * in_stride + in_idx];
  }
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::linalg {

void pad_ifftshift(float* input, float* output, int nx, int ny, int px, int py, int npad_x,
                   int npad_y, cudaStream_t stream)
{
  cudaMemsetAsync(output, 0, sizeof(float) * px * py, stream);
  dim3 block_dim(16, 16);
  dim3 grid_dim(CEIL_DIV(ny, block_dim.x), CEIL_DIV(nx, block_dim.y));
  kernel::pad_ifftshift_kernel<<<grid_dim, block_dim, 0, stream>>>(input, output, nx, ny, px, py,
                                                                   npad_x, npad_y);
}

void pad_ifftshift_batched(float* input, float* output, int nx, int ny, int px, int py,
                           int npad_x, int npad_y, int n_batch, cudaStream_t stream)
{
  cudaMemsetAsync(output, 0, sizeof(float) * px * py * n_batch, stream);
  dim3 block_dim(16, 16);
  dim3 grid_dim(CEIL_DIV(ny, block_dim.x), CEIL_DIV(nx, block_dim.y));
  kernel::pad_ifftshift_batched_kernel<<<grid_dim, block_dim, 0, stream>>>(
      input, output, nx, ny, px, py, npad_x, npad_y, n_batch);
}

void fftshift_crop(float* input, float* output, int nx, int ny, int px, int py, int npad_x,
                   int npad_y, int n_batch, cudaStream_t stream)
{
  dim3 block_dim(16, 16);
  dim3 grid_dim(CEIL_DIV(ny, block_dim.x), CEIL_DIV(nx, block_dim.y));
  kernel::fftshift_crop_kernel<<<grid_dim, block_dim, 0, stream>>>(input, output, nx, ny, px, py,
                                                                   npad_x, npad_y, n_batch);
}

std::pair<int, int> compute_padding(int npix_x, int npix_y, float padding)
{
  return {static_cast<int>(ceilf((padding - 1.0f) * npix_x / 2.0f)),
          static_cast<int>(ceilf((padding - 1.0f) * npix_y / 2.0f))};
}

int next_fast_size(int n)
{
  static constexpr int radices[] = {2, 3, 5, 7};
  while (true) {
    int m = n;
    for (int r : radices) while (m % r == 0) m /= r;
    if (m == 1) return n;
    ++n;
  }
}

convolve_ctx::convolve_ctx(int input_nrow_, int input_ncol_, int forward_batch_, int backward_batch_,
                           int n_backward_plans, float padding)
{
  const auto [npad_row_min, npad_col_min] = compute_padding(input_nrow_, input_ncol_, padding);

  input_nrow = input_nrow_;
  input_ncol = input_ncol_;
  // Round padded size up to the next 7-smooth value so cuFFT picks Cooley-Tukey
  // over Bluestein. When (padded - input) is odd, padding is asymmetric: input
  // starts at offset `padding_*`; the far side gets one extra zero pixel. Both
  // pad_ifftshift and fftshift_crop use this offset symmetrically, so the
  // round-trip is exact.
  padded_nrow = next_fast_size(input_nrow_ + 2 * npad_row_min);
  padded_ncol = next_fast_size(input_ncol_ + 2 * npad_col_min);
  padding_nrow = (padded_nrow - input_nrow_) / 2;
  padding_ncol = (padded_ncol - input_ncol_) / 2;
  freq_nrow = padded_nrow;
  freq_ncol = padded_ncol / 2 + 1;
  forward_batch = forward_batch_;
  backward_batch = backward_batch_;
  plans_forward.resize(1);
  plans_backward.resize(n_backward_plans);

  // Disable cuFFT auto-allocation so all plans share a single caller-managed workspace.
  // Plans execute sequentially, so one buffer of max(plan_work_sizes) suffices.
  // See cuFFT §2.14 Caller Allocated Work Area.
  std::array<int, 2> fft_size{padded_nrow, padded_ncol};
  auto make_plan = [&](cufftHandle& plan, cufftType type, int batch) {
    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftSetAutoAllocation(plan, 0));
    size_t plan_work = 0;
    CUFFT_CALL(cufftMakePlanMany(plan, 2, fft_size.data(), nullptr, 1, 0, nullptr, 1, 0, type, batch, &plan_work));
    work_size = std::max(work_size, plan_work);
  };

  make_plan(plans_forward[0], CUFFT_R2C, forward_batch);
  for (auto& p : plans_backward) make_plan(p, CUFFT_C2R, backward_batch);
}

}  // namespace fast_deconv::linalg
