#pragma once
#include <cufft.h>

#include <fast_deconv/util/cuda_macros.hpp>

#ifndef CUFFT_CALL
#define CUFFT_CALL(call)                                               \
  {                                                                    \
    auto status = static_cast<cufftResult>(call);                      \
    if (status != CUFFT_SUCCESS)                                       \
      fprintf(stderr,                                                  \
              "ERROR: CUFFT call \"%s\" in line %d of file %s failed " \
              "with "                                                  \
              "code (%d).\n",                                          \
              #call, __LINE__, __FILE__, status);                      \
  }
#endif  // CUFFT_CALL

using complex_type = cufftComplex;

namespace fast_deconv::linalg::detail {

// Pads and ifftshifts a 2D image in one pass.
// Input:  (nx, ny) real, origin at center
// Output: (px, py) real, origin at (0,0), zero-padded
// Output must be zeroed before launch.
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

// Fftshifts and crops a batched 2D image in one pass.
// Input:  (n_batch, px, py) real, origin at (0,0) (FFT output)
// Output: (n_batch, nx, ny) real, origin at center, cropped
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

void pad_ifftshift(float* input, float* output, int nx, int ny, int px, int py, int npad_x,
                   int npad_y, cudaStream_t stream)
{
  cudaMemsetAsync(output, 0, sizeof(float) * px * py, stream);
  dim3 block_dim(16, 16);
  dim3 grid_dim(CEIL_DIV(ny, block_dim.x), CEIL_DIV(nx, block_dim.y));
  pad_ifftshift_kernel<<<grid_dim, block_dim, 0, stream>>>(input, output, nx, ny, px, py, npad_x,
                                                           npad_y);
}

void fftshift_crop(float* input, float* output, int nx, int ny, int px, int py, int npad_x,
                   int npad_y, int n_batch, cudaStream_t stream)
{
  dim3 block_dim(16, 16);
  dim3 grid_dim(CEIL_DIV(ny, block_dim.x), CEIL_DIV(nx, block_dim.y));
  fftshift_crop_kernel<<<grid_dim, block_dim, 0, stream>>>(input, output, nx, ny, px, py, npad_x,
                                                           npad_y, n_batch);
}

inline std::pair<int, int> compute_padding(int npix_x, int npix_y, float padding)
{
  return {static_cast<int>(ceilf((padding - 1.0f) * npix_x / 2.0f)),
          static_cast<int>(ceilf((padding - 1.0f) * npix_y / 2.0f))};
}

}  // namespace fast_deconv::linalg::detail
