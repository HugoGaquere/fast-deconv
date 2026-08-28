#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <fast_deconv/linalg/fft.hpp>
#include <utility>
#include <vector>

#include "helpers/device_buffers.hpp"
#include "helpers/gpu_test.hpp"
#include "helpers/host_oracles.hpp"

namespace linalg = fast_deconv::linalg;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

// ============================================================================
// next_fast_size / compute_padding — pure host
// ============================================================================

namespace {

bool is_7_smooth(int n)
{
  for (int r : {2, 3, 5, 7})
    while (n % r == 0) n /= r;
  return n == 1;
}

}  // namespace

TEST(NextFastSize, KnownValues)
{
  EXPECT_EQ(linalg::next_fast_size(1), 1);
  EXPECT_EQ(linalg::next_fast_size(7), 7);
  EXPECT_EQ(linalg::next_fast_size(11), 12);
  EXPECT_EQ(linalg::next_fast_size(13), 14);
  EXPECT_EQ(linalg::next_fast_size(121), 125);
  EXPECT_EQ(linalg::next_fast_size(509), 512);
  EXPECT_EQ(linalg::next_fast_size(512), 512);
}

TEST(NextFastSize, ReturnsSmallest7SmoothAtLeastN)
{
  for (int n = 1; n <= 2000; ++n) {
    const int m = linalg::next_fast_size(n);
    ASSERT_GE(m, n);
    ASSERT_TRUE(is_7_smooth(m)) << "next_fast_size(" << n << ") = " << m << " is not 7-smooth";
    for (int k = n; k < m; ++k)
      ASSERT_FALSE(is_7_smooth(k)) << "next_fast_size(" << n << ") = " << m << " but " << k << " is 7-smooth";
  }
}

TEST(ComputePadding, MatchesCeilFormulaPerAxis)
{
  // pad = ceil((padding - 1) * npix / 2), computed independently per axis.
  EXPECT_EQ(linalg::compute_padding(100, 100, 1.5f), (std::pair<int, int>{25, 25}));
  EXPECT_EQ(linalg::compute_padding(100, 60, 1.5f), (std::pair<int, int>{25, 15}));
  EXPECT_EQ(linalg::compute_padding(5, 6, 1.5f), (std::pair<int, int>{2, 2}));  // ceil(1.25), ceil(1.5)
  EXPECT_EQ(linalg::compute_padding(100, 100, 1.0f), (std::pair<int, int>{0, 0}));
}

// ============================================================================
// pad_ifftshift / pad_ifftshift_batched / fftshift_crop — pure data movement,
// compared element-exact against a host replica of the index mapping.
// ============================================================================

namespace {

// Host oracle for pad_ifftshift: input (nx, ny) centered into (px, py) at
// offset (npad_x, npad_y), then origin moved to (0,0) by an ifftshift of the
// padded grid; everything else zero.
std::vector<float> host_pad_ifftshift(const std::vector<float>& in, int nx, int ny, int px, int py, int npad_x,
                                      int npad_y)
{
  std::vector<float> out(static_cast<std::size_t>(px) * py, 0.0f);
  for (int r = 0; r < nx; ++r) {
    for (int c = 0; c < ny; ++c) {
      const int out_r = (r + npad_x + (px + 1) / 2) % px;
      const int out_c = (c + npad_y + (py + 1) / 2) % py;
      out.at(flat(out_r, out_c, py)) = in.at(flat(r, c, ny));
    }
  }
  return out;
}

// The layout kernels only read the geometry, so set it explicitly — these cases
// pick padded sizes the padding-factor constructor would not produce.
linalg::fft_dims explicit_dims(int nx, int ny, int px, int py)
{
  linalg::fft_dims d;
  d.input_nrow = nx;
  d.input_ncol = ny;
  d.padded_nrow = px;
  d.padded_ncol = py;
  d.padding_nrow = (px - nx) / 2;
  d.padding_ncol = (py - ny) / 2;
  d.freq_nrow = px;
  d.freq_ncol = py / 2 + 1;
  return d;
}

// Distinct, order-revealing values so any permutation of the layout shows up.
std::vector<float> iota_image(int n, float offset = 0.0f)
{
  std::vector<float> img(n);
  for (int i = 0; i < n; ++i) img.at(i) = offset + static_cast<float>(i + 1);
  return img;
}

}  // namespace

class FftLayout : public fdtest::GpuTest {};

TEST_F(FftLayout, PadIfftshiftMatchesHostOracle)
{
  // Odd pad deltas on both axes (8-5=3, 9-6=3) exercise the convention where
  // the far side gets the extra zero pixel: npad = (padded - input) / 2.
  const int nx = 5, ny = 6, px = 8, py = 9;
  const int npad_x = (px - nx) / 2, npad_y = (py - ny) / 2;

  const auto in = iota_image(nx * ny);
  const auto sr = res().make_ctx();

  fdtest::device_buffer<float> d_in(res(), sr, in);
  fdtest::device_buffer<float> d_out(res(), sr, static_cast<std::size_t>(px) * py);

  linalg::pad_ifftshift_async(sr, explicit_dims(nx, ny, px, py), d_in.get(), d_out.get());
  sr.wait();

  const auto out = d_out.to_host();
  const auto expected = host_pad_ifftshift(in, nx, ny, px, py, npad_x, npad_y);
  for (int i = 0; i < px * py; ++i) ASSERT_FLOAT_EQ(out.at(i), expected.at(i)) << "flat index " << i;
}

TEST_F(FftLayout, PadIfftshiftBatchedMatchesSingleImageOracle)
{
  const int nx = 5, ny = 6, px = 8, py = 9, n_batch = 3;
  const int npad_x = (px - nx) / 2, npad_y = (py - ny) / 2;
  const int in_stride = nx * ny, out_stride = px * py;

  // Three distinct slices so cross-batch mixups are visible.
  std::vector<float> in(n_batch * in_stride);
  for (int b = 0; b < n_batch; ++b) {
    const auto slice = iota_image(in_stride, 100.0f * b);
    std::copy(slice.begin(), slice.end(), in.begin() + b * in_stride);
  }

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_in(res(), sr, in);
  fdtest::device_buffer<float> d_out(res(), sr, static_cast<std::size_t>(n_batch) * out_stride);

  linalg::pad_ifftshift_batched_async(sr, explicit_dims(nx, ny, px, py), d_in.get(), d_out.get(), n_batch);
  sr.wait();

  const auto out = d_out.to_host();
  for (int b = 0; b < n_batch; ++b) {
    const std::vector<float> slice(in.begin() + b * in_stride, in.begin() + (b + 1) * in_stride);
    const auto expected = host_pad_ifftshift(slice, nx, ny, px, py, npad_x, npad_y);
    for (int i = 0; i < out_stride; ++i)
      ASSERT_FLOAT_EQ(out.at(b * out_stride + i), expected.at(i)) << "batch " << b << " flat index " << i;
  }
}

TEST_F(FftLayout, PadThenCropRoundTripIsIdentity)
{
  // pad_ifftshift → fftshift_crop must reproduce the input bit-exactly for
  // both odd and even pad deltas (the asymmetric-padding convention is used
  // symmetrically by both kernels).
  const struct {
    int nx, ny, px, py;
  } cases[] = {
      {5, 6, 8, 9},    // odd deltas
      {6, 6, 10, 12},  // even deltas
      {7, 5, 7, 5},    // no padding at all
  };

  const auto sr = res().make_ctx();

  for (const auto& cs : cases) {
    const auto in = iota_image(cs.nx * cs.ny);

    fdtest::device_buffer<float> d_in(res(), sr, in);
    fdtest::device_buffer<float> d_pad(res(), sr, static_cast<std::size_t>(cs.px) * cs.py);
    fdtest::device_buffer<float> d_back(res(), sr, static_cast<std::size_t>(cs.nx) * cs.ny);

    const auto dims = explicit_dims(cs.nx, cs.ny, cs.px, cs.py);
    linalg::pad_ifftshift_async(sr, dims, d_in.get(), d_pad.get());
    linalg::fftshift_crop_async(sr, dims, d_pad.get(), d_back.get(), /*n_batch=*/1);
    sr.wait();

    const auto back = d_back.to_host();
    for (int i = 0; i < cs.nx * cs.ny; ++i)
      ASSERT_FLOAT_EQ(back.at(i), in.at(i))
          << "case (" << cs.nx << "x" << cs.ny << " -> " << cs.px << "x" << cs.py << "), flat index " << i;
  }
}

TEST_F(FftLayout, BatchedRoundTripIsIdentityPerSlice)
{
  const int nx = 5, ny = 6, px = 8, py = 9, n_batch = 3;
  const int in_stride = nx * ny, pad_stride = px * py;

  std::vector<float> in(n_batch * in_stride);
  for (int b = 0; b < n_batch; ++b) {
    const auto slice = iota_image(in_stride, 100.0f * b);
    std::copy(slice.begin(), slice.end(), in.begin() + b * in_stride);
  }

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_in(res(), sr, in);
  fdtest::device_buffer<float> d_pad(res(), sr, static_cast<std::size_t>(n_batch) * pad_stride);
  fdtest::device_buffer<float> d_back(res(), sr, static_cast<std::size_t>(n_batch) * in_stride);

  const auto dims = explicit_dims(nx, ny, px, py);
  linalg::pad_ifftshift_batched_async(sr, dims, d_in.get(), d_pad.get(), n_batch);
  linalg::fftshift_crop_async(sr, dims, d_pad.get(), d_back.get(), n_batch);
  sr.wait();

  const auto back = d_back.to_host();
  for (int i = 0; i < n_batch * in_stride; ++i) ASSERT_FLOAT_EQ(back.at(i), in.at(i)) << "flat index " << i;
}
