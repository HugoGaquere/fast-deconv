#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "helpers/device_buffers.hpp"
#include "helpers/gpu_test.hpp"
#include "helpers/host_oracles.hpp"
#include "helpers/rng.hpp"

namespace core = fast_deconv::core;
namespace matrix = fast_deconv::matrix;
namespace fdtest = fast_deconv::test;

using fdtest::cpu_argmax;
using fdtest::flat;

class TiledArgmax : public fdtest::GpuTest {
 protected:
  // Upload @p img, run a full pass over a fresh ctx, free, return result.
  matrix::peak run_once(const core::exec_ctx& sr, const std::vector<float>& img, int w, int h, int tile)
  {
    fdtest::device_buffer<float> d(res(), sr, img);

    matrix::tiled_argmax_ctx ws{sr, core::dims<2>(h, w), tile};

    auto out = ws.run(core::span2d<float>(d.get(), h, w));
    sr.wait();
    return out;  // ws frees its own device buffers in its destructor
  }
};

// A single planted peak in an otherwise-zero image. Smallest sanity check:
// does the block reduce find the value and carry the right flat index?
TEST_F(TiledArgmax, SingleKnownPeak)
{
  const int w = 64, h = 64;
  std::vector<float> img(w * h, 0.0f);
  const int pr = 40, pc = 50;
  img.at(flat(pr, pc, w)) = 5.0f;

  const auto sr = res().make_ctx();

  auto [val, idx] = run_once(sr, img, w, h, 32);
  EXPECT_FLOAT_EQ(val, 5.0f);
  EXPECT_EQ(idx, flat(pr, pc, w));
}

// Ragged geometry: 130x70 with 32x32 tiles -> partial edge tiles on both axes.
// Random data, so we only assert on the VALUE (tie-safe) against the CPU max.
TEST_F(TiledArgmax, RandomRaggedMatchesCpuValue)
{
  const int w = 130, h = 70;
  std::mt19937 rng(1234);
  std::vector<float> img(w * h);
  fdtest::fill_uniform(rng, img, -1.0f, 1.0f);

  const auto sr = res().make_ctx();

  auto [val, idx] = run_once(sr, img, w, h, 32);
  auto [ref_val, ref_idx] = cpu_argmax(img);

  EXPECT_FLOAT_EQ(val, ref_val);
  // Value at the returned index must equal the reported value (index sanity).
  ASSERT_GE(idx, 0);
  ASSERT_LT(idx, static_cast<int>(img.size()));
  EXPECT_FLOAT_EQ(img.at(idx), val);
}

// Unique global peak planted in a far, ragged corner tile. Checks BOTH value
// and index across many tiles + a partial edge tile.
TEST_F(TiledArgmax, UniquePeakIndexAcrossTiles)
{
  const int w = 200, h = 200;  // 32-tiles -> 7x7 grid, last tiles are 8 wide/tall
  std::mt19937 rng(7);
  std::vector<float> img(w * h);
  fdtest::fill_uniform(rng, img, 0.0f, 1.0f);

  const int pr = 177, pc = 183;    // inside the ragged bottom-right region
  img.at(flat(pr, pc, w)) = 9.0f;  // strictly above everything in [0,1)

  const auto sr = res().make_ctx();

  auto [val, idx] = run_once(sr, img, w, h, 32);
  EXPECT_FLOAT_EQ(val, 9.0f);
  EXPECT_EQ(idx, flat(pr, pc, w));
}

// All-negative image. Catches the classic bug of initialising the running max
// to 0.0f / data[0] instead of -FLT_MAX: a buggy kernel reports 0 at a bogus
// index here. The planted -0.5 is the unique (least-negative) maximum.
TEST_F(TiledArgmax, AllNegativeInitialisesToNegInf)
{
  const int w = 96, h = 96;
  std::mt19937 rng(99);
  std::vector<float> img(w * h);
  fdtest::fill_uniform(rng, img, -2.0f, -1.0f);

  const int pr = 70, pc = 11;
  img.at(flat(pr, pc, w)) = -0.5f;  // unique max, still negative

  const auto sr = res().make_ctx();

  auto [val, idx] = run_once(sr, img, w, h, 32);
  EXPECT_FLOAT_EQ(val, -0.5f);
  EXPECT_EQ(idx, flat(pr, pc, w));
}

// Reuse one ctx across two calls, mimicking
// the clean loop: peak moves between cycles, the second full pass must find it.
// (The incremental dirty-footprint path is covered by IncrementalRefreshesDirtyFootprint.)
TEST_F(TiledArgmax, WorkspaceReuseAcrossCalls)
{
  const int w = 64, h = 64;
  const auto sr = res().make_ctx();

  fdtest::device_buffer<float> d(res(), sr, static_cast<std::size_t>(w) * h);
  matrix::tiled_argmax_ctx ws{sr, core::dims<2>(h, w), 32};
  const core::span2d<float> view(d.get(), h, w);

  // Cycle 1: peak A.
  std::vector<float> img(w * h, 0.0f);
  const int ar = 10, ac = 12;
  img.at(flat(ar, ac, w)) = 5.0f;
  d.from_host(img);
  auto [v1, i1] = ws.run(view);
  sr.wait();
  EXPECT_FLOAT_EQ(v1, 5.0f);
  EXPECT_EQ(i1, flat(ar, ac, w));

  // Cycle 2: clear A, plant a larger peak B elsewhere; reuse the same ctx.
  img.at(flat(ar, ac, w)) = 0.0f;
  const int br = 55, bc = 60;
  img.at(flat(br, bc, w)) = 7.0f;
  d.from_host(img);
  auto [v2, i2] = ws.run(view);
  sr.wait();
  EXPECT_FLOAT_EQ(v2, 7.0f);
  EXPECT_EQ(i2, flat(br, bc, w));
}

// Incremental path: seed every tile with a full pass, then dirty only a footprint
// (as a clean subtraction would) and refresh via argmax_incremental. The new peak
// planted inside the footprint must win; the untouched cached tiles must survive.
TEST_F(TiledArgmax, IncrementalRefreshesDirtyFootprint)
{
  const int w = 200, h = 200;  // 32-tiles -> 7x7 grid (ragged edges)
  std::mt19937 rng(2024);
  std::vector<float> img(w * h);
  fdtest::fill_uniform(rng, img, 0.0f, 1.0f);  // background in [0,1)

  // Initial global peak A, away from where we'll clean.
  const int ar = 30, ac = 40;
  img.at(flat(ar, ac, w)) = 5.0f;

  const auto sr = res().make_ctx();

  fdtest::device_buffer<float> d(res(), sr, img);

  matrix::tiled_argmax_ctx ws{sr, core::dims<2>(h, w), 32};
  const core::span2d<float> view(d.get(), h, w);

  // Full pass seeds all tiles; global max is A.
  auto [v0, i0] = ws.run(view);
  sr.wait();
  EXPECT_FLOAT_EQ(v0, 5.0f);
  EXPECT_EQ(i0, flat(ar, ac, w));

  // "Clean" a footprint centered at (pr, pc): plant a bigger peak B inside it and
  // push only those pixels to the device (everything else is unchanged).
  const int pr = 120, pc = 110, foot = 64;  // footprint [88,152) x [78,142)
  const int br = 125, bc = 118;             // inside the footprint
  img.at(flat(br, bc, w)) = 7.0f;
  CHECK_CUDA(cudaMemcpyAsync(d.get() + flat(br, bc, w), &img.at(flat(br, bc, w)), sizeof(float), cudaMemcpyHostToDevice,
                             sr.cuda_stream));

  auto [v1, i1] = ws.run_incremental(view, pr, pc, foot, foot);
  sr.wait();
  EXPECT_FLOAT_EQ(v1, 7.0f);
  EXPECT_EQ(i1, flat(br, bc, w));

  // A sat in an untouched tile: its cached maximum must still be combined in. Drop
  // B back below A and refresh the same footprint; A must re-emerge as the winner.
  img.at(flat(br, bc, w)) = 0.0f;
  CHECK_CUDA(cudaMemcpyAsync(d.get() + flat(br, bc, w), &img.at(flat(br, bc, w)), sizeof(float), cudaMemcpyHostToDevice,
                             sr.cuda_stream));
  auto [v2, i2] = ws.run_incremental(view, pr, pc, foot, foot);
  sr.wait();
  EXPECT_FLOAT_EQ(v2, 5.0f);
  EXPECT_EQ(i2, flat(ar, ac, w));
}
