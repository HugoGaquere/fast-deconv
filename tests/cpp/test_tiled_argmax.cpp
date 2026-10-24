#include <gtest/gtest.h>

#include <cstdint>
#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <random>
#include <utility>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/device_buffers.hpp"
#include "helpers/host_oracles.hpp"
#include "helpers/rng.hpp"

namespace core = fast_deconv::core;
namespace matrix = fast_deconv::matrix;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

class TiledArgmax : public fdtest::BackendTest {
 protected:
  // Upload @p img, run a full pass over a fresh ctx, free, return result.
  matrix::peak run_once(const core::exec_ctx& sr, const std::vector<float>& img, int w, int h, int tile)
  {
    fdtest::device_buffer<float> d(sr, img);

    matrix::tiled_argmax_ctx ws{sr, core::dims<2>(h, w), tile};

    auto out = ws.run(core::span2d<float>(d.get(), h, w));
    sr.wait();
    return out;
  }
};

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

  auto [idx, val, signed_val] = run_once(sr, img, w, h, 32);
  EXPECT_FLOAT_EQ(val, 9.0f);
  EXPECT_EQ(idx, flat(pr, pc, w));
}

// All-negative image. Catches the classic bug of initialising the running max
// to 0.0f / data[0] instead of -inf: a buggy kernel reports 0 at a bogus
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

  auto [idx, val, signed_val] = run_once(sr, img, w, h, 32);
  EXPECT_FLOAT_EQ(val, -0.5f);
  EXPECT_EQ(idx, flat(pr, pc, w));
}

// Incremental path: seed every tile with a full pass, then dirty only a footprint
// (as a clean subtraction would) and refresh via run_incremental. The new peak
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

  fdtest::device_buffer<float> d(sr, img);

  matrix::tiled_argmax_ctx ws{sr, core::dims<2>(h, w), 32};
  const core::span2d<float> view(d.get(), h, w);

  // Full pass seeds all tiles; global max is A.
  auto [i0, v0, signed_v0] = ws.run(view);
  sr.wait();
  EXPECT_FLOAT_EQ(v0, 5.0f);
  EXPECT_EQ(i0, flat(ar, ac, w));

  // "Clean" a footprint centered at (pr, pc): plant a bigger peak B inside it and
  // push only those pixels to the device (everything else is unchanged).
  const int pr = 120, pc = 110, foot = 64;  // footprint [88,152) x [78,142)
  const int br = 125, bc = 118;             // inside the footprint
  img.at(flat(br, bc, w)) = 7.0f;
  sr.copy(d.get() + flat(br, bc, w), &img.at(flat(br, bc, w)), 1);

  auto [i1, v1, signed_v1] = ws.run_incremental(view, pr, pc, foot, foot);
  sr.wait();
  EXPECT_FLOAT_EQ(v1, 7.0f);
  EXPECT_EQ(i1, flat(br, bc, w));

  // A sat in an untouched tile: its cached maximum must still be combined in. Drop
  // B back below A and refresh the same footprint; A must re-emerge as the winner.
  img.at(flat(br, bc, w)) = 0.0f;
  sr.copy(d.get() + flat(br, bc, w), &img.at(flat(br, bc, w)), 1);
  auto [i2, v2, signed_v2] = ws.run_incremental(view, pr, pc, foot, foot);
  sr.wait();
  EXPECT_FLOAT_EQ(v2, 5.0f);
  EXPECT_EQ(i2, flat(ar, ac, w));
}

// The clean loop searches under a peak_criterion, and run_incremental reuses the one of the last run().
// A huge masked pixel must never win, and abs must rank a negative pixel by its magnitude and keep its sign
// in signed_value, in the full pass and in the incremental refresh alike.
TEST_F(TiledArgmax, CriterionMasksAndRanksByAbsInFullAndIncrementalPasses)
{
  const int w = 200, h = 200;  // 32-tiles -> 7x7 grid (ragged edges)
  std::mt19937 rng(31);
  std::vector<float> img(w * h);
  fdtest::fill_uniform(rng, img, -1.0f, 1.0f);
  std::vector<uint8_t> mask(w * h);
  for (auto& m : mask) m = fdtest::uniform01(rng) < 0.2f ? 1 : 0;

  const int masked = flat(150, 20, w), negative = flat(60, 170, w);
  img.at(masked) = 9.0f;
  mask.at(masked) = 1;
  img.at(negative) = -4.0f;
  mask.at(negative) = 0;
  // The mask is static over a run: fix the footprint pixels' entries now, their values come later.
  const int masked2 = flat(100, 90, w), negative2 = flat(130, 120, w);
  mask.at(masked2) = 1;
  mask.at(negative2) = 0;

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d(sr, img);
  fdtest::device_buffer<bool> d_mask(sr, std::vector<bool>(mask.begin(), mask.end()));

  matrix::tiled_argmax_ctx ws{sr, core::dims<2>(h, w), 32};
  const core::span2d<float> view(d.get(), h, w);

  const auto p0 = ws.run(view, {.mask = d_mask.get(), .absolute = true});
  sr.wait();
  EXPECT_EQ(p0.index, negative);
  EXPECT_FLOAT_EQ(p0.value, fdtest::masked_max(img, mask, /*use_abs=*/true));
  EXPECT_FLOAT_EQ(p0.signed_value, -4.0f);

  // Inside a footprint: a bigger masked pixel and a bigger negative one. Only the latter may win.
  const int pr = 120, pc = 110, foot = 64;  // footprint [88,152) x [78,142)
  img.at(masked2) = 20.0f;
  img.at(negative2) = -6.0f;
  d.from_host(img);

  const auto p1 = ws.run_incremental(view, pr, pc, foot, foot);
  sr.wait();
  EXPECT_EQ(p1.index, negative2);
  EXPECT_FLOAT_EQ(p1.value, 6.0f);
  EXPECT_FLOAT_EQ(p1.signed_value, -6.0f);
}
