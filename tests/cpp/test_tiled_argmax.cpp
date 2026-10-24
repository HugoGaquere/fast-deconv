#include <gtest/gtest.h>

#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <memory>
#include <random>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/host_oracles.hpp"
#include "helpers/rng.hpp"

namespace core = fast_deconv::core;
namespace matrix = fast_deconv::matrix;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

// 32-px tiles over 200 px: a 7x7 grid whose last row and column of tiles are partial, [192, 200).
constexpr int kSide = 200;
constexpr int kTile = 32;

class TiledArgmax : public fdtest::BackendTest {};

// The unique max of an all-negative image, in the partial corner tile. Catches a running max initialised to
// 0 instead of -inf, and edge tiles that drop pixels.
TEST_F(TiledArgmax, FindsTheMaxOfAllNegativeDataInThePartialCornerTile)
{
  std::mt19937 rng(7);
  std::vector<float> img(kSide * kSide);
  fdtest::fill_uniform(rng, img, -2.0f, -1.0f);
  const int peak = flat(197, 195, kSide);
  img.at(peak) = -0.5f;

  const auto sr = res().make_ctx();
  matrix::tiled_argmax_ctx argmax{sr, core::dims<2>(kSide, kSide), kTile};

  const auto p = argmax.run(sr.stage(core::host_span2d<const float>(img.data(), kSide, kSide)));
  EXPECT_EQ(p.index, peak);
  EXPECT_FLOAT_EQ(p.value, -0.5f);
}

// The clean loop's sequence: run() seeds every tile under the scale's criterion, then after each subtraction
// run_incremental() rescans only the tiles under the PSF footprint, with the same criterion, and combines
// them with the cached maxima of the others. The footprint here is at the bottom-right corner, so it is
// clipped to the image and ends in the partial tiles.
TEST_F(TiledArgmax, IncrementalRefreshKeepsTheCriterionAndTheCachedTiles)
{
  std::vector<float> img(kSide * kSide, 0.0f);
  const auto mask = std::make_unique<bool[]>(kSide * kSide);
  const int a = flat(60, 170, kSide);        // far from the footprint
  const int b = flat(196, 198, kSide);       // inside it
  const int masked = flat(170, 180, kSide);  // inside it, masked
  mask[masked] = true;
  img.at(a) = -4.0f;

  const auto sr = res().make_ctx();
  const auto d_mask = sr.stage(core::host_span2d<const bool>(mask.get(), kSide, kSide));
  const auto stage_img = [&] { return sr.stage(core::host_span2d<const float>(img.data(), kSide, kSide)); };
  matrix::tiled_argmax_ctx argmax{sr, core::dims<2>(kSide, kSide), kTile};
  const auto refresh = [&] { return argmax.run_incremental(stage_img(), /*peak_row=*/190, /*peak_col=*/195, 64, 64); };

  // Seed: with abs, A wins by magnitude and keeps its sign.
  const auto p0 = argmax.run(stage_img(), {.mask = d_mask.data_handle(), .absolute = true});
  EXPECT_EQ(p0.index, a);
  EXPECT_FLOAT_EQ(p0.signed_value, -4.0f);

  // Inside the footprint, a bigger masked pixel and a bigger negative B: only B may win.
  img.at(masked) = 20.0f;
  img.at(b) = -6.0f;
  const auto p1 = refresh();
  EXPECT_EQ(p1.index, b);
  EXPECT_FLOAT_EQ(p1.value, 6.0f);
  EXPECT_FLOAT_EQ(p1.signed_value, -6.0f);

  // B drops below A: A's tile was not rescanned, so its cached maximum must win again.
  img.at(b) = 0.0f;
  EXPECT_EQ(refresh().index, a);
}
