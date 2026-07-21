#include <gtest/gtest.h>

#include <cstdint>
#include <fast_deconv/algorithm/ddmsc_types.hpp>
#include <fast_deconv/util/utils.hpp>
#include <utility>
#include <vector>

// Pure index math and host-side bookkeeping — no GPU work.

namespace util = fast_deconv::util;
namespace ddmsc = fast_deconv::algorithm::ddmsc;

// ============================================================================
// util::unravel_index_2D — row-major flat index → (y, x)
// ============================================================================

TEST(UnravelIndex2D, CornersAndInteriorOnNonSquareGrid)
{
  const int width = 7;  // grid is (height=5, width=7), deliberately non-square

  EXPECT_EQ(util::unravel_index_2D(0, width), (std::pair<int, int>{0, 0}));
  EXPECT_EQ(util::unravel_index_2D(width - 1, width), (std::pair<int, int>{0, width - 1}));
  EXPECT_EQ(util::unravel_index_2D(width, width), (std::pair<int, int>{1, 0}));

  const int y = 3, x = 4;
  EXPECT_EQ(util::unravel_index_2D(y * width + x, width), (std::pair<int, int>{y, x}));

  const int last = 5 * width - 1;
  EXPECT_EQ(util::unravel_index_2D(last, width), (std::pair<int, int>{4, width - 1}));
}

// ============================================================================
// ddmsc_result::add_component — parallel-array bookkeeping
// ============================================================================

TEST(DdmscResult, AddComponentKeepsParallelArraysInSync)
{
  ddmsc::ddmsc_result result(/*max_iter=*/10, /*coeff_order=*/2);
  EXPECT_TRUE(result.peak_coords.empty());
  EXPECT_TRUE(result.scales.empty());
  EXPECT_TRUE(result.gains.empty());

  result.add_component({3, 4}, /*scale=*/1, /*gain=*/0.1f);
  result.add_component({7, 2}, /*scale=*/0, /*gain=*/0.2f);

  ASSERT_EQ(result.peak_coords.size(), 2u);
  ASSERT_EQ(result.scales.size(), 2u);
  ASSERT_EQ(result.gains.size(), 2u);

  EXPECT_EQ(result.peak_coords.at(0), (std::pair<int, int>{3, 4}));
  EXPECT_EQ(result.peak_coords.at(1), (std::pair<int, int>{7, 2}));
  EXPECT_EQ(result.scales.at(0), 1);
  EXPECT_EQ(result.scales.at(1), 0);
  EXPECT_FLOAT_EQ(result.gains.at(0), 0.1f);
  EXPECT_FLOAT_EQ(result.gains.at(1), 0.2f);
}
