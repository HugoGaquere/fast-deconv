#pragma once

#include <vector_types.h>
#include <cstdint>
#include <vector>

namespace fast_deconv::algorithm::wscms {

struct Facets {
  std::vector<std::vector<float>> edges;
  std::vector<float2> centers;
};

struct Params {
  float peak_factor;
  std::uint32_t max_iter;
  float2 cell_size_radian;
};

}  // namespace fast_deconv::algorithm::wscms
