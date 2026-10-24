#pragma once

namespace fast_deconv::matrix {

/// What max_and_rms() sums x and x^2 in: float, as CUB's tree reduction keeps the partial sums short.
using stats_sum_type = float;

}  // namespace fast_deconv::matrix
