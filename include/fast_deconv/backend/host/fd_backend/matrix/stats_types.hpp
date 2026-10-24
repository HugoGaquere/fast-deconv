#pragma once

namespace fast_deconv::matrix {

/// What max_and_rms() sums x and x^2 in: double, as a long float sum on a host thread drifts where the device tree
/// reduction does not.
using stats_sum_type = double;

}  // namespace fast_deconv::matrix
