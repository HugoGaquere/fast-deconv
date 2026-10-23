#pragma once
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>
#include <fd_backend/linalg/fft.hpp>

namespace fast_deconv::linalg {

/**
 * @brief out = sum_c weights[c] * A[c]: the weighted sum of the planes of @p A.
 *
 * Enqueued on @p ctx; wait() before reading @p out from the host.
 *
 * @param[in]  ctx      Execution context.
 * @param[in]  A        Planes, device, (n_planes, nrow, ncol).
 * @param[in]  weights  Weight per plane, device, (n_planes,).
 * @param[out] out      Weighted sum, device, (nrow, ncol), written entirely.
 */
void weighted_sum_async(const core::exec_ctx& ctx, core::span3d<const float> A, core::span1d<const float> weights,
                        core::span2d<float> out);

}  // namespace fast_deconv::linalg
