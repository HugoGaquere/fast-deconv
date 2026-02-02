#pragma once

#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace fast_deconv::algorithm::wscms::detail {

// All spectral fitting is done on the host since the matrices are tiny
// (nchan ~ 4-16, order ~ 2-8). This avoids cuBLAS row/column-major issues
// and the overhead of launching many small kernels.

struct SpectralFitWorkspace {
    // Device buffer for per-channel coefficients (float32, for use in subtract kernels)
    float* d_per_channel_f;
    size_t total_bytes;

    static SpectralFitWorkspace allocate(int nch, int /*order*/, cudaStream_t stream) {
        SpectralFitWorkspace ws{};
        ws.total_bytes = sizeof(float) * nch;
        CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&ws.d_per_channel_f),
                                    ws.total_bytes, stream));
        return ws;
    }

    void free(cudaStream_t stream) {
        CHECK_CUDA(cudaFreeAsync(d_per_channel_f, stream));
    }
};

// Host-side small matrix inverse using Gauss-Jordan elimination.
// Matrix is row-major, dim x dim.
inline bool small_matrix_inverse_host(const double* src, double* dst, int dim) {
    double aug[MAX_SPECTRAL_ORDER * 2 * MAX_SPECTRAL_ORDER];
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            aug[i * 2 * dim + j] = src[i * dim + j];
            aug[i * 2 * dim + dim + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int col = 0; col < dim; col++) {
        int pivot = col;
        double max_val = std::abs(aug[col * 2 * dim + col]);
        for (int row = col + 1; row < dim; row++) {
            double val = std::abs(aug[row * 2 * dim + col]);
            if (val > max_val) { max_val = val; pivot = row; }
        }
        if (max_val < 1e-15) return false;

        if (pivot != col) {
            for (int j = 0; j < 2 * dim; j++)
                std::swap(aug[col * 2 * dim + j], aug[pivot * 2 * dim + j]);
        }

        double diag = aug[col * 2 * dim + col];
        for (int j = 0; j < 2 * dim; j++)
            aug[col * 2 * dim + j] /= diag;

        for (int row = 0; row < dim; row++) {
            if (row == col) continue;
            double factor = aug[row * 2 * dim + col];
            for (int j = 0; j < 2 * dim; j++)
                aug[row * 2 * dim + j] -= factor * aug[col * 2 * dim + j];
        }
    }

    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            dst[i * dim + j] = aug[i * 2 * dim + dim + j];

    return true;
}

// Host-side spectral fit.
// Copies small vectors from device, computes on host, copies per_channel back.
inline void spectral_fit(
    const MinorCycleContext& ctx,
    core::span_4d<float> dirty,   // (nch, npol, h, w)
    int x, int y,
    SpectralFitWorkspace& ws,
    float* h_coeffs_out,    // host output: compact coefficients (order,)
    int& n_coeffs_out,
    core::stream_resources& resources)
{
    auto stream = resources.stream;
    const int nch = static_cast<int>(ctx.Xdes.extent(0));
    const int order = static_cast<int>(ctx.Xdes.extent(1));
    const int h = static_cast<int>(dirty.extent(2));
    const int w = static_cast<int>(dirty.extent(3));

    n_coeffs_out = order;

    // Sync to ensure all prior GPU work is done before we read from device
    resources.sync();

    // Copy small arrays to host
    // Xdes (nch x order)
    float h_Xdes[MAX_SPECTRAL_ORDER * MAX_SPECTRAL_ORDER];
    CHECK_CUDA(cudaMemcpy(h_Xdes, ctx.Xdes.data_handle(),
                          sizeof(float) * nch * order, cudaMemcpyDeviceToHost));

    // sqrt_weights (nch)
    float h_sqrt_w[MAX_SPECTRAL_ORDER];
    CHECK_CUDA(cudaMemcpy(h_sqrt_w, ctx.sqrt_weights.data_handle(),
                          sizeof(float) * nch, cudaMemcpyDeviceToHost));

    // jones_norm[:, 0, x, y] - strided extraction
    float h_jn[MAX_SPECTRAL_ORDER];
    {
        const size_t jn_ch_stride = ctx.jones_norm.extent(1) * ctx.jones_norm.extent(2) * ctx.jones_norm.extent(3);
        const size_t pixel_offset = static_cast<size_t>(x) * w + y;
        for (int ch = 0; ch < nch; ch++) {
            CHECK_CUDA(cudaMemcpy(&h_jn[ch],
                ctx.jones_norm.data_handle() + ch * jn_ch_stride + pixel_offset,
                sizeof(float), cudaMemcpyDeviceToHost));
        }
    }

    // dirty[:, 0, x, y] - strided extraction
    float h_dirty_col[MAX_SPECTRAL_ORDER];
    {
        const size_t dirty_ch_stride = dirty.extent(1) * dirty.extent(2) * dirty.extent(3);
        const size_t pixel_offset = static_cast<size_t>(x) * w + y;
        for (int ch = 0; ch < nch; ch++) {
            CHECK_CUDA(cudaMemcpy(&h_dirty_col[ch],
                dirty.data_handle() + ch * dirty_ch_stride + pixel_offset,
                sizeof(float), cudaMemcpyDeviceToHost));
        }
    }

    // Step 1: Compute SAX = sqrt(jones_norm) * Xdes (if beam_enable)
    double h_SAX[MAX_SPECTRAL_ORDER * MAX_SPECTRAL_ORDER]; // (nch x order)
    for (int ch = 0; ch < nch; ch++) {
        double scale = ctx.beam_enable ? std::sqrt(static_cast<double>(h_jn[ch])) : 1.0;
        for (int o = 0; o < order; o++) {
            h_SAX[ch * order + o] = scale * static_cast<double>(h_Xdes[ch * order + o]);
        }
    }

    // Step 2: Compute WX = sqrt_weights * SAX (float64)
    double h_WX[MAX_SPECTRAL_ORDER * MAX_SPECTRAL_ORDER]; // (nch x order)
    for (int ch = 0; ch < nch; ch++) {
        double sw = static_cast<double>(h_sqrt_w[ch]);
        for (int o = 0; o < order; o++) {
            h_WX[ch * order + o] = sw * h_SAX[ch * order + o];
        }
    }

    // Step 3: Compute pseudo-inverse
    double h_pinv[MAX_SPECTRAL_ORDER * MAX_SPECTRAL_ORDER]; // (order x nch)
    if (nch >= order) {
        // gram = WX^T @ WX (order x order)
        double h_gram[MAX_SPECTRAL_ORDER * MAX_SPECTRAL_ORDER];
        for (int i = 0; i < order; i++) {
            for (int j = 0; j < order; j++) {
                double sum = 0.0;
                for (int k = 0; k < nch; k++)
                    sum += h_WX[k * order + i] * h_WX[k * order + j];
                h_gram[i * order + j] = sum;
            }
        }

        // gram_inv = inv(gram)
        double h_gram_inv[MAX_SPECTRAL_ORDER * MAX_SPECTRAL_ORDER];
        small_matrix_inverse_host(h_gram, h_gram_inv, order);

        // pinv = gram_inv @ WX^T : (order x order) @ (order x nch) = (order x nch)
        for (int i = 0; i < order; i++) {
            for (int j = 0; j < nch; j++) {
                double sum = 0.0;
                for (int k = 0; k < order; k++)
                    sum += h_gram_inv[i * order + k] * h_WX[j * order + k]; // WX^T[k][j] = WX[j][k]
                h_pinv[i * nch + j] = sum;
            }
        }
    } else {
        // gram = WX @ WX^T (nch x nch)
        double h_gram[MAX_SPECTRAL_ORDER * MAX_SPECTRAL_ORDER];
        for (int i = 0; i < nch; i++) {
            for (int j = 0; j < nch; j++) {
                double sum = 0.0;
                for (int k = 0; k < order; k++)
                    sum += h_WX[i * order + k] * h_WX[j * order + k];
                h_gram[i * nch + j] = sum;
            }
        }

        double h_gram_inv[MAX_SPECTRAL_ORDER * MAX_SPECTRAL_ORDER];
        small_matrix_inverse_host(h_gram, h_gram_inv, nch);

        // pinv = WX^T @ gram_inv : (order x nch) @ (nch x nch) = (order x nch)
        for (int i = 0; i < order; i++) {
            for (int j = 0; j < nch; j++) {
                double sum = 0.0;
                for (int k = 0; k < nch; k++)
                    sum += h_WX[k * order + i] * h_gram_inv[k * nch + j]; // WX^T[i][k] = WX[k][i]
                h_pinv[i * nch + j] = sum;
            }
        }
    }

    // Step 4: Wy = sqrt_weights * apparent_flux
    double h_Wy[MAX_SPECTRAL_ORDER];
    for (int ch = 0; ch < nch; ch++)
        h_Wy[ch] = static_cast<double>(h_sqrt_w[ch]) * static_cast<double>(h_dirty_col[ch]);

    // Step 5: coeffs = pinv @ Wy : (order x nch) @ (nch,) = (order,)
    double h_coeffs_d[MAX_SPECTRAL_ORDER];
    for (int i = 0; i < order; i++) {
        double sum = 0.0;
        for (int j = 0; j < nch; j++)
            sum += h_pinv[i * nch + j] * h_Wy[j];
        h_coeffs_d[i] = sum;
    }

    // Step 6: per_channel = SAX @ coeffs : (nch x order) @ (order,) = (nch,)
    float h_per_channel[MAX_SPECTRAL_ORDER];
    for (int ch = 0; ch < nch; ch++) {
        double sum = 0.0;
        for (int o = 0; o < order; o++)
            sum += h_SAX[ch * order + o] * h_coeffs_d[o];
        h_per_channel[ch] = static_cast<float>(sum);
    }

    // Copy per_channel to device for use in subtract kernels
    CHECK_CUDA(cudaMemcpyAsync(ws.d_per_channel_f, h_per_channel,
                                sizeof(float) * nch, cudaMemcpyHostToDevice, stream));

    // Output compact coefficients
    for (int i = 0; i < order; i++)
        h_coeffs_out[i] = static_cast<float>(h_coeffs_d[i]);
}

}  // namespace fast_deconv::algorithm::wscms::detail
