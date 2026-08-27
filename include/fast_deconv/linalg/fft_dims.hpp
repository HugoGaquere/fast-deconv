#pragma once
#include <cmath>
#include <utility>

namespace fast_deconv::linalg {

// Smallest m >= n whose prime factors are all in {2, 3, 5, 7}.
inline int next_fast_size(int n)
{
  static constexpr int radices[] = {2, 3, 5, 7};
  while (true) {
    int m = n;
    for (int r : radices)
      while (m % r == 0) m /= r;
    if (m == 1) return n;
    ++n;
  }
}

// Compute padding amounts (rows, cols) for a target padding factor.
inline std::pair<int, int> compute_padding(int npix_x, int npix_y, float padding)
{
  return {static_cast<int>(std::ceil((padding - 1.0f) * npix_x / 2.0f)),
          static_cast<int>(std::ceil((padding - 1.0f) * npix_y / 2.0f))};
}

/// Padded-grid geometry for an FFT convolution: where the unpadded input sits
/// inside the padded buffer and how big the half-complex spectrum is. Pure
/// arithmetic, shared by every backend.
struct fft_dims {
  int input_nrow = 0, input_ncol = 0;      // unpadded input size
  int padding_nrow = 0, padding_ncol = 0;  // input start offset within padded buffer
                                           // (= (padded - input) / 2; far side gets one extra
                                           // zero pixel when the difference is odd)
  int padded_nrow = 0, padded_ncol = 0;    // padded spatial size (rounded up to next 7-smooth)
  int freq_nrow = 0, freq_ncol = 0;        // half-complex frequency size

  fft_dims() = default;

  /// Round the padded size up to the next 7-smooth value so cuFFT picks
  /// Cooley-Tukey over Bluestein. When (padded - input) is odd, padding is
  /// asymmetric: input starts at offset `padding_*`; the far side gets one
  /// extra zero pixel. Both pad_ifftshift and fftshift_crop use this offset
  /// symmetrically, so the round-trip is exact.
  fft_dims(int nrow, int ncol, float padding) : input_nrow(nrow), input_ncol(ncol)
  {
    const auto [npad_row_min, npad_col_min] = compute_padding(nrow, ncol, padding);
    padded_nrow = next_fast_size(nrow + 2 * npad_row_min);
    padded_ncol = next_fast_size(ncol + 2 * npad_col_min);
    padding_nrow = (padded_nrow - nrow) / 2;
    padding_ncol = (padded_ncol - ncol) / 2;
    freq_nrow = padded_nrow;
    freq_ncol = padded_ncol / 2 + 1;
  }

  int input_total() const { return input_nrow * input_ncol; }
  int padded_total() const { return padded_nrow * padded_ncol; }
  int freq_total() const { return freq_nrow * freq_ncol; }
};

}  // namespace fast_deconv::linalg
