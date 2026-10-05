#pragma once
#include <fast_deconv/core/dims.hpp>

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

/// Padded-grid geometry for the FFT convolution: the input sits at the top-left (0, 0) of the padded
/// buffer with zeros after it, and the half-complex spectrum size. Pure arithmetic.
struct fft_dims {
  int input_nrow = 0, input_ncol = 0;    // unpadded input size
  int padded_nrow = 0, padded_ncol = 0;  // padded spatial size: next 7-smooth >= input + gap
  int freq_nrow = 0, freq_ncol = 0;      // half-complex frequency size

  fft_dims() = default;

  /// At least @p gap zero pixels after the input on each axis. The padded size is rounded up to the next
  /// 7-smooth value so cuFFT picks Cooley-Tukey over Bluestein.
  fft_dims(int nrow, int ncol, int gap) : input_nrow(nrow), input_ncol(ncol)
  {
    padded_nrow = next_fast_size(nrow + gap);
    padded_ncol = next_fast_size(ncol + gap);
    freq_nrow = padded_nrow;
    freq_ncol = padded_ncol / 2 + 1;
    // padded_total()/freq_total()/input_total() all return int; padding is the caller's.
    core::check_plane_fits_int32(padded_nrow, padded_ncol, "padded fft");
  }

  int input_total() const { return input_nrow * input_ncol; }
  int padded_total() const { return padded_nrow * padded_ncol; }
  int freq_total() const { return freq_nrow * freq_ncol; }
};

}  // namespace fast_deconv::linalg
