#pragma once
#include <cstdint>

namespace fast_deconv::linalg {

// Smallest m >= n whose prime factors are all in {2, 3, 5, 7}.
inline std::int64_t next_fast_size(std::int64_t n)
{
  static constexpr std::int64_t radices[] = {2, 3, 5, 7};
  while (true) {
    std::int64_t m = n;
    for (std::int64_t r : radices)
      while (m % r == 0) m /= r;
    if (m == 1) return n;
    ++n;
  }
}

/**
 * @brief Padded-grid geometry for the FFT convolution.
 *
 * The input sits at the top-left (0, 0) of the padded buffer with zeros after it, plus the half-complex
 * spectrum size. Pure arithmetic.
 */
struct fft_dims {
  std::int64_t input_nrow = 0, input_ncol = 0;    // unpadded input size
  std::int64_t padded_nrow = 0, padded_ncol = 0;  // padded spatial size: next 7-smooth >= input + gap
  std::int64_t freq_nrow = 0, freq_ncol = 0;      // half-complex frequency size

  fft_dims() = default;

  /**
   * @brief At least @p gap zero pixels after the input on each axis.
   *
   * The padded size is rounded up to the next 7-smooth value so cuFFT picks Cooley-Tukey over Bluestein.
   *
   * @param[in] nrow  Input rows.
   * @param[in] ncol  Input columns.
   * @param[in] gap   Minimum zero padding in pixels per axis.
   */
  fft_dims(std::int64_t nrow, std::int64_t ncol, std::int64_t gap) : input_nrow(nrow), input_ncol(ncol)
  {
    padded_nrow = next_fast_size(nrow + gap);
    padded_ncol = next_fast_size(ncol + gap);
    freq_nrow = padded_nrow;
    freq_ncol = padded_ncol / 2 + 1;
  }

  std::int64_t padded_total() const { return padded_nrow * padded_ncol; }
  std::int64_t freq_total() const { return freq_nrow * freq_ncol; }
};

}  // namespace fast_deconv::linalg
