#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace fast_deconv::test {

// std::uniform_real_distribution / std::normal_distribution output sequences
// are implementation-defined, so seeded data built with them (and anything
// derived from it, like the non-regression baseline) would differ between
// standard libraries. Everything here consumes raw std::mt19937 draws through
// fully-specified arithmetic instead.

inline constexpr double two_pi = 6.283185307179586476925286766559;
inline constexpr double inv_2pow32 = 1.0 / 4294967296.0;  // 2^-32

// Uniform in [0, 1); consumes exactly one engine draw.
inline float uniform01(std::mt19937& rng) { return static_cast<float>(static_cast<double>(rng()) * inv_2pow32); }

// Uniform in [lo, hi); consumes exactly one engine draw.
inline float uniform(std::mt19937& rng, float lo, float hi) { return lo + (hi - lo) * uniform01(rng); }

inline void fill_uniform(std::mt19937& rng, std::vector<float>& v, float lo, float hi)
{
  for (auto& x : v) x = uniform(rng, lo, hi);
}

// Standard normal via Box–Muller; consumes exactly two engine draws.
inline float normal01(std::mt19937& rng)
{
  const double u1 = (static_cast<double>(rng()) + 1.0) * inv_2pow32;  // (0, 1], keeps log() finite
  const double u2 = static_cast<double>(rng()) * inv_2pow32;          // [0, 1)
  return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2));
}

inline void add_normal_noise(std::mt19937& rng, std::vector<float>& v, float sigma)
{
  for (auto& x : v) x += sigma * normal01(rng);
}

}  // namespace fast_deconv::test
