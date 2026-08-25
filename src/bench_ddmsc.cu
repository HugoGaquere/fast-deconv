/// Benchmark: run the DDMSC minor-cycle loop on synthetic data with fixed
/// stopping criteria, sweeping over a cartesian grid of DDMSC parameters.
///
/// Unlike example_ddmsc (which loads real dump_ref exports), this driver
/// generates all inputs in-process so the benchmark has no dependency on the
/// slow Python pipeline and can vary problem dimensions freely.
///
/// The stopping criteria are pinned (see make_fixed_params) so that each run
/// performs a deterministic amount of work: exactly K outer scale-selections,
/// each running M clean iterations, for K*M total minor iterations. With the
/// data-dependent early exits neutralised, the loop's wall-time becomes a
/// function of the problem *dimensions*, not the sky *content* -- which is
/// exactly what makes synthetic data sufficient here.
///
/// For each config the timed region wraps only run_ddmsc_cycles. One
/// ddmsc::context is built per config and reused across every repetition
/// (warmup + timed) -- rebuilding it each run would re-pay the expensive cuFFT
/// plan creation for no measurement benefit, since the build is excluded from
/// timing. The auto-mask component history is the only state that carries
/// across run_ddmsc_cycles calls, so it is cleared before each run to keep the
/// work deterministic. The dirty image is reset from a pristine master copy
/// before each run because run_ddmsc_cycles mutates it in place.
///
/// Host-side synthetic inputs are content-independent for timing, so the
/// expensive per-pixel generators are run once and cached across configs (keyed
/// by the dimensions each buffer depends on); only the device uploads repeat.
///
/// Usage:
///   bench_ddmsc [--mode=cartesian|ofat]
///               [--sizes=L] [--nfreq=L] [--nscales=L] [--nfacet=L]
///               [--norder=L] [--psf-frac=L] [--K=L] [--M=L]
///               [--runs=N] [--warmup=N] [--device=N] [--seed=N]
///               [--csv=PATH] [--max-configs=N] [--force] [--dry-run]
///
/// Each axis flag takes a comma-separated list (ranges not supported). `--sizes`
/// sets nrow==ncol; `--psf-frac` sets the PSF side length as a fraction of the
/// image side. `--mode` controls how the lists are combined into configs:
///   cartesian (default) -- the full product of every axis.
///   ofat                -- one-factor-at-a-time: each non-size axis is varied
///                          across its list while the others are pinned to their
///                          median, and that is crossed with the full --sizes
///                          sweep. This is exactly the slice plot_bench.py keeps
///                          (it holds every other axis at its median), so it
///                          reproduces the same marginalized plots at a fraction
///                          of the cost.

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fast_deconv/algorithm/ddmsc_cycles.hpp>
#include <fast_deconv/algorithm/ddmsc_types.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fstream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace core = fast_deconv::core;
namespace ddmsc = fast_deconv::algorithm::ddmsc;

#define BENCH_CHECK_CUDA(call)                                                                            \
  do {                                                                                                    \
    cudaError_t _e = (call);                                                                              \
    if (_e != cudaSuccess) {                                                                              \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(_e)); \
      std::exit(1);                                                                                       \
    }                                                                                                     \
  } while (0)

// ------------------------------------------------------------------------- //
//  Config + CLI
// ------------------------------------------------------------------------- //

struct bench_config {
  int nrow, ncol;
  int n_freq;
  int n_scales;
  int n_facet;
  int n_order;
  int psf_nrow, psf_ncol;
  int K;  // outer scale-selections
  int M;  // clean iterations per scale-selection
};

struct bench_options {
  std::vector<int> sizes{1024};
  std::vector<int> nfreqs{4};
  std::vector<int> nscales{4};
  std::vector<int> nfacets{1};
  std::vector<int> norders{2};
  std::vector<double> psf_fracs{1.0};
  std::vector<int> Ks{5};
  std::vector<int> Ms{250};
  int runs = 10;
  int warmup = 3;
  int device = 0;
  uint64_t seed = 42;
  std::string csv_path;
  int max_configs = 500;
  bool force = false;
  bool dry_run = false;
  std::string mode = "cartesian";  // cartesian | ofat
};

static std::vector<int> parse_int_list(const std::string& s)
{
  std::vector<int> out;
  std::size_t i = 0;
  while (i <= s.size()) {
    const std::size_t j = s.find(',', i);
    const std::string tok = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
    if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return out;
}

static std::vector<double> parse_double_list(const std::string& s)
{
  std::vector<double> out;
  std::size_t i = 0;
  while (i <= s.size()) {
    const std::size_t j = s.find(',', i);
    const std::string tok = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
    if (!tok.empty()) out.push_back(std::atof(tok.c_str()));
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return out;
}

// Median of a list, matching plot_bench.py's baseline: sort the distinct values
// and take the middle element. Used as the OFAT hold-at value so the generated
// cross lines up with the rows the plotter selects.
template <typename T>
static T median_value(std::vector<T> v)
{
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v.at(v.size() / 2);
}

// ------------------------------------------------------------------------- //
//  Synthetic data generation (host side; uploaded once per config)
// ------------------------------------------------------------------------- //

// DDFacet-style scale extents in pixels (see scripts/ddmsc_sigmas_bias.md):
// scale 0 is the delta scale (alpha == 0); subsequent scales grow ~geometrically
// and are forced odd. We synthesise a plausible sequence of the requested length.
static std::vector<double> gen_alphas(int n_scales)
{
  std::vector<double> a;
  a.reserve(n_scales);
  a.push_back(0.0);  // delta scale
  if (n_scales >= 2) a.push_back(4.0);
  double prev = 4.0;
  while (static_cast<int>(a.size()) < n_scales) {
    double next = std::ceil(1.5 * prev);
    int ni = static_cast<int>(next);
    if (ni % 2 == 0) ni += 1;  // force odd, mirroring the reference
    a.push_back(static_cast<double>(ni));
    prev = ni;
  }
  return a;
}

// sigma_i = (alpha_i == 0) ? 0 : 3*alpha_i/16
static std::vector<float> gen_scale_sigmas(const std::vector<double>& alphas)
{
  std::vector<float> sig(alphas.size());
  for (std::size_t i = 0; i < alphas.size(); ++i)
    sig[i] = alphas[i] == 0.0 ? 0.0f : static_cast<float>(3.0 * alphas[i] / 16.0);
  return sig;
}

// bias[0] = 1; bias[i] = beta^(-log2(alpha_i / 8))
static std::vector<float> gen_scale_bias(const std::vector<double>& alphas, double beta = 0.6)
{
  std::vector<float> bias(alphas.size(), 1.0f);
  for (std::size_t i = 1; i < alphas.size(); ++i) {
    if (alphas[i] <= 0.0) continue;
    bias[i] = static_cast<float>(std::pow(beta, -std::log2(alphas[i] / 8.0)));
  }
  return bias;
}

// Spectral design matrix: Taylor basis x_f^o around the band centre.
static std::vector<float> gen_xdes(int n_freq, int n_order)
{
  std::vector<float> xdes(static_cast<std::size_t>(n_freq) * n_order);
  for (int f = 0; f < n_freq; ++f) {
    const double x = (f - (n_freq - 1) / 2.0) / std::max(1, n_freq);
    double p = 1.0;
    for (int o = 0; o < n_order; ++o) {
      xdes[static_cast<std::size_t>(f) * n_order + o] = static_cast<float>(p);
      p *= x;
    }
  }
  return xdes;
}

// Synthetic dirty beam: central Gaussian (peak 1.0) plus a couple of decaying
// sidelobe rings. Identical across facets and frequencies.
static std::vector<float> gen_psfs(int n_facet, int n_freq, int psf_nrow, int psf_ncol)
{
  const int cy = psf_nrow / 2, cx = psf_ncol / 2;
  const double core_sigma = std::max(1.0, std::min(psf_nrow, psf_ncol) / 40.0);
  std::vector<float> plane(static_cast<std::size_t>(psf_nrow) * psf_ncol);
  for (int y = 0; y < psf_nrow; ++y) {
    for (int x = 0; x < psf_ncol; ++x) {
      const double dy = y - cy, dx = x - cx;
      const double r = std::sqrt(dy * dy + dx * dx);
      const double core = std::exp(-(r * r) / (2.0 * core_sigma * core_sigma));
      // decaying oscillatory sidelobes
      const double sidelobe = 0.08 * std::exp(-r / (8.0 * core_sigma)) * std::cos(r / core_sigma);
      plane[static_cast<std::size_t>(y) * psf_ncol + x] = static_cast<float>(core + sidelobe);
    }
  }
  plane[static_cast<std::size_t>(cy) * psf_ncol + cx] = 1.0f;  // pin exact peak

  std::vector<float> psfs(static_cast<std::size_t>(n_facet) * n_freq * psf_nrow * psf_ncol);
  const std::size_t plane_n = plane.size();
  for (std::size_t i = 0; i < static_cast<std::size_t>(n_facet) * n_freq; ++i)
    std::copy(plane.begin(), plane.end(), psfs.begin() + i * plane_n);
  return psfs;
}

// Per-pixel facet assignment: simple row-banding into n_facet contiguous bands.
static std::vector<int> gen_map_pixel_facet(int nrow, int ncol, int n_facet)
{
  std::vector<int> m(static_cast<std::size_t>(nrow) * ncol);
  for (int r = 0; r < nrow; ++r) {
    int fid = std::min(n_facet - 1, r * n_facet / nrow);
    for (int c = 0; c < ncol; ++c) m[static_cast<std::size_t>(r) * ncol + c] = fid;
  }
  return m;
}

// Multi-frequency dirty image: a mean sky of bright Gaussian sources (placed in
// the interior to keep PSF-subtraction ROIs unclipped) plus light noise, then
// scaled per frequency. Strong positive flux keeps the residual peak above the
// (zeroed) stop-flux through all K*M subtractions.
static std::vector<float> gen_dirty(int n_freq, int nrow, int ncol, uint64_t seed)
{
  std::mt19937_64 rng(seed);
  const std::size_t npix = static_cast<std::size_t>(nrow) * ncol;

  // Mean sky: windowed Gaussian sources.
  std::vector<float> mean(npix, 0.0f);
  const int margin = std::max(8, std::min(nrow, ncol) / 8);
  const int n_src = std::max(8, static_cast<int>(npix / 200000));  // scale with image
  std::uniform_int_distribution<int> ry(margin, nrow - margin - 1);
  std::uniform_int_distribution<int> rx(margin, ncol - margin - 1);
  std::uniform_real_distribution<double> amp(1.0, 10.0);
  std::uniform_real_distribution<double> wid(1.5, 8.0);
  for (int s = 0; s < n_src; ++s) {
    const int sy = ry(rng), sx = rx(rng);
    const double a = amp(rng), w = wid(rng);
    const int rad = static_cast<int>(std::ceil(3.0 * w));
    for (int dy = -rad; dy <= rad; ++dy) {
      const int y = sy + dy;
      if (y < 0 || y >= nrow) continue;
      for (int dx = -rad; dx <= rad; ++dx) {
        const int x = sx + dx;
        if (x < 0 || x >= ncol) continue;
        const double g = a * std::exp(-(dy * dy + dx * dx) / (2.0 * w * w));
        mean[static_cast<std::size_t>(y) * ncol + x] += static_cast<float>(g);
      }
    }
  }
  // Light positive-biased noise floor.
  std::normal_distribution<double> noise(0.02, 0.01);
  for (std::size_t i = 0; i < npix; ++i) mean[i] += static_cast<float>(noise(rng));

  // Replicate across frequency with a mild spectral slope (no per-channel noise
  // needed: with fixed iteration counts the timing is content-independent).
  std::vector<float> dirty(static_cast<std::size_t>(n_freq) * npix);
  for (int f = 0; f < n_freq; ++f) {
    const double x = (f - (n_freq - 1) / 2.0) / std::max(1, n_freq);
    const float scale = static_cast<float>(1.0 + 0.2 * x);
    for (std::size_t i = 0; i < npix; ++i) dirty[static_cast<std::size_t>(f) * npix + i] = mean[i] * scale;
  }
  return dirty;
}

// ------------------------------------------------------------------------- //
//  Fixed-criteria parameters
// ------------------------------------------------------------------------- //

// Pin every data-dependent early-exit so the loop runs exactly K*M minor
// iterations. See the loop guards in ddmsc.cu.
static ddmsc::params make_fixed_params(const bench_config& c)
{
  ddmsc::params p{};
  p.max_iteration = c.K * c.M;  // total minor iters across all scale selections
  p.divergence_factor = 1e30f;  // never "diverges"
  p.flux_threshold = -1e30f;    // floor never bites
  p.stop_rms_factor = 0.0f;     // composed stop_flux -> 0
  p.stop_peak_factor = 0.0f;
  p.stop_cycle_factor = 0.0f;  // disables sidelobe term
  p.stop_sidelobe_level = 0.0f;
  p.clean_negative = false;
  p.peak_factor = -1.0f;  // clean-loop threshold negative -> always runs full M
  p.gamma = 0.1f;
  p.max_clean_iteration = c.M;
  p.scale_stall_threshold = 0.0f;  // |drms| < 0 never true -> no stall / retirement
  p.enable_auto_mask = false;      // keep mask static -> content-independent timing
  p.force_enable_auto_mask = false;
  p.auto_mask_peak_threshold = std::nullopt;
  p.auto_mask_rms_threshold = std::nullopt;
  return p;
}

// ------------------------------------------------------------------------- //
//  Stats + result row
// ------------------------------------------------------------------------- //

struct run_stats {
  double mean_ms, std_ms, min_ms, max_ms, median_ms;
};

static run_stats summarize(std::vector<double> ms)
{
  run_stats s{};
  const std::size_t n = ms.size();
  std::sort(ms.begin(), ms.end());
  s.min_ms = ms.front();
  s.max_ms = ms.back();
  s.median_ms = (n % 2) ? ms[n / 2] : 0.5 * (ms[n / 2 - 1] + ms[n / 2]);
  s.mean_ms = std::accumulate(ms.begin(), ms.end(), 0.0) / n;
  double acc = 0.0;
  for (double v : ms) acc += (v - s.mean_ms) * (v - s.mean_ms);
  s.std_ms = (n > 1) ? std::sqrt(acc / (n - 1)) : 0.0;  // sample std
  return s;
}

// Padded spatial dims (rows, cols) for an input pair, mirroring exactly what
// the FFT contexts compute: linalg::compute_padding for the per-dim pad, then
// linalg::next_fast_size to round up to the nearest 7-smooth Cooley-Tukey size.
static std::pair<int, int> padded_spatial(int nrow, int ncol, float padding)
{
  namespace linalg = fast_deconv::linalg;
  const auto [npad_r, npad_c] = linalg::compute_padding(nrow, ncol, padding);
  return {linalg::next_fast_size(nrow + 2 * npad_r), linalg::next_fast_size(ncol + 2 * npad_c)};
}

// Exact cuFFT work-area bytes for one convolve_ctx-style plan set: a single R2C
// plan plus n_backward_plans C2R plans over the same padded grid, all sharing
// one work area sized to the max across plans (see convolve_ctx::convolve_ctx).
// cuFFT workspace is not analytically predictable, so we build the very same
// plans with auto-allocation disabled and read back the size cuFFT reports --
// this queries the size without reserving the work area. A failed query (e.g.
// size limits) falls back to one padded half-complex grid per batch.
static std::size_t cufft_work_bytes(int padded_nrow, int padded_ncol, int forward_batch, int backward_batch,
                                    int n_backward_plans)
{
  int fft_size[2] = {padded_nrow, padded_ncol};
  std::size_t work_max = 0;
  const std::size_t freq_grid = static_cast<std::size_t>(padded_nrow) * (padded_ncol / 2 + 1);
  auto query = [&](cufftType type, int batch) {
    cufftHandle plan = 0;
    if (cufftCreate(&plan) != CUFFT_SUCCESS) {
      work_max = std::max(work_max, freq_grid * sizeof(cufftComplex));
      return;
    }
    cufftSetAutoAllocation(plan, 0);
    std::size_t wk = 0;
    if (cufftMakePlanMany(plan, 2, fft_size, nullptr, 1, 0, nullptr, 1, 0, type, batch, &wk) == CUFFT_SUCCESS)
      work_max = std::max(work_max, wk);
    else
      work_max = std::max(work_max, freq_grid * sizeof(cufftComplex) * std::max(1, batch));
    cufftDestroy(plan);
  };
  query(CUFFT_R2C, forward_batch);
  for (int i = 0; i < n_backward_plans; ++i) query(CUFFT_C2R, backward_batch);
  return work_max;
}

// Precise device-memory model for one benchmarked run. Three parts:
//
//   (1) Standalone driver input buffers the harness cudaMalloc's and feeds in
//       (PSFs, dirty + its pristine master, jones_norm, ...). These live outside
//       the resources pool but still draw from device free memory.
//   (2) The two persistent cuFFT work areas (scale- and PSF-domain contexts),
//       queried directly from cuFFT for exactness -- the single biggest accuracy
//       gain over a closed-form guess.
//   (3) The peak working set of run_ddmsc_cycles inside the pool. Buffer sizes
//       are exact element counts mirroring the alloc_async calls in ddmsc.cu /
//       scales.cu; we take the max across the two heavy, non-overlapping phases
//       (PSF-domain precompute vs. per-iteration scale-domain convolve).
//
// The only non-modeled terms are tiny CUB reduction scratch buffers (a few KB,
// covered by a flat allowance) and fixed CUDA/cuBLAS/cuFFT context overhead
// (input-independent: a roughly constant per-process baseline that free_b at the
// pre-check already excludes). The trailing factor absorbs pool sub-allocation
// rounding only.
static std::size_t estimate_bytes(const bench_config& c)
{
  constexpr std::size_t F = sizeof(float);          // real element
  constexpr std::size_t CX = sizeof(cufftComplex);  // half-complex element
  constexpr std::size_t kCubTemp = 2048;            // flat per-reduction scratch + tiny outputs
  constexpr float fft_padding = 1.1f;               // must match run_config

  const ddmsc::params p = make_fixed_params(c);  // the exact params this config runs with

  const std::size_t nf = c.n_freq, ns = c.n_scales, nfac = c.n_facet, no = c.n_order;
  const std::size_t npix = static_cast<std::size_t>(c.nrow) * c.ncol;
  const std::size_t psf_npix = static_cast<std::size_t>(c.psf_nrow) * c.psf_ncol;

  // Padded / half-complex grids for both FFT contexts.
  const auto [img_pr, img_pc] = padded_spatial(c.nrow, c.ncol, fft_padding);
  const auto [psf_pr, psf_pc] = padded_spatial(c.psf_nrow, c.psf_ncol, fft_padding);
  const std::size_t img_pad = static_cast<std::size_t>(img_pr) * img_pc;
  const std::size_t img_freq = static_cast<std::size_t>(img_pr) * (img_pc / 2 + 1);
  const std::size_t psf_pad = static_cast<std::size_t>(psf_pr) * psf_pc;
  const std::size_t psf_freq = static_cast<std::size_t>(psf_pr) * (psf_pc / 2 + 1);

  // (1) Driver input buffers. The static inputs (psfs/xdes/sigmas/mask) are
  //   staged into the context pool by its constructor; the per-run
  //   dirty/jones/weights remain standalone cudaMalloc. Both are live device
  //   memory, so the total footprint is unchanged.
  std::size_t input_buffers = 0;
  input_buffers += nfac * nf * psf_npix * F;  // raw_psfs (context-staged)
  input_buffers += nf * no * F;               // xdes (context-staged)
  input_buffers += ns * F;                    // scale_sigmas (context-staged)
  input_buffers += npix * sizeof(bool);       // scale_mask (context-staged)
  input_buffers += nf * npix * F;             // d_jones
  input_buffers += nf * F;                    // d_weights
  input_buffers += 2 * nf * npix * F;         // d_dirty + d_dirty_master

  // (2) Persistent cuFFT work area: one buffer shared by both FFT contexts
  //   (their plans run sequentially on the same stream), sized to the max.
  //   scale ctx: R2C batch 1, C2R batch ns-1.  psf ctx: R2C/C2R batch nf (two C2R plans).
  const std::size_t work_areas = std::max(cufft_work_bytes(img_pr, img_pc, 1, static_cast<int>(ns) - 1, 1),
                                          cufft_work_bytes(psf_pr, psf_pc, c.n_freq, c.n_freq, 2));

  // (3) run_ddmsc_cycles pool working set.
  // Always-live for the whole call (allocated before PSF precompute, freed last):
  const std::size_t coeffs_cap = static_cast<std::size_t>(p.max_iteration + p.max_clean_iteration) * no;
  std::size_t live = 0;
  live += npix * F;                       // mean_residual
  live += coeffs_cap * F;                 // d_all_coeffs
  live += ns * nfac * psf_npix * F;       // conv2_psfs
  live += ns * nfac * nf * psf_npix * F;  // conv_psfs
  live += 2 * kCubTemp;                   // stats_workspace + argmax (peak) workspace

  // Allocated only after the PSF precompute, then live for the rest of the call:
  std::size_t scale_loop = 0;
  scale_loop += ns * img_freq * F;  // scale_kernels (freq domain)
  scale_loop += ns * npix * F;      // scales_x_dirty
  if (p.enable_auto_mask || p.force_enable_auto_mask)
    scale_loop += ns * npix * sizeof(bool);  // mask_per_scale (lazily allocated)

  // Phase A -- PSF-domain precompute transients (convolve_psfs_with_scale_async):
  //   padded_psf + padded_conv + padded_conv2 (3 x), freq_psf/conv/conv2 (3 x),
  //   conv2_cropped, and the per-scale kernel.
  const std::size_t phase_psf = 3 * nf * psf_pad * F + 3 * nf * psf_freq * CX + nf * psf_npix * F + psf_freq * F;

  // Phase B -- scale-domain convolve transients (convolve_with_scales):
  //   dirty_padded + scaled_dirty (ns x img_pad) and dirty_freq + scaled_dirty_freq (ns x img_freq).
  const std::size_t phase_scale = ns * img_pad * F + ns * img_freq * CX;

  // Peak pool usage: work areas + always-live + the heavier of the two phases
  // (phase A runs before the scale_loop buffers exist; phase B runs after).
  const std::size_t pool_peak = work_areas + live + std::max(phase_psf, scale_loop + phase_scale);

  const std::size_t total = input_buffers + pool_peak;
  return static_cast<std::size_t>(total * 1.03);  // pool sub-allocation rounding
}

template <typename T>
static T* device_upload(const T* host, std::size_t count)
{
  T* d = nullptr;
  BENCH_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d), count * sizeof(T)));
  BENCH_CHECK_CUDA(cudaMemcpy(d, host, count * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

// ------------------------------------------------------------------------- //
//  Host-data cache (reused across configs)
// ------------------------------------------------------------------------- //

// Synthetic inputs are content-independent for timing, so identical-dimension
// configs can share a single host buffer instead of regenerating the per-pixel
// data every time. Keyed by exactly the dims each generator depends on; values
// live for the whole sweep -- trading host RAM for far less CPU generation
// (high payoff in OFAT mode, where most configs share these dims).
struct host_data_cache {
  std::map<std::tuple<int, int, int, int>, std::vector<float>> psfs;  // (n_facet, n_freq, psf_nrow, psf_ncol)
  std::map<std::tuple<int, int, int>, std::vector<int>> map_facet;    // (nrow, ncol, n_facet)
  std::map<std::tuple<int, int, int>, std::vector<float>> dirty;      // (n_freq, nrow, ncol)
  std::map<std::tuple<int, int, int>, std::vector<float>> jones;      // (n_freq, nrow, ncol), constant 1.0
  std::map<std::tuple<int, int>, std::vector<unsigned char>> mask;    // (nrow, ncol), constant 0
};

template <typename Map, typename Key, typename Gen>
static const typename Map::mapped_type& cache_get(Map& cache, const Key& key, Gen gen)
{
  auto it = cache.find(key);
  if (it == cache.end()) it = cache.emplace(key, gen()).first;
  return it->second;
}

// ------------------------------------------------------------------------- //
//  Per-config benchmark
// ------------------------------------------------------------------------- //

struct config_result {
  bench_config cfg;
  run_stats stats{};
  int total_iters = 0;
  std::size_t n_components = 0;
  double used_mem_mb = 0;     // actual peak device memory in use during the run
  std::string status = "ok";  // ok | iter_mismatch | skipped_oom
};

static config_result run_config(const bench_config& c, const bench_options& opt, host_data_cache& cache)
{
  config_result out;
  out.cfg = c;

  // OOM pre-check against current free memory. The estimate already carries a
  // small rounding allowance (see estimate_bytes), so skip only when it
  // genuinely exceeds free memory.
  size_t free_b = 0, total_b = 0;
  BENCH_CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
  if (estimate_bytes(c) > free_b) {
    out.status = "skipped_oom";
    return out;
  }

  const std::size_t npix = static_cast<std::size_t>(c.nrow) * c.ncol;

  // ----- Generate synthetic inputs (host) -----
  // Tiny per-config buffers: a handful of elements each (O(n_scales)/O(n_freq)),
  // so a cache node would cost more than regenerating them.
  const std::vector<double> alphas = gen_alphas(c.n_scales);
  const std::vector<float> h_sigmas = gen_scale_sigmas(alphas);
  const std::vector<float> h_bias = gen_scale_bias(alphas);
  const std::vector<float> h_xdes = gen_xdes(c.n_freq, c.n_order);
  std::vector<float> h_weights(c.n_freq, 1.0f / c.n_freq);

  // Large / per-pixel buffers: fetched from the cross-config cache, keyed by the
  // dims each depends on. The dirty seed is derived from those dims so identical
  // configs get identical content regardless of generation order.
  const uint64_t dirty_seed = opt.seed ^ (static_cast<uint64_t>(c.n_freq) * 0x9E3779B97F4A7C15ull +
                                          static_cast<uint64_t>(c.nrow) * 0xC2B2AE3D27D4EB4Full +
                                          static_cast<uint64_t>(c.ncol) * 0x165667B19E3779F9ull);
  const std::vector<float>& h_psfs = cache_get(cache.psfs, std::make_tuple(c.n_facet, c.n_freq, c.psf_nrow, c.psf_ncol),
                                               [&] { return gen_psfs(c.n_facet, c.n_freq, c.psf_nrow, c.psf_ncol); });
  const std::vector<int>& h_map = cache_get(cache.map_facet, std::make_tuple(c.nrow, c.ncol, c.n_facet),
                                            [&] { return gen_map_pixel_facet(c.nrow, c.ncol, c.n_facet); });
  const std::vector<float>& h_dirty = cache_get(cache.dirty, std::make_tuple(c.n_freq, c.nrow, c.ncol),
                                                [&] { return gen_dirty(c.n_freq, c.nrow, c.ncol, dirty_seed); });
  const std::vector<float>& h_jones = cache_get(cache.jones, std::make_tuple(c.n_freq, c.nrow, c.ncol), [&] {
    return std::vector<float>(static_cast<std::size_t>(c.n_freq) * npix, 1.0f);
  });
  const std::vector<unsigned char>& h_mask =
      cache_get(cache.mask, std::make_tuple(c.nrow, c.ncol), [&] { return std::vector<unsigned char>(npix, 0); });

  // ----- Upload the per-run device buffers (once); the context stages its
  // static inputs (psfs/xdes/sigmas/mask) from the host itself. -----
  float* d_jones = device_upload(h_jones.data(), h_jones.size());
  float* d_weights = device_upload(h_weights.data(), h_weights.size());
  float* d_dirty_master = device_upload(h_dirty.data(), h_dirty.size());  // pristine
  float* d_dirty = nullptr;
  BENCH_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_dirty), h_dirty.size() * sizeof(float)));

  // ----- Views (host arrays kept alive for span validity) -----
  core::host_span4d<float> raw_psfs(const_cast<float*>(h_psfs.data()), c.n_facet, c.n_freq, c.psf_nrow, c.psf_ncol);
  core::host_span2d<float> xdes(const_cast<float*>(h_xdes.data()), c.n_freq, c.n_order);
  core::host_span2d<bool> mask(reinterpret_cast<bool*>(const_cast<unsigned char*>(h_mask.data())), c.nrow, c.ncol);
  core::host_vect<float> scale_sigmas(const_cast<float*>(h_sigmas.data()), c.n_scales);
  core::host_vect<float> scale_bias(const_cast<float*>(h_bias.data()), c.n_scales);
  core::host_span2d<int> map_pixel_facet(const_cast<int*>(h_map.data()), c.nrow, c.ncol);
  core::device_span3d<float> dirty(d_dirty, c.n_freq, c.nrow, c.ncol);
  core::device_span3d<float> jones_norm(d_jones, c.n_freq, c.nrow, c.ncol);
  core::device_vect<float> weights_freq(d_weights, c.n_freq);

  const ddmsc::params params = make_fixed_params(c);
  const float fft_padding = 1.1f;

  cudaEvent_t ev_start, ev_stop;
  BENCH_CHECK_CUDA(cudaEventCreate(&ev_start));
  BENCH_CHECK_CUDA(cudaEventCreate(&ev_stop));

  const std::size_t dirty_bytes = h_dirty.size() * sizeof(float);
  std::vector<double> times;
  times.reserve(opt.runs);
  int last_iters = -1;
  std::size_t last_components = 0;
  double peak_used_mb = 0.0;

  // One context per config, reused across all repetitions: rebuilding it each
  // run would re-pay the expensive cuFFT plan creation, and the build is
  // excluded from the timed region anyway.
  ddmsc::context ctx(opt.device, raw_psfs, xdes, mask, scale_sigmas, scale_bias, map_pixel_facet, c.nrow, c.ncol,
                     c.n_freq, fft_padding);
  ctx.state();  // build the device state here, not inside the first timed run

  const int total_runs = opt.warmup + opt.runs;
  for (int r = 0; r < total_runs; ++r) {
    // Reset dirty from the pristine master (run_ddmsc_cycles mutates it).
    BENCH_CHECK_CUDA(cudaMemcpy(d_dirty, d_dirty_master, dirty_bytes, cudaMemcpyDeviceToDevice));

    // Auto-mask history is the only state carried between calls -> clear it so
    // each repetition does identical, deterministic work.
    ctx.historical_peak_coords.clear();
    ctx.historical_scales.clear();

    BENCH_CHECK_CUDA(cudaDeviceSynchronize());  // exclude any pending work from timing
    BENCH_CHECK_CUDA(cudaEventRecord(ev_start));
    ddmsc::ddmsc_result res = ddmsc::run_ddmsc_cycles(ctx, params, dirty, jones_norm, weights_freq);
    BENCH_CHECK_CUDA(cudaEventRecord(ev_stop));
    BENCH_CHECK_CUDA(cudaEventSynchronize(ev_stop));

    // Actual device memory in use now that the full K*M run is done but the
    // context still holds its allocations -- the resources pool only releases to
    // the OS once ctx is destroyed at the end of this config. Keep the
    // high-water mark across runs.
    {
      size_t mem_free = 0, mem_total = 0;
      BENCH_CHECK_CUDA(cudaMemGetInfo(&mem_free, &mem_total));
      peak_used_mb = std::max(peak_used_mb, (mem_total - mem_free) / (1024.0 * 1024.0));
    }

    if (r >= opt.warmup) {
      float ms = 0.0f;
      BENCH_CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      times.push_back(ms);
      if (last_iters >= 0 && res.total_iterations != last_iters) out.status = "iter_mismatch";
      last_iters = res.total_iterations;
      last_components = res.peak_coords.size();
    }
  }

  out.stats = summarize(times);
  out.total_iters = last_iters;
  out.n_components = last_components;
  out.used_mem_mb = peak_used_mb;
  if (last_iters != c.K * c.M && out.status == "ok") out.status = "iter_mismatch";

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);
  cudaFree(d_dirty);
  cudaFree(d_dirty_master);
  cudaFree(d_weights);
  cudaFree(d_jones);

  return out;
}

// ------------------------------------------------------------------------- //
//  Derived metrics + output
// ------------------------------------------------------------------------- //

static double iters_per_sec(const config_result& r)
{
  return r.total_iters > 0 ? r.total_iters / (r.stats.mean_ms / 1000.0) : 0.0;
}
static double ms_per_iter(const config_result& r) { return r.total_iters > 0 ? r.stats.mean_ms / r.total_iters : 0.0; }
static double mpix_iter_per_sec(const config_result& r)
{
  const double pix = static_cast<double>(r.cfg.nrow) * r.cfg.ncol * r.total_iters;
  return r.stats.mean_ms > 0 ? pix / (r.stats.mean_ms / 1000.0) / 1e6 : 0.0;
}

static const char* CSV_HEADER =
    "nrow,ncol,n_freq,n_scales,n_facet,n_order,psf_nrow,psf_ncol,K,M,"
    "total_iters,n_components,runs,warmup,mean_ms,std_ms,min_ms,max_ms,median_ms,"
    "iters_per_s,ms_per_iter,mpix_iter_per_s,used_mem_mb,status\n";

static void write_csv_row(std::ostream& os, const config_result& r, const bench_options& opt)
{
  const bench_config& c = r.cfg;
  os << c.nrow << ',' << c.ncol << ',' << c.n_freq << ',' << c.n_scales << ',' << c.n_facet << ',' << c.n_order << ','
     << c.psf_nrow << ',' << c.psf_ncol << ',' << c.K << ',' << c.M << ',' << r.total_iters << ',' << r.n_components
     << ',' << opt.runs << ',' << opt.warmup << ',' << r.stats.mean_ms << ',' << r.stats.std_ms << ',' << r.stats.min_ms
     << ',' << r.stats.max_ms << ',' << r.stats.median_ms << ',' << iters_per_sec(r) << ',' << ms_per_iter(r) << ','
     << mpix_iter_per_sec(r) << ',' << r.used_mem_mb << ',' << r.status << '\n';
}

// ------------------------------------------------------------------------- //
//  GPU info banner
// ------------------------------------------------------------------------- //

static void print_gpu_info(int device)
{
  cudaDeviceProp p{};
  BENCH_CHECK_CUDA(cudaGetDeviceProperties(&p, device));
  size_t free_b = 0, total_b = 0;
  BENCH_CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
  int driver_v = 0, runtime_v = 0;
  cudaDriverGetVersion(&driver_v);
  cudaRuntimeGetVersion(&runtime_v);
  const double gib = 1024.0 * 1024.0 * 1024.0;

  // clockRate / memoryClockRate were removed from cudaDeviceProp in CUDA 13;
  // query them via cudaDeviceGetAttribute, which works on both 12 and 13.
  int core_khz = 0, mem_khz = 0;
  cudaDeviceGetAttribute(&core_khz, cudaDevAttrClockRate, device);
  cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, device);

  printf("==================== GPU ====================\n");
  printf("  device %d        : %s (sm_%d%d)\n", device, p.name, p.major, p.minor);
  printf("  multiprocessors : %d SMs\n", p.multiProcessorCount);
  printf("  clocks          : core %.0f MHz  mem %.0f MHz (%d-bit bus)\n", core_khz / 1000.0, mem_khz / 1000.0,
         p.memoryBusWidth);
  printf("  global memory   : %.2f GiB total, %.2f GiB free\n", p.totalGlobalMem / gib, free_b / gib);
  printf("  L2 cache        : %.2f MiB\n", p.l2CacheSize / (1024.0 * 1024.0));
  printf("  CUDA driver     : %d.%d   runtime: %d.%d\n", driver_v / 1000, (driver_v % 1000) / 10, runtime_v / 1000,
         (runtime_v % 1000) / 10);
  printf("=============================================\n\n");
}

// ------------------------------------------------------------------------- //
//  Main
// ------------------------------------------------------------------------- //

int main(int argc, char** argv)
{
  bench_options opt;
  auto usage = [&]() {
    fprintf(stderr,
            "Usage: %s [--mode=cartesian|ofat] [--sizes=L] [--nfreq=L] [--nscales=L]\n"
            "          [--nfacet=L] [--norder=L] [--psf-frac=L] [--K=L] [--M=L] [--runs=N]\n"
            "          [--warmup=N] [--device=N] [--seed=N] [--csv=PATH] [--max-configs=N]\n"
            "          [--force] [--dry-run]\n"
            "  L = comma-separated list. --mode=cartesian (default) sweeps the full product\n"
            "  of all axes; --mode=ofat varies one axis at a time around each axis' median,\n"
            "  crossed with the full --sizes sweep.\n",
            argv[0]);
  };

  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto val = [&](const char* pfx) { return a.substr(std::strlen(pfx)); };
    if (a.rfind("--sizes=", 0) == 0)
      opt.sizes = parse_int_list(val("--sizes="));
    else if (a.rfind("--nfreq=", 0) == 0)
      opt.nfreqs = parse_int_list(val("--nfreq="));
    else if (a.rfind("--nscales=", 0) == 0)
      opt.nscales = parse_int_list(val("--nscales="));
    else if (a.rfind("--nfacet=", 0) == 0)
      opt.nfacets = parse_int_list(val("--nfacet="));
    else if (a.rfind("--norder=", 0) == 0)
      opt.norders = parse_int_list(val("--norder="));
    else if (a.rfind("--psf-frac=", 0) == 0)
      opt.psf_fracs = parse_double_list(val("--psf-frac="));
    else if (a.rfind("--K=", 0) == 0)
      opt.Ks = parse_int_list(val("--K="));
    else if (a.rfind("--M=", 0) == 0)
      opt.Ms = parse_int_list(val("--M="));
    else if (a.rfind("--runs=", 0) == 0)
      opt.runs = std::atoi(val("--runs=").c_str());
    else if (a.rfind("--warmup=", 0) == 0)
      opt.warmup = std::atoi(val("--warmup=").c_str());
    else if (a.rfind("--device=", 0) == 0)
      opt.device = std::atoi(val("--device=").c_str());
    else if (a.rfind("--seed=", 0) == 0)
      opt.seed = std::strtoull(val("--seed=").c_str(), nullptr, 10);
    else if (a.rfind("--csv=", 0) == 0)
      opt.csv_path = val("--csv=");
    else if (a.rfind("--max-configs=", 0) == 0)
      opt.max_configs = std::atoi(val("--max-configs=").c_str());
    else if (a.rfind("--mode=", 0) == 0)
      opt.mode = val("--mode=");
    else if (a == "--force")
      opt.force = true;
    else if (a == "--dry-run")
      opt.dry_run = true;
    else {
      fprintf(stderr, "Unrecognized argument: %s\n", a.c_str());
      usage();
      return 1;
    }
  }

  if (opt.runs < 1) opt.runs = 1;
  if (opt.warmup < 0) opt.warmup = 0;
  if (opt.mode != "cartesian" && opt.mode != "ofat") {
    fprintf(stderr, "Unknown --mode=%s (expected cartesian or ofat)\n", opt.mode.c_str());
    usage();
    return 1;
  }

  // Build the config list. A single builder applies the n_scales>=2 guard and
  // derives the PSF side; `seen` dedups (the OFAT cross revisits the baseline on
  // every axis, and either mode tolerates duplicate values in a --flag list).
  std::vector<bench_config> configs;
  std::set<std::tuple<int, int, int, int, int, double, int, int>> seen;
  auto add_config = [&](int sz, int nf, int ns, int nfac, int no, double pf, int K, int M) {
    if (!seen.insert(std::make_tuple(sz, nf, ns, nfac, no, pf, K, M)).second) return;
    if (ns < 2) {
      fprintf(stderr, "Skipping n_scales=%d (<2 unsupported by scale FFT batching)\n", ns);
      return;
    }
    bench_config c{};
    c.nrow = c.ncol = sz;
    c.n_freq = nf;
    c.n_scales = ns;
    c.n_facet = nfac;
    c.n_order = no;
    c.psf_nrow = c.psf_ncol = std::max(8, static_cast<int>(std::lround(pf * sz)));
    c.K = K;
    c.M = M;
    configs.push_back(c);
  };

  if (opt.mode == "ofat") {
    // Hold every axis at its median, then vary one axis at a time across its
    // list -- each sweep crossed with the full --sizes range (the plotter puts
    // image size on the x-axis, so each marginal line needs the whole sweep).
    const int b_nf = 2;      // pinned to match MEDIAN_OVERRIDE n_freq=2 in plot_bench.py
    const int b_ns = 5;      // pinned to match MEDIAN_OVERRIDE n_scales=5 in plot_bench.py
    const int b_nfac = 100;  // pinned to match MEDIAN_OVERRIDE n_facet=100 in plot_bench.py
    const int b_no = median_value(opt.norders);
    const double b_pf = median_value(opt.psf_fracs);
    const int b_K = median_value(opt.Ks);
    const int b_M = median_value(opt.Ms);
    for (int sz : opt.sizes) {
      add_config(sz, b_nf, b_ns, b_nfac, b_no, b_pf, b_K, b_M);  // baseline curve
      for (int nf : opt.nfreqs) add_config(sz, nf, b_ns, b_nfac, b_no, b_pf, b_K, b_M);
      for (int ns : opt.nscales) add_config(sz, b_nf, ns, b_nfac, b_no, b_pf, b_K, b_M);
      for (int nfac : opt.nfacets) add_config(sz, b_nf, b_ns, nfac, b_no, b_pf, b_K, b_M);
      for (int no : opt.norders) add_config(sz, b_nf, b_ns, b_nfac, no, b_pf, b_K, b_M);
      for (double pf : opt.psf_fracs) add_config(sz, b_nf, b_ns, b_nfac, b_no, pf, b_K, b_M);
      for (int K : opt.Ks) add_config(sz, b_nf, b_ns, b_nfac, b_no, b_pf, K, b_M);
      for (int M : opt.Ms) add_config(sz, b_nf, b_ns, b_nfac, b_no, b_pf, b_K, M);
    }
  } else {
    for (int sz : opt.sizes)
      for (int nf : opt.nfreqs)
        for (int ns : opt.nscales)
          for (int nfac : opt.nfacets)
            for (int no : opt.norders)
              for (double pf : opt.psf_fracs)
                for (int K : opt.Ks)
                  for (int M : opt.Ms) add_config(sz, nf, ns, nfac, no, pf, K, M);
  }

  printf("Mode: %s | Total configs: %zu  (runs=%d, warmup=%d per config)\n", opt.mode.c_str(), configs.size(), opt.runs,
         opt.warmup);

  BENCH_CHECK_CUDA(cudaSetDevice(opt.device));
  print_gpu_info(opt.device);  // show device + free memory to compare against est_mem_MB

  if (opt.dry_run) {
    printf("%-6s %-6s %-6s %-7s %-7s %-7s %-10s %-4s %-4s %-10s\n", "nrow", "ncol", "nfreq", "nscales", "nfacet",
           "norder", "psf", "K", "M", "est_mem_MB");
    for (const auto& c : configs)
      printf("%-6d %-6d %-6d %-7d %-7d %-7d %dx%-7d %-4d %-4d %-10.1f\n", c.nrow, c.ncol, c.n_freq, c.n_scales,
             c.n_facet, c.n_order, c.psf_nrow, c.psf_ncol, c.K, c.M, estimate_bytes(c) / (1024.0 * 1024.0));
    return 0;
  }

  if (static_cast<int>(configs.size()) > opt.max_configs && !opt.force) {
    fprintf(stderr,
            "Refusing to run %zu configs (> --max-configs=%d). Re-run with --force or narrow the "
            "sweep, or use --dry-run to inspect.\n",
            configs.size(), opt.max_configs);
    return 1;
  }

  fast_deconv::log::set_level(spdlog::level::warn);  // silence per-run INFO banners

  std::ofstream csv;
  if (!opt.csv_path.empty()) {
    csv.open(opt.csv_path);
    if (!csv) {
      fprintf(stderr, "Failed to open %s for writing\n", opt.csv_path.c_str());
      return 1;
    }
    csv << CSV_HEADER;
  }

  host_data_cache cache;  // reused across configs: trades host RAM for fewer regenerations
  printf("%-5s %-5s %-3s %-3s %-4s %-3s %-9s %-4s %-4s | %10s %9s %12s %8s  %s\n", "nrow", "ncol", "nf", "ns", "nfac",
         "no", "psf", "K", "M", "mean_ms", "std_ms", "iters/s", "ms/it", "status");

  for (std::size_t i = 0; i < configs.size(); ++i) {
    const config_result r = run_config(configs[i], opt, cache);
    printf("%-5d %-5d %-3d %-3d %-4d %-3d %4dx%-4d %-4d %-4d | %10.3f %9.3f %12.1f %8.4f  %s\n", r.cfg.nrow, r.cfg.ncol,
           r.cfg.n_freq, r.cfg.n_scales, r.cfg.n_facet, r.cfg.n_order, r.cfg.psf_nrow, r.cfg.psf_ncol, r.cfg.K, r.cfg.M,
           r.stats.mean_ms, r.stats.std_ms, iters_per_sec(r), ms_per_iter(r), r.status.c_str());
    fflush(stdout);
    if (csv.is_open()) {
      write_csv_row(csv, r, opt);
      csv.flush();
    }
  }

  if (csv.is_open()) printf("\nWrote %s\n", opt.csv_path.c_str());
  return 0;
}
