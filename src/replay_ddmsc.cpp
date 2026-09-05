/// Replay a DDMSC session captured from DDFacet, without launching DDFacet.
///
/// Usage:  replay_ddmsc <dump_dir> [--device=N] [--cycle=N] [--runs=N]
///                      [--psf-cache=lazy_pair|lazy_scale|eager_all]
///                      [--psf-cache-budget=<bytes, K/M/G suffix ok, 0=unbounded>]
///
/// --runs=N replays the whole dump N times on a fresh session each time and
/// reports min/median/mean/max of the per-run total; the first cycle of a run
/// carries the one-off device setup (uploads, cuFFT plans), the same for every run.
///
/// The cache flags override the fast-deconv-side knobs, which the dump does not
/// carry (DDFacet never sets them); everything else comes from the dump.
///
/// The dump is produced by running DDFacet with FAST_DECONV_DUMP=<dir> (see
/// fast_deconv/__init__.py): `init/` holds the ctor inputs, `cycle_<N>/` holds
/// one major cycle's run() inputs plus every algorithm parameter as it stood
/// for that call. Cycles replay in order on a single Ddmsc, so the auto-mask
/// history accumulates exactly as it does in the real run.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fast_deconv/algorithm/ddmsc.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <filesystem>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "npy_loader.hpp"

namespace core = fast_deconv::core;
namespace algo = fast_deconv::algorithm;
namespace ddmsc = fast_deconv::algorithm::ddmsc;

/// Same order as psf_cache_mode.
static constexpr std::array<const char*, 3> cache_mode_names{"lazy_pair", "lazy_scale", "eager_all"};

/// Byte count with an optional K/M/G (binary) suffix.
static std::size_t parse_bytes(const std::string& s)
{
  char* end = nullptr;
  double v = std::strtod(s.c_str(), &end);
  switch (*end) {
    case 'G':
    case 'g':
      v *= 1024;
      [[fallthrough]];
    case 'M':
    case 'm':
      v *= 1024;
      [[fallthrough]];
    case 'K':
    case 'k':
      v *= 1024;
      break;
    default:
      break;
  }
  return static_cast<std::size_t>(v);
}

/// A NaN scalar in the dump is an unset optional on the Python side.
static std::optional<float> opt_finite(float v) { return std::isnan(v) ? std::nullopt : std::optional<float>(v); }

int main(int argc, char** argv)
{
  std::string dir;
  int device_id = 0;
  int only_cycle = -1;
  std::optional<algo::psf_cache_mode> cache_policy;
  std::optional<std::size_t> cache_budget;
  int runs = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    if (a.rfind("--device=", 0) == 0) {
      device_id = std::atoi(a.c_str() + 9);
    } else if (a.rfind("--cycle=", 0) == 0) {
      only_cycle = std::atoi(a.c_str() + 8);
    } else if (a.rfind("--runs=", 0) == 0) {
      runs = std::max(1, std::atoi(a.c_str() + 7));
    } else if (a.rfind("--psf-cache=", 0) == 0) {
      const auto* it = std::find_if(cache_mode_names.begin(), cache_mode_names.end(),
                                    [&](const char* n) { return a.compare(12, std::string::npos, n) == 0; });
      if (it == cache_mode_names.end()) {
        fprintf(stderr, "Unknown --psf-cache mode: %s\n", a.c_str() + 12);
        return 1;
      }
      cache_policy = static_cast<algo::psf_cache_mode>(it - cache_mode_names.begin());
    } else if (a.rfind("--psf-cache-budget=", 0) == 0) {
      cache_budget = parse_bytes(a.substr(19));
    } else if (!a.empty() && a[0] != '-' && dir.empty()) {
      dir = a;
    } else {
      fprintf(stderr, "Unrecognized argument: %s\n", argv[i]);
      dir.clear();
      break;
    }
  }
  if (dir.empty()) {
    fprintf(stderr,
            "Usage: %s <dump_dir> [--device=N] [--cycle=N] [--runs=N] [--psf-cache=%s|%s|%s] "
            "[--psf-cache-budget=SIZE]\n",
            argv[0], cache_mode_names[0], cache_mode_names[1], cache_mode_names[2]);
    return 1;
  }

  auto load_init = [&](const char* name) { return npy::load_npy(dir + "/init/" + name + ".npy"); };

  // Held for the whole run: the ctor keeps host views on these until run()
  // stages them to the device.
  auto npy_raw_psfs = load_init("raw_psfs");
  auto npy_xdes = load_init("xdes");
  auto npy_mask = load_init("scale_mask");
  auto npy_scale_sigmas = load_init("scale_sigmas");
  auto npy_scale_bias = load_init("scale_bias");
  auto npy_map_pixel = load_init("map_pixel_facet");
  const float fft_padding = load_init("fft_padding").scalar<float>();

  const int n_facet = static_cast<int>(npy_raw_psfs.shape[0]);
  const int n_freq = static_cast<int>(npy_raw_psfs.shape[1]);
  const int n_scales = static_cast<int>(npy_scale_sigmas.shape[0]);
  const int nrow = static_cast<int>(npy_map_pixel.shape[0]);
  const int ncol = static_cast<int>(npy_map_pixel.shape[1]);

  core::host_span4d<float> raw_psfs(npy_raw_psfs.as_float32(), n_facet, n_freq, npy_raw_psfs.shape[2],
                                    npy_raw_psfs.shape[3]);
  core::host_span2d<float> xdes(npy_xdes.as_float32(), n_freq, npy_xdes.shape[1]);
  core::host_span2d<bool> mask(npy_mask.as_bool(), npy_mask.shape[0], npy_mask.shape[1]);
  core::host_span1d<float> scale_sigmas(npy_scale_sigmas.as_float32(), n_scales);
  core::host_span1d<float> scale_bias(npy_scale_bias.as_float32(), n_scales);
  core::host_span2d<int> map_pixel_facet(npy_map_pixel.as_int32(), nrow, ncol);

  printf("Replaying %s (device=%d, runs=%d)\n", dir.c_str(), device_id, runs);
  printf("  image: %dx%d  psf: %zux%zu  freq: %d  facets: %d  scales: %d\n", nrow, ncol, npy_raw_psfs.shape[2],
         npy_raw_psfs.shape[3], n_freq, n_facet, n_scales);

  const int first_cycle = only_cycle < 0 ? 0 : only_cycle;
  std::vector<double> run_ms;
  for (int run = 0; run < runs; ++run) {
    // A fresh session per run: the auto-mask history accumulates across cycles, so
    // a second replay on the same Ddmsc would not do the same work.
    ddmsc::Ddmsc imager(raw_psfs, xdes, mask, scale_sigmas, scale_bias, map_pixel_facet, nrow, ncol, n_freq,
                        fft_padding, device_id);

    if (cache_policy) imager.set_psf_cache_policy(*cache_policy);
    if (cache_budget) imager.set_psf_cache_budget_bytes(*cache_budget);
    if (run == 0)
      printf("  psf cache: %s, budget %zu bytes\n", cache_mode_names.at(static_cast<int>(imager.psf_cache_policy())),
             imager.psf_cache_budget_bytes());

    double total_ms = 0.0;
    for (int cycle = first_cycle;; ++cycle) {
      const std::string cdir = dir + "/cycle_" + std::to_string(cycle) + "/";
      if (!std::filesystem::exists(cdir)) {
        if (cycle == first_cycle) {
          fprintf(stderr, "No cycle_%d/ in %s\n", cycle, dir.c_str());
          return 1;
        }
        break;
      }
      auto load = [&](const char* name) { return npy::load_npy(cdir + name + ".npy"); };

      auto npy_dirty = load("dirty");
      auto npy_jones_norm = load("jones_norm");
      auto npy_weights = load("weights_freq");

      imager.set_clean_negative(load("clean_negative").scalar<bool>());
      imager.set_peak_factor(load("peak_factor").scalar<float>());
      imager.set_gamma(load("gamma").scalar<float>());
      imager.set_max_sub_iteration(load("max_sub_iteration").scalar<int>());
      imager.set_flux_threshold(load("flux_threshold").scalar<float>());
      imager.set_stop_rms_factor(load("stop_rms_factor").scalar<float>());
      imager.set_stop_peak_factor(load("stop_peak_factor").scalar<float>());
      imager.set_stop_cycle_factor(load("stop_cycle_factor").scalar<float>());
      imager.set_stop_sidelobe_level(load("stop_sidelobe_level").scalar<float>());
      imager.set_max_iteration(load("max_iteration").scalar<int>());
      imager.set_divergence_factor(load("divergence_factor").scalar<float>());
      imager.set_stall_threshold(load("stall_threshold").scalar<float>());
      imager.set_auto_mask(load("auto_mask").scalar<bool>());
      imager.set_force_auto_mask(load("force_auto_mask").scalar<bool>());
      imager.set_auto_mask_peak_threshold(opt_finite(load("auto_mask_peak_threshold").scalar<float>()));
      imager.set_auto_mask_rms_threshold(opt_finite(load("auto_mask_rms_threshold").scalar<float>()));

      core::host_span3d<float> dirty(npy_dirty.as_float32(), n_freq, nrow, ncol);
      core::host_span3d<const float> jones_norm(npy_jones_norm.as_float32(), n_freq, nrow, ncol);
      core::host_span1d<const float> weights_freq(npy_weights.as_float32(), n_freq);

      const auto t0 = std::chrono::steady_clock::now();
      const ddmsc::ddmsc_result result = imager.run(dirty, jones_norm, weights_freq);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      total_ms += ms;

      if (run == 0)
        printf("cycle %d: %.3f ms, %zu components, %d iterations, flux %.6g (stop %.6g)\n", cycle, ms,
               result.peak_coords.size(), result.total_iterations, result.final_flux, result.stop_flux);
      if (only_cycle >= 0) {
        break;
      }
    }
    printf("run %d: %.3f ms (%.3f s)\n", run, total_ms, total_ms / 1000.0);
    run_ms.push_back(total_ms);
  }

  if (runs > 1) {
    std::sort(run_ms.begin(), run_ms.end());
    const double mean = std::accumulate(run_ms.begin(), run_ms.end(), 0.0) / runs;
    const double median = runs % 2 ? run_ms.at(runs / 2) : 0.5 * (run_ms.at(runs / 2 - 1) + run_ms.at(runs / 2));
    printf("Over %d runs: min %.3f ms  median %.3f ms  mean %.3f ms  max %.3f ms\n", runs, run_ms.front(), median, mean,
           run_ms.back());
  }
  return 0;
}
