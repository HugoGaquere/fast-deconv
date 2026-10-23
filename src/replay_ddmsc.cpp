// Replay a DDMSC session captured from DDFacet, without launching DDFacet.
//
// Usage:  replay_ddmsc <dump_dir> [--device=N] [--cycle=N] [--runs=N]
//                      [--max-iter=N] [--max-clean-iter=N]
//                      [--peak-factor=F] [--stall-threshold=F] [--stop-rms-factor=F] [--stop-peak-factor=F]
//                      [--stop-sidelobe-level=F] [--divergence-factor=F] [--auto-mask=0|1] [--force-scale=S]
//                      [--flux-threshold=F] [--stop-cycle-factor=F] [--crop=N|NROWxNCOL]
//                      [--bench] [--outer=N] [--csv=FILE] [--label=STR]
//
// --runs=N replays the whole dump N times on a fresh session each time and
// reports min/median/mean/max of the per-run total; the first cycle of a run
// carries the one-off device setup (uploads, cuFFT plans), the same for every run.
//
// --max-iter (total minor iterations) and --max-clean-iter (clean loop length)
// override the dump's caps; --cycle=0 --max-iter=1 --max-clean-iter=1 runs every
// kernel of the cycle once, which is what an ncu kernel-replay profile wants.
//
// The other flags override the dump's value of the parameter they name.
// --force-scale=S sets the scale bias to 1 for S and 0 elsewhere, so S wins
// every scale selection; --bench below pins every other data-dependent decision.
//
// --crop=N (or NROWxNCOL) replays the centred N x N window of the image: the
// dirty, jones_norm, pixel-to-facet map and mask are cropped, the per-facet
// PSFs are kept whole. Meant for scaling tests over the image size.
//
// --bench times a fixed amount of work on one cycle (--cycle, default 0), so
// GPUs compare fairly. It defaults every flag above that pins a decision
// (scale 1, peak factor 0, every stop and stall off, no auto-mask; an explicit
// flag still wins) and runs --outer=N outer iterations
// (default 200) of --max-clean-iter minor iterations each. Each of the --runs
// repetitions also replays a 1-outer baseline on a fresh session, so
//   per_outer_ms = (total_ms - baseline_ms) / (N - 1)
// excludes the one-off setup (uploads, cuFFT plans, the first outer's PSF builds;
// facets first hit later are built inside the timed outers). An untimed
// 1-outer warm-up first absorbs the process's CUDA initialisation. --csv=FILE
// appends one row per repetition, tagged with --label and the host name.
//
// The dump is produced by running DDFacet with FAST_DECONV_DUMP=<dir> (see
// fast_deconv/__init__.py): `init/` holds the ctor inputs, `cycle_<N>/` holds
// one major cycle's run() inputs plus every algorithm parameter as it stood
// for that call. Cycles replay in order on a single Ddmsc, so the auto-mask
// history accumulates exactly as it does in the real run.

#include <unistd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fast_deconv/algorithm/ddmsc.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "npy_loader.hpp"

namespace core = fast_deconv::core;
namespace algo = fast_deconv::algorithm;
namespace ddmsc = fast_deconv::algorithm::ddmsc;

// A NaN scalar in the dump is an unset optional on the Python side.
static std::optional<float> opt_finite(float v) { return std::isnan(v) ? std::nullopt : std::optional<float>(v); }

// Crops the last two axes of a C-order array to their centred rows x cols window, in place.
// Every kept row moves to a lower offset, so no row still to be read is overwritten.
template <typename T>
static void crop_center(npy::NpyArray& arr, size_t rows, size_t cols)
{
  const size_t nd = arr.ndim();
  const size_t src_rows = arr.shape.at(nd - 2);
  const size_t src_cols = arr.shape.at(nd - 1);
  const size_t row0 = (src_rows - rows) / 2;
  const size_t col0 = (src_cols - cols) / 2;
  const size_t n_planes = arr.size() / (src_rows * src_cols);
  T* data = arr.as<T>();
  for (size_t p = 0; p < n_planes; ++p)
    for (size_t r = 0; r < rows; ++r)
      std::memmove(data + (p * rows + r) * cols, data + (p * src_rows + row0 + r) * src_cols + col0, cols * sizeof(T));
  arr.shape.at(nd - 2) = rows;
  arr.shape.at(nd - 1) = cols;
}

// Parses a[begin, end) as a T; exits on an empty, partial or out-of-range value.
template <typename T>
static T parse_arg(const std::string& a, std::size_t begin, std::size_t end = std::string::npos)
{
  const char* first = a.data() + begin;
  const char* last = a.data() + std::min(end, a.size());
  T v{};
  const auto [ptr, ec] = std::from_chars(first, last, v);
  if (ec != std::errc{} || ptr != last) {
    fprintf(stderr, "Invalid value in argument: %s\n", a.c_str());
    std::exit(1);
  }
  return v;
}

int main(int argc, char** argv)
{
  std::string dir;
  int device_id = 0;
  int only_cycle = -1;
  int runs = 1;
  std::optional<int> max_iter;
  std::optional<int> max_clean_iter;
  std::optional<float> peak_factor;
  std::optional<float> stall_threshold;
  std::optional<float> stop_rms_factor;
  std::optional<float> stop_peak_factor;
  std::optional<float> stop_sidelobe_level;
  std::optional<float> divergence_factor;
  std::optional<bool> auto_mask;
  std::optional<int> force_scale;
  std::optional<float> flux_threshold;
  std::optional<float> stop_cycle_factor;
  std::optional<std::array<int, 2>> crop;  // {rows, cols}
  bool bench = false;
  int outer = 200;
  std::string csv_path;
  std::string label;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    if (a.rfind("--device=", 0) == 0) {
      device_id = parse_arg<int>(a, 9);
    } else if (a.rfind("--cycle=", 0) == 0) {
      only_cycle = parse_arg<int>(a, 8);
    } else if (a.rfind("--runs=", 0) == 0) {
      runs = std::max(1, parse_arg<int>(a, 7));
    } else if (a.rfind("--max-iter=", 0) == 0) {
      max_iter = std::max(1, parse_arg<int>(a, 11));
    } else if (a.rfind("--max-clean-iter=", 0) == 0) {
      max_clean_iter = std::max(1, parse_arg<int>(a, 17));
    } else if (a.rfind("--peak-factor=", 0) == 0) {
      peak_factor = parse_arg<float>(a, 14);
    } else if (a.rfind("--stall-threshold=", 0) == 0) {
      stall_threshold = parse_arg<float>(a, 18);
    } else if (a.rfind("--stop-rms-factor=", 0) == 0) {
      stop_rms_factor = parse_arg<float>(a, 18);
    } else if (a.rfind("--stop-peak-factor=", 0) == 0) {
      stop_peak_factor = parse_arg<float>(a, 19);
    } else if (a.rfind("--stop-sidelobe-level=", 0) == 0) {
      stop_sidelobe_level = parse_arg<float>(a, 22);
    } else if (a.rfind("--divergence-factor=", 0) == 0) {
      divergence_factor = parse_arg<float>(a, 20);
    } else if (a.rfind("--auto-mask=", 0) == 0) {
      auto_mask = parse_arg<int>(a, 12) != 0;
    } else if (a.rfind("--force-scale=", 0) == 0) {
      force_scale = parse_arg<int>(a, 14);
    } else if (a.rfind("--crop=", 0) == 0) {
      const std::size_t x = a.find('x', 7);
      const int rows = parse_arg<int>(a, 7, x);
      crop = std::array<int, 2>{rows, x == std::string::npos ? rows : parse_arg<int>(a, x + 1)};
    } else if (a.rfind("--flux-threshold=", 0) == 0) {
      flux_threshold = parse_arg<float>(a, 17);
    } else if (a.rfind("--stop-cycle-factor=", 0) == 0) {
      stop_cycle_factor = parse_arg<float>(a, 20);
    } else if (a == "--bench") {
      bench = true;
    } else if (a.rfind("--outer=", 0) == 0) {
      outer = parse_arg<int>(a, 8);
    } else if (a.rfind("--csv=", 0) == 0) {
      csv_path = a.substr(6);
    } else if (a.rfind("--label=", 0) == 0) {
      label = a.substr(8);
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
            "Usage: %s <dump_dir> [--device=N] [--cycle=N] [--runs=N] [--max-iter=N] [--max-clean-iter=N]\n"
            "       [--peak-factor=F] [--stall-threshold=F] [--stop-rms-factor=F] [--stop-peak-factor=F]\n"
            "       [--stop-sidelobe-level=F] [--divergence-factor=F] [--auto-mask=0|1] [--force-scale=S]\n"
            "       [--flux-threshold=F] [--stop-cycle-factor=F] [--crop=N|NROWxNCOL]\n"
            "       [--bench] [--outer=N] [--csv=FILE] [--label=STR]\n",
            argv[0]);
    return 1;
  }
  if (!bench && !csv_path.empty()) {
    fprintf(stderr, "--csv needs --bench\n");
    return 1;
  }
  if (bench) {
    if (max_iter) {
      fprintf(stderr, "--bench caps the run with --outer; drop --max-iter\n");
      return 1;
    }
    if (outer < 2) {
      fprintf(stderr, "--outer=%d: the bench needs at least 2 outer iterations to subtract the baseline\n", outer);
      return 1;
    }
    // Pin every data-dependent decision, so two GPUs do the same work whatever their rounding.
    force_scale = force_scale.value_or(1);
    peak_factor = peak_factor.value_or(0.f);
    stall_threshold = stall_threshold.value_or(0.f);
    flux_threshold = flux_threshold.value_or(0.f);
    stop_rms_factor = stop_rms_factor.value_or(0.f);
    stop_peak_factor = stop_peak_factor.value_or(0.f);
    stop_cycle_factor = stop_cycle_factor.value_or(0.f);
    stop_sidelobe_level = stop_sidelobe_level.value_or(0.f);
    divergence_factor = divergence_factor.value_or(1e30f);
    auto_mask = auto_mask.value_or(false);
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

  const int n_facet = static_cast<int>(npy_raw_psfs.shape[0]);
  const int n_freq = static_cast<int>(npy_raw_psfs.shape[1]);
  const int n_scales = static_cast<int>(npy_scale_sigmas.shape[0]);
  const int full_nrow = static_cast<int>(npy_map_pixel.shape[0]);
  const int full_ncol = static_cast<int>(npy_map_pixel.shape[1]);

  if (crop) {
    const auto [rows, cols] = *crop;
    if (rows < 1 || cols < 1 || rows > full_nrow || cols > full_ncol) {
      fprintf(stderr, "--crop=%dx%d does not fit in the %dx%d image\n", rows, cols, full_nrow, full_ncol);
      return 1;
    }
    crop_center<std::int32_t>(npy_map_pixel, rows, cols);
    crop_center<bool>(npy_mask, rows, cols);
  }
  const int nrow = static_cast<int>(npy_map_pixel.shape[0]);
  const int ncol = static_cast<int>(npy_map_pixel.shape[1]);

  if (force_scale) {
    if (*force_scale < 0 || *force_scale >= n_scales) {
      fprintf(stderr, "--force-scale=%d is out of range [0, %d)\n", *force_scale, n_scales);
      return 1;
    }
    // Selection maximises peak * bias, so a zero bias takes every other scale out of the running.
    float* bias = npy_scale_bias.as_float32();
    std::fill(bias, bias + n_scales, 0.f);
    bias[*force_scale] = 1.f;
  }

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
  if (crop)
    printf("  crop: centred %dx%d window of %dx%d (origin %d,%d)\n", nrow, ncol, full_nrow, full_ncol,
           (full_nrow - nrow) / 2, (full_ncol - ncol) / 2);
  if (force_scale) printf("  forced scale: %d\n", *force_scale);

  const int first_cycle = only_cycle < 0 ? 0 : only_cycle;
  const auto cycle_dir = [&](int cycle) { return dir + "/cycle_" + std::to_string(cycle) + "/"; };

  // One cycle's run() inputs, cropped to the replayed window.
  struct cycle_inputs {
    npy::NpyArray dirty;
    npy::NpyArray jones_norm;
    npy::NpyArray weights;
  };
  const auto load_inputs = [&](const std::string& cdir) {
    cycle_inputs in{npy::load_npy(cdir + "dirty.npy"), npy::load_npy(cdir + "jones_norm.npy"),
                    npy::load_npy(cdir + "weights_freq.npy")};
    if (crop) {
      crop_center<float>(in.dirty, nrow, ncol);
      crop_center<float>(in.jones_norm, nrow, ncol);
    }
    return in;
  };

  // A fresh session: the auto-mask history accumulates across cycles, so a
  // second replay on the same Ddmsc would not do the same work.
  const auto make_imager = [&] {
    return std::make_unique<ddmsc::Ddmsc>(raw_psfs, xdes, mask, scale_sigmas, scale_bias, map_pixel_facet, nrow, ncol,
                                          device_id);
  };

  // The cycle's parameters as dumped, each overridden by its flag when given.
  const auto configure = [&](ddmsc::Ddmsc& imager, const std::string& cdir) {
    auto load = [&](const char* name) { return npy::load_npy(cdir + name + ".npy"); };
    imager.set_clean_negative(load("clean_negative").scalar<bool>());
    imager.set_peak_factor(peak_factor.value_or(load("peak_factor").scalar<float>()));
    imager.set_gamma(load("gamma").scalar<float>());
    imager.set_max_sub_iteration(max_clean_iter.value_or(load("max_sub_iteration").scalar<int>()));
    imager.set_flux_threshold(flux_threshold.value_or(load("flux_threshold").scalar<float>()));
    imager.set_stop_rms_factor(stop_rms_factor.value_or(load("stop_rms_factor").scalar<float>()));
    imager.set_stop_peak_factor(stop_peak_factor.value_or(load("stop_peak_factor").scalar<float>()));
    imager.set_stop_cycle_factor(stop_cycle_factor.value_or(load("stop_cycle_factor").scalar<float>()));
    imager.set_stop_sidelobe_level(stop_sidelobe_level.value_or(load("stop_sidelobe_level").scalar<float>()));
    imager.set_max_iteration(max_iter.value_or(load("max_iteration").scalar<int>()));
    imager.set_divergence_factor(divergence_factor.value_or(load("divergence_factor").scalar<float>()));
    imager.set_stall_threshold(stall_threshold.value_or(load("stall_threshold").scalar<float>()));
    imager.set_auto_mask(auto_mask.value_or(load("auto_mask").scalar<bool>()));
    imager.set_force_auto_mask(load("force_auto_mask").scalar<bool>());
    imager.set_auto_mask_peak_threshold(opt_finite(load("auto_mask_peak_threshold").scalar<float>()));
    imager.set_auto_mask_rms_threshold(opt_finite(load("auto_mask_rms_threshold").scalar<float>()));
  };

  // Wall time of one run() call, which writes the residual back into in.dirty.
  const auto timed_run = [&](ddmsc::Ddmsc& imager, cycle_inputs& in) {
    core::host_span3d<float> dirty(in.dirty.as_float32(), n_freq, nrow, ncol);
    core::host_span3d<const float> jones_norm(in.jones_norm.as_float32(), n_freq, nrow, ncol);
    core::host_span1d<const float> weights_freq(in.weights.as_float32(), n_freq);

    const auto t0 = std::chrono::steady_clock::now();
    ddmsc::ddmsc_result result = imager.run(dirty, jones_norm, weights_freq);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return std::pair<double, ddmsc::ddmsc_result>(ms, std::move(result));
  };

  // {min, median, mean, max}
  auto stats = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    const double median = n % 2 ? v.at(n / 2) : 0.5 * (v.at(n / 2 - 1) + v.at(n / 2));
    return std::array<double, 4>{v.front(), median, std::accumulate(v.begin(), v.end(), 0.0) / n, v.back()};
  };

  if (!std::filesystem::exists(cycle_dir(first_cycle))) {
    fprintf(stderr, "No cycle_%d/ in %s\n", first_cycle, dir.c_str());
    return 1;
  }

  if (bench) {
    const std::string cdir = cycle_dir(first_cycle);
    const int clean_iter = max_clean_iter.value_or(npy::load_npy(cdir + "max_sub_iteration.npy").scalar<int>());
    printf(
        "  bench: cycle %d, %d outer x %d clean iterations, %d runs, each after a 1-outer baseline (untimed warm-up "
        "first)\n",
        first_cycle, outer, clean_iter, runs);

    // Loaded once; every run() starts from this residual.
    cycle_inputs in = load_inputs(cdir);
    const std::vector<float> initial_dirty(in.dirty.as_float32(), in.dirty.as_float32() + in.dirty.size());

    // Times n_outer outer iterations on a fresh session, and whether they did exactly the pinned work.
    const auto bench_once = [&](int n_outer) {
      std::copy(initial_dirty.begin(), initial_dirty.end(), in.dirty.as_float32());
      const auto imager = make_imager();
      configure(*imager, cdir);
      imager->set_max_iteration(n_outer * clean_iter);
      auto [ms, result] = timed_run(*imager, in);
      const bool pinned = result.outer_iterations == n_outer &&
                          result.n_components() == static_cast<size_t>(n_outer) * static_cast<size_t>(clean_iter);
      if (!pinned)
        fprintf(stderr, "warning: expected %d outer x %d clean iterations, got %d outer and %zu components\n", n_outer,
                clean_iter, result.outer_iterations, result.n_components());
      return std::tuple<double, ddmsc::ddmsc_result, bool>(ms, std::move(result), pinned);
    };

    std::ofstream csv;
    if (!csv_path.empty()) {
      const bool new_file = !std::filesystem::exists(csv_path) || std::filesystem::file_size(csv_path) == 0;
      csv.open(csv_path, std::ios::app);
      if (!csv) {
        fprintf(stderr, "Cannot open %s\n", csv_path.c_str());
        return 1;
      }
      if (new_file)
        csv << "label,host,dump,cycle,nrow,ncol,scale,outer,clean_iter,run,total_ms,baseline_ms,per_outer_ms,"
               "components,outer_iterations,final_flux,final_rms,pinned\n";
    }
    std::array<char, 256> host{};
    gethostname(host.data(), host.size() - 1);

    // The process's first run() also pays one-off CUDA costs (context, module loading, cuFFT
    // init) that would inflate the first baseline; an untimed warm-up absorbs them.
    bench_once(1);

    std::vector<double> total_ms, baseline_ms, per_outer_ms;
    for (int run = 0; run < runs; ++run) {
      const auto [base_ms, base_result, base_pinned] = bench_once(1);
      const auto [ms, result, pinned] = bench_once(outer);
      const double per_outer = (ms - base_ms) / (outer - 1);
      total_ms.push_back(ms);
      baseline_ms.push_back(base_ms);
      per_outer_ms.push_back(per_outer);

      printf("run %d: %.3f ms, baseline %.3f ms, %.3f ms per outer (%zu components, %d outer, flux %.6g, rms %.6g)\n",
             run, ms, base_ms, per_outer, result.n_components(), result.outer_iterations, result.final_flux,
             result.final_rms);
      if (csv.is_open()) {
        csv << label << ',' << host.data() << ',' << dir << ',' << first_cycle << ',' << nrow << ',' << ncol << ','
            << *force_scale << ',' << outer << ',' << clean_iter << ',' << run << ',' << ms << ',' << base_ms << ','
            << per_outer << ',' << result.n_components() << ',' << result.outer_iterations << ',' << result.final_flux
            << ',' << result.final_rms << ',' << (pinned && base_pinned ? 1 : 0) << '\n';
        csv.flush();
      }
    }

    printf("Bench summary (ms):          min      median        mean         max\n");
    const std::array<std::pair<const char*, const std::vector<double>*>, 3> rows{
        {{"total", &total_ms}, {"baseline (1 outer)", &baseline_ms}, {"per outer", &per_outer_ms}}};
    for (const auto& [name, v] : rows) {
      const auto s = stats(*v);
      printf("  %-20s %11.3f %11.3f %11.3f %11.3f\n", name, s[0], s[1], s[2], s[3]);
    }
    if (csv.is_open()) printf("Appended %d rows to %s\n", runs, csv_path.c_str());
    return 0;
  }

  std::vector<double> run_ms;
  std::vector<std::vector<double>> cycle_ms;  // [cycle - first_cycle][run]
  for (int run = 0; run < runs; ++run) {
    const auto imager = make_imager();

    double total_ms = 0.0;
    for (int cycle = first_cycle;; ++cycle) {
      const std::string cdir = cycle_dir(cycle);
      if (!std::filesystem::exists(cdir)) break;

      cycle_inputs in = load_inputs(cdir);
      configure(*imager, cdir);
      const auto [ms, result] = timed_run(*imager, in);
      total_ms += ms;
      if (cycle_ms.size() <= static_cast<size_t>(cycle - first_cycle)) cycle_ms.emplace_back();
      cycle_ms.at(cycle - first_cycle).push_back(ms);

      if (run == 0)
        printf("cycle %d: %.3f ms, %zu components, %d outer, flux %.6g -> %.6g (stop %.6g), rms %.6g -> %.6g\n", cycle,
               ms, result.n_components(), result.outer_iterations, result.initial_flux, result.final_flux,
               result.stop_flux, result.initial_rms, result.final_rms);
      if (only_cycle >= 0) {
        break;
      }
    }
    printf("run %d: %.3f ms (%.3f s)\n", run, total_ms, total_ms / 1000.0);
    run_ms.push_back(total_ms);
  }

  const auto total = stats(run_ms);
  printf("Per-cycle summary (ms):\n");
  if (runs > 1) printf("  cycle         min      median        mean         max   %%total\n");
  for (size_t k = 0; k < cycle_ms.size(); ++k) {
    const auto c = stats(cycle_ms.at(k));
    const int cycle = first_cycle + static_cast<int>(k);
    if (runs > 1)
      printf("  %5d %11.3f %11.3f %11.3f %11.3f %7.1f%%\n", cycle, c[0], c[1], c[2], c[3], 100.0 * c[2] / total[2]);
    else
      printf("  cycle %3d %12.3f ms %6.1f%%\n", cycle, c[2], 100.0 * c[2] / total[2]);
  }
  if (runs > 1)
    printf("Over %d runs: min %.3f ms  median %.3f ms  mean %.3f ms  max %.3f ms\n", runs, total[0], total[1], total[2],
           total[3]);
  return 0;
}
