/// Example: launch run_ddmsc_cycles with real data dumped from DDFacet.
///
/// Usage:  example_ddmsc <dump_dir> [--cycles=LIST] [--device=N] [--history=DIR] [--csv=PATH]
///
/// `--cycles` selects which `cycle_<N>/` subdirectories to load and run, as
/// a comma-separated list with optional ranges (e.g. `1,2,4-6`). Cycles
/// execute in the order given on a single shared context: the auto-mask
/// history accumulated by DDMSC carries over between cycles, mirroring the
/// real DDFacet flow. Default: `1`. The deprecated `--cycle=N` flag is
/// accepted as a synonym for a single-cycle spec.
/// `--device` selects the CUDA device (default: 0). The example calls
/// cudaSetDevice on that device before any cudaMalloc/cudaMemcpy so the host
/// uploads target the right GPU; ddmsc::context propagates the same id to
/// core::resources and its streams, which are bound to that device.
/// `--csv` writes per-cycle stats (timing, component count, etc.) to a CSV
/// file. Use scripts/plot_cycle_timing.py to chart the output.
/// `--force-auto-mask-last` forces auto-masking on the last cycle of the set,
/// overriding that cycle's dumped `force_auto_mask` flag. Earlier cycles keep
/// their dumped value.
/// `--max-iter=N` / `--max-clean-iter=N` override the dumped iteration caps:
/// total minor iterations (hence the number of outer scale selections) and
/// inner clean iterations per scale selection, respectively. Used to bound a
/// profiling run (nsys/ncu) to a short, deterministic slice of the cycle; omit
/// them to run the dump's full schedule.
///
/// The dump directory must contain `init/` and `cycle_<N>/` subdirectories
/// produced by FastDDFacet's dump_ref utility (set DUMP_REF=<dir> when
/// running DDF.py). `init/` holds the one-time setup (raw PSFs, scale
/// kernels, Ddmsc ctor inputs, auto-masking thresholds). `cycle_<N>/` holds
/// the per-cycle inputs and runtime parameters (dirty, jones_norm,
/// weights, mask, stop limits, etc.).

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fast_deconv/algorithm/ddmsc_cycles.hpp>
#include <fast_deconv/algorithm/ddmsc_types.hpp>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/util/dump.hpp>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "npy_loader.hpp"

namespace core = fast_deconv::core;
namespace scale = fast_deconv::scale;
namespace ddmsc = fast_deconv::algorithm::ddmsc;

/// Helper: allocate device memory and copy host data into it. Exits with a
/// diagnostic on failure -- a silent OOM here would feed garbage pointers into
/// the run and corrupt the timings/results downstream.
template <typename T>
T* device_upload(const T* host_ptr, std::size_t count)
{
  T* ptr = nullptr;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
  return ptr;
}

/// Treat a NaN scalar from the dump as "field unset" (nullopt).
static std::optional<float> opt_finite(float v) { return std::isnan(v) ? std::nullopt : std::optional<float>(v); }

/// Parse a cycle spec like "1,2,4-6" into an ordered list of cycle ids.
/// Returns an empty vector for malformed input (caller treats as error).
static std::vector<int> parse_cycles(const std::string& spec)
{
  std::vector<int> out;
  std::size_t i = 0;
  while (i <= spec.size()) {
    const std::size_t j = spec.find(',', i);
    const std::string tok = spec.substr(i, j == std::string::npos ? std::string::npos : j - i);
    if (tok.empty()) return {};
    const std::size_t dash = tok.find('-');
    if (dash == std::string::npos) {
      out.push_back(std::atoi(tok.c_str()));
    } else {
      const int lo = std::atoi(tok.substr(0, dash).c_str());
      const int hi = std::atoi(tok.substr(dash + 1).c_str());
      if (lo <= hi) {
        for (int k = lo; k <= hi; ++k) out.push_back(k);
      } else {
        for (int k = lo; k >= hi; --k) out.push_back(k);
      }
    }
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return out;
}

struct cycle_stat {
  int cycle_id;
  double elapsed_ms;
  std::size_t n_components;
  int total_iterations;
  float final_flux;
  float stop_flux;
};

int main(int argc, char** argv)
{
  auto usage = [&]() {
    fprintf(stderr,
            "Usage: %s <dump_dir> [--cycles=LIST] [--device=N] [--history=DIR] [--csv=PATH] "
            "[--dump-result=DIR] [--force-auto-mask-last] [--max-iter=N] [--max-clean-iter=N]\n",
            argv[0]);
  };

  std::string dir;
  std::string history_dir;
  std::string csv_path;
  std::string dump_result_dir;
  std::string cycles_spec = "1";
  int device_id = 0;
  bool force_auto_mask_last = false;
  int max_iter_override = -1;        // <0: keep the dumped max_iteration
  int max_clean_iter_override = -1;  // <0: keep the dumped max_clean_iteration
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a == "--force-auto-mask-last") {
      force_auto_mask_last = true;
    } else if (a.rfind("--cycles=", 0) == 0) {
      cycles_spec = a.substr(9);
    } else if (a.rfind("--cycle=", 0) == 0) {
      // Back-compat: singular flag accepted as a single-cycle spec.
      cycles_spec = a.substr(8);
    } else if (a.rfind("--device=", 0) == 0) {
      device_id = std::atoi(a.c_str() + 9);
    } else if (a.rfind("--history=", 0) == 0) {
      history_dir = a.substr(10);
    } else if (a.rfind("--csv=", 0) == 0) {
      csv_path = a.substr(6);
    } else if (a.rfind("--dump-result=", 0) == 0) {
      dump_result_dir = a.substr(14);
    } else if (a.rfind("--max-iter=", 0) == 0) {
      max_iter_override = std::atoi(a.c_str() + 11);
    } else if (a.rfind("--max-clean-iter=", 0) == 0) {
      max_clean_iter_override = std::atoi(a.c_str() + 17);
    } else if (!a.empty() && a[0] != '-' && dir.empty()) {
      dir = a;
    } else {
      fprintf(stderr, "Unrecognized argument: %s\n", argv[i]);
      usage();
      return 1;
    }
  }
  if (dir.empty()) {
    usage();
    return 1;
  }

  const std::vector<int> cycle_ids = parse_cycles(cycles_spec);
  if (cycle_ids.empty()) {
    fprintf(stderr, "Could not parse --cycles=%s\n", cycles_spec.c_str());
    usage();
    return 1;
  }

  // Bind this thread to the requested device so the raw cudaMalloc/cudaMemcpy
  // calls below target it. ddmsc::context will pass the same id down to
  // core::resources and its streams, which are bound to the same device.
  cudaError_t set_err = cudaSetDevice(device_id);
  if (set_err != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(set_err));
    return 1;
  }

  auto load_init = [&](const char* name) { return npy::load_npy(dir + "/init/" + name + ".npy"); };
  auto load_cycle = [&](int cid, const char* name) {
    return npy::load_npy(dir + "/cycle_" + std::to_string(cid) + "/" + name + ".npy");
  };

  // ----- init/ : one-time setup -----
  auto npy_raw_psfs = load_init("raw_psfs");
  auto npy_xdes = load_init("xdes");
  auto npy_scale_sigmas = load_init("scale_sigmas");
  auto npy_scale_bias = load_init("scale_bias");
  auto npy_map_pixel = load_init("map_pixel_facet");
  auto npy_fft_padding = load_init("fft_padding");
  auto npy_gamma = load_init("gamma");
  auto npy_auto_mask_peak_th = load_init("auto_mask_peak_threshold");
  auto npy_auto_mask_rms_th = load_init("auto_mask_rms_threshold");

  // We need the first listed cycle's dirty + mask up front: dirty to size the
  // FFT plans baked into the context, mask to provide a valid view to the
  // context constructor. Subsequent cycles reuse the same context, hot-
  // swapping the mask via ctx.workspace.mask and feeding new params.
  const int first_cid = cycle_ids.front();
  auto npy_dirty0 = load_cycle(first_cid, "dirty");
  auto npy_mask0 = load_cycle(first_cid, "mask");

  // ----- Extract constant dimensions from loaded shapes -----
  // dump_ref squeezes arrays, removing the stokes=1 dimension.
  // raw_psfs: (n_facet, n_freq, psf_nrow, psf_ncol)
  const int n_facet = static_cast<int>(npy_raw_psfs.shape[0]);
  const int n_freq = static_cast<int>(npy_raw_psfs.shape[1]);
  const int psf_nrow = static_cast<int>(npy_raw_psfs.shape[2]);
  const int psf_ncol = static_cast<int>(npy_raw_psfs.shape[3]);

  // dirty: (n_freq, nrow, ncol) — these dims must be the same across all
  // requested cycles (FFT plans are built once).
  const int nrow = static_cast<int>(npy_dirty0.shape[1]);
  const int ncol = static_cast<int>(npy_dirty0.shape[2]);

  const int n_scales = static_cast<int>(npy_scale_sigmas.shape[0]);
  const int n_order = static_cast<int>(npy_xdes.shape[1]);

  printf("Loaded dump from %s (cycles=%s, device=%d)\n", dir.c_str(), cycles_spec.c_str(), device_id);
  printf("  image: %dx%d  psf: %dx%d  freq: %d  order: %d  facets: %d  scales: %d  cycles_to_run: %zu\n", nrow, ncol,
         psf_nrow, psf_ncol, n_freq, n_order, n_facet, n_scales, cycle_ids.size());

  // ----- Upload constant (init) arrays to device -----
  float* d_raw_psfs = device_upload(npy_raw_psfs.as_float32(), npy_raw_psfs.size());
  float* d_xdes = device_upload(npy_xdes.as_float32(), npy_xdes.size());
  float* d_scale_sig = device_upload(npy_scale_sigmas.as_float32(), npy_scale_sigmas.size());
  // First cycle's mask is uploaded up front so the context ctor receives a
  // valid span. d_mask_current then tracks whichever mask buffer is currently
  // bound to ctx.workspace.mask; cycles after the first free the previous
  // buffer and upload a fresh one.
  bool* d_mask_current = device_upload(npy_mask0.as_bool(), npy_mask0.size());

  // Host arrays (no device upload)
  float* h_scale_bias = npy_scale_bias.as_float32();
  int* h_map_pixel = npy_map_pixel.as_int32();

  // ----- Build mdspan views for constant inputs -----
  core::device_span4d<float> raw_psfs(d_raw_psfs, n_facet, n_freq, psf_nrow, psf_ncol);
  core::device_span2d<float> xdes(d_xdes, n_freq, n_order);
  core::device_span2d<bool> mask0(d_mask_current, static_cast<int>(npy_mask0.shape[0]),
                                  static_cast<int>(npy_mask0.shape[1]));
  core::device_vect<float> scale_sigmas(d_scale_sig, n_scales);
  core::host_vect<float> scale_bias(h_scale_bias, n_scales);
  core::host_span2d<int> map_pixel_facet(h_map_pixel, nrow, ncol);

  const float fft_padding = npy_fft_padding.scalar<float>();

  // ----- Build DDMSC context (resources + workspace + FFT plans) -----
  ddmsc::context ctx(device_id, raw_psfs, xdes, mask0, scale_sigmas, scale_bias, map_pixel_facet, nrow, ncol, n_freq,
                     fft_padding);

  // ----- Optionally seed auto-mask history from a previous DicoModel -----
  // Seeded once before the cycle loop; DDMSC appends to this history during
  // each cycle, so subsequent cycles see the accumulated components.
  if (!history_dir.empty()) {
    auto npy_hist_coords = npy::load_npy(history_dir + "/historical_peak_coords.npy");
    auto npy_hist_scales = npy::load_npy(history_dir + "/historical_scales.npy");
    const int n_hist = static_cast<int>(npy_hist_scales.size());
    const int* hc = npy_hist_coords.as_int32();
    const int* hs = npy_hist_scales.as_int32();
    auto& wsr = ctx.workspace;
    wsr.historical_peak_coords.reserve(n_hist);
    wsr.historical_scales.reserve(n_hist);
    int n_kept = 0;
    for (int i = 0; i < n_hist; ++i) {
      const int r = hc[2 * i + 0];
      const int c = hc[2 * i + 1];
      if (r < 0 || r >= nrow || c < 0 || c >= ncol) continue;
      if (hs[i] < 0 || hs[i] >= n_scales) continue;
      wsr.historical_peak_coords.emplace_back(r, c);
      wsr.historical_scales.push_back(hs[i]);
      ++n_kept;
    }
    printf("Seeded auto-mask history from %s: %d/%d components in-bounds\n", history_dir.c_str(), n_kept, n_hist);
  }

  // ----- Cycle loop -----
  std::vector<cycle_stat> stats;
  stats.reserve(cycle_ids.size());
  double total_ms = 0.0;

  for (std::size_t idx = 0; idx < cycle_ids.size(); ++idx) {
    const int cid = cycle_ids[idx];
    printf("\n===== cycle %d (%zu/%zu) =====\n", cid, idx + 1, cycle_ids.size());

    // Reuse the pre-loaded cycle-0 dirty/mask on the first iteration to avoid
    // a redundant disk read.
    auto npy_dirty = (idx == 0) ? std::move(npy_dirty0) : load_cycle(cid, "dirty");
    auto npy_jones_norm = load_cycle(cid, "jones_norm");
    auto npy_weights_freq = load_cycle(cid, "weights_freq");
    auto npy_mask = (idx == 0) ? std::move(npy_mask0) : load_cycle(cid, "mask");

    auto npy_max_iteration = load_cycle(cid, "max_iteration");
    auto npy_max_sub_iter = load_cycle(cid, "max_sub_iteration");
    auto npy_divergence = load_cycle(cid, "divergence_factor");
    auto npy_flux_threshold = load_cycle(cid, "flux_threshold");
    auto npy_rms_factor = load_cycle(cid, "stop_rms_factor");
    auto npy_stop_peak_factor = load_cycle(cid, "stop_peak_factor");
    auto npy_cycle_factor = load_cycle(cid, "stop_cycle_factor");
    auto npy_sidelobe_level = load_cycle(cid, "stop_sidelobe_level");
    auto npy_stall = load_cycle(cid, "stall_threshold");
    auto npy_peak_factor = load_cycle(cid, "peak_factor");
    auto npy_clean_negative = load_cycle(cid, "clean_negative");
    auto npy_force_auto_mask = load_cycle(cid, "force_auto_mask");

    // Dirty shape must match the FFT plans baked into ctx.
    if (static_cast<int>(npy_dirty.shape[0]) != n_freq || static_cast<int>(npy_dirty.shape[1]) != nrow ||
        static_cast<int>(npy_dirty.shape[2]) != ncol) {
      fprintf(stderr, "Cycle %d dirty shape (%zu,%zu,%zu) differs from cycle %d's (%d,%d,%d) -- aborting\n", cid,
              npy_dirty.shape[0], npy_dirty.shape[1], npy_dirty.shape[2], first_cid, n_freq, nrow, ncol);
      return 1;
    }

    // Per-cycle device uploads. The mask is special-cased so iter 0 reuses
    // the buffer already uploaded for the context ctor.
    float* d_dirty = device_upload(npy_dirty.as_float32(), npy_dirty.size());
    float* d_jones_norm = device_upload(npy_jones_norm.as_float32(), npy_jones_norm.size());
    float* d_weights = device_upload(npy_weights_freq.as_float32(), npy_weights_freq.size());
    if (idx > 0) {
      cudaFree(d_mask_current);
      d_mask_current = device_upload(npy_mask.as_bool(), npy_mask.size());
    }

    const int mask_nrow = static_cast<int>(npy_mask.shape[0]);
    const int mask_ncol = static_cast<int>(npy_mask.shape[1]);

    core::device_span3d<float> dirty(d_dirty, n_freq, nrow, ncol);
    core::device_span3d<float> jones_norm(d_jones_norm, n_freq, nrow, ncol);
    core::device_vect<float> weights_freq(d_weights, n_freq);
    core::device_span2d<bool> mask_cycle(d_mask_current, mask_nrow, mask_ncol);

    // Hot-swap the mask for this cycle.
    ctx.workspace.mask = mask_cycle;

    // --force-auto-mask-last forces auto-masking on the final cycle of the
    // set regardless of the dump's per-cycle force_auto_mask flag.
    const bool is_last_cycle = idx + 1 == cycle_ids.size();
    const bool force_auto_mask = npy_force_auto_mask.scalar<bool>() || (force_auto_mask_last && is_last_cycle);

    ddmsc::params params{
        .max_iteration = npy_max_iteration.scalar<int>(),
        .divergence_factor = npy_divergence.scalar<float>(),
        .flux_threshold = npy_flux_threshold.scalar<float>(),
        .stop_rms_factor = npy_rms_factor.scalar<float>(),
        .stop_peak_factor = npy_stop_peak_factor.scalar<float>(),
        .stop_cycle_factor = npy_cycle_factor.scalar<float>(),
        .stop_sidelobe_level = npy_sidelobe_level.scalar<float>(),
        .clean_negative = npy_clean_negative.scalar<bool>(),
        .peak_factor = npy_peak_factor.scalar<float>(),
        .gamma = npy_gamma.scalar<float>(),
        .max_clean_iteration = npy_max_sub_iter.scalar<int>(),
        .scale_stall_threshold = npy_stall.scalar<float>(),
        .enable_auto_mask = true,
        .force_enable_auto_mask = force_auto_mask,
        .auto_mask_peak_threshold = opt_finite(npy_auto_mask_peak_th.scalar<float>()),
        .auto_mask_rms_threshold = opt_finite(npy_auto_mask_rms_th.scalar<float>()),
    };

    // Profiling/debug overrides: bound the work so a profiler sees a short,
    // deterministic slice. max_clean_iteration caps inner (minor) iterations
    // per scale selection; max_iteration caps total minor iterations, hence the
    // number of outer scale selections. Real stop thresholds may still end the
    // cycle earlier -- these only ever shorten it.
    if (max_iter_override > 0) params.max_iteration = max_iter_override;
    if (max_clean_iter_override > 0) params.max_clean_iteration = max_clean_iter_override;
    if (max_iter_override > 0 || max_clean_iter_override > 0)
      printf("  iteration overrides: max_iteration=%d max_clean_iteration=%d\n", params.max_iteration,
             params.max_clean_iteration);

    printf("Running DDMSC on %dx%d image, mask: %dx%d, %d scales, %d freq, %d facets...\n", nrow, ncol, mask_nrow,
           mask_ncol, n_scales, n_freq, n_facet);

    const auto t_start = std::chrono::steady_clock::now();
    ddmsc::ddmsc_result result = ddmsc::run_ddmsc_cycles(ctx, params, dirty, jones_norm, weights_freq);
    // Checked sync: an async kernel failure must not be recorded as a valid
    // (and absurdly fast) cycle timing.
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    stats.push_back(
        {cid, elapsed_ms, result.peak_coords.size(), result.total_iterations, result.final_flux, result.stop_flux});
    total_ms += elapsed_ms;

    printf("Cycle %d done in %.3f ms (%.3f s). %zu components (scales=%zu gains=%zu coeffs=%zu)\n", cid, elapsed_ms,
           elapsed_ms / 1000.0, result.peak_coords.size(), result.scales.size(), result.gains.size(),
           result.coeffs.size());

    const std::size_t n_print = std::min<std::size_t>(5, result.peak_coords.size());
    for (std::size_t i = 0; i < n_print; ++i) {
      printf("  [%zu] (row=%d, col=%d) scale=%d gain=%.6f coeffs=[", i, result.peak_coords[i].first,
             result.peak_coords[i].second, result.scales[i], result.gains[i]);
      for (std::size_t k = 0; k < result.coeffs[i].size(); ++k) {
        printf("%s%.6f", k == 0 ? "" : ", ", result.coeffs[i][k]);
      }
      printf("]\n");
    }

    // Optional result dump for the fidelity comparison: the mutated dirty is
    // the per-frequency residual after this cycle; components go to a CSV in
    // the same format the DDFacet reference exports are expected to use.
    // Dump failures (disk full, bad path) only warn: the timings collected so
    // far are the primary product and must still reach the --csv output.
    if (!dump_result_dir.empty()) {
      try {
        const std::string cycle_dir = dump_result_dir + "/cycle_" + std::to_string(cid);
        std::filesystem::create_directories(cycle_dir);
        fast_deconv::util::dump_npy(cycle_dir + "/residual.npy", dirty);
        std::ofstream comp(cycle_dir + "/components.csv");
        if (!comp) throw std::runtime_error("cannot open " + cycle_dir + "/components.csv");
        const std::size_t comp_n_order = result.coeffs.empty() ? 0 : result.coeffs.front().size();
        comp << "row,col,scale,gain";
        for (std::size_t k = 0; k < comp_n_order; ++k) comp << ",coeff" << k;
        comp << '\n';
        for (std::size_t i = 0; i < result.peak_coords.size(); ++i) {
          comp << result.peak_coords.at(i).first << ',' << result.peak_coords.at(i).second << ',' << result.scales.at(i)
               << ',' << result.gains.at(i);
          // coeffs are filled by a separate path than the component lists and
          // may legitimately be shorter; emit only what exists for this row.
          if (i < result.coeffs.size())
            for (float cval : result.coeffs.at(i)) comp << ',' << cval;
          comp << '\n';
        }
        printf("Dumped residual + %zu components to %s\n", result.peak_coords.size(), cycle_dir.c_str());
      } catch (const std::exception& e) {
        fprintf(stderr, "Warning: --dump-result failed for cycle %d: %s\n", cid, e.what());
      }
    }

    cudaFree(d_dirty);
    cudaFree(d_jones_norm);
    cudaFree(d_weights);
  }

  // ----- Summary -----
  printf("\n===== summary =====\n");
  for (const auto& s : stats) {
    printf("  cycle %d: %.3f ms (%.3f s)  components=%zu  iters=%d\n", s.cycle_id, s.elapsed_ms, s.elapsed_ms / 1000.0,
           s.n_components, s.total_iterations);
  }
  printf("Total: %.3f ms (%.3f s) across %zu cycle(s)\n", total_ms, total_ms / 1000.0, stats.size());

  if (!csv_path.empty()) {
    std::ofstream csv(csv_path);
    if (!csv) {
      fprintf(stderr, "Failed to open %s for writing\n", csv_path.c_str());
      return 1;
    }
    csv << "cycle_id,elapsed_ms,n_components,total_iterations,final_flux,stop_flux\n";
    for (const auto& s : stats) {
      csv << s.cycle_id << ',' << s.elapsed_ms << ',' << s.n_components << ',' << s.total_iterations << ','
          << s.final_flux << ',' << s.stop_flux << '\n';
    }
    printf("Wrote %s\n", csv_path.c_str());
  }

  // ----- Cleanup -----
  cudaFree(d_mask_current);
  cudaFree(d_raw_psfs);
  cudaFree(d_xdes);
  cudaFree(d_scale_sig);

  return 0;
}
