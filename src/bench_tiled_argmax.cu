/// Benchmark + profiling driver for the tiled argmax (matrix::tiled_argmax_ctx::run and
/// run_incremental).
///
/// Motivation: in the DDMSC clean loop a minor iteration subtracts a PSF stamp
/// at the current peak, dirtying only a footprint-sized sub-region of the mean
/// dirty image. Today the loop reruns a full-image argmax every iteration; the
/// incremental argmax instead recomputes only the tiles overlapping the dirtied
/// footprint and re-combines them against the cached per-tile maxima.
///
/// The key question is whether the incremental approach beats the *production*
/// baseline, so this driver times three things on the same data:
///   - native   : cub::DeviceReduce::ArgMax over the whole image (matrix::argmax_ctx)
///                -- exactly what the clean loop runs today; the reference.
///   - full     : the tiled full pass (tiled_argmax_ctx::run).
///   - incr     : the tiled incremental pass (tiled_argmax_ctx::run_incremental).
/// Speedups in the CSV (speedup_incr, speedup_full) are taken against `native`,
/// not against the tiled full pass -- the tiled full pass craters for tiny tiles
/// and would otherwise report meaningless ratios.
///
/// The tile size is the key tuning knob and trades off three costs:
///   - small tiles  -> many tiles -> the single-block final combine rescans more
///                     cached slots every call, and more tiles fall inside the
///                     dirtied footprint;
///   - large tiles  -> each dirty tile recomputes more clean pixels (the footprint
///                     covers a smaller fraction of an edge tile -> more waste).
/// So the incremental cost is U-shaped in tile size. Pass --tiles to sweep it and
/// --csv to dump the curve for scripts/plot_tiled_argmax.py.
///
/// Both entry points run the same pipeline:
///   - tiled_argmax_reduce   (per-tile reduction; FULL pass uses the whole tile
///                            grid, the INCREMENTAL pass a small dirty sub-grid)
///   - cub::DeviceReduce     (device-wide combine over all cached per-tile maxima)
/// so under ncu the per-tile kernel for the two versions is distinguished by
/// launch (grid dims); the CUB reduce kernels follow each.
///
/// Pass --tiles and --psfs together for a 2D sweep: the CSV gets one row per
/// (footprint, tile) pair, so scripts/plot_tiled_argmax_2d.py can show the
/// optimal tile as a function of footprint size.
///
/// Usage:
///   bench_tiled_argmax [--size=N | --width=N --height=N]
///                      [--psf=N | --psfs=L] [--tile=N | --tiles=L]
///                      [--reps=N] [--warmup=N] [--device=N] [--seed=N]
///                      [--csv=PATH] [--validate]
///
///   --size       square image side (default 20000); --width/--height override.
///   --psf        single square footprint side (default 1700); ignored if --psfs set.
///   --psfs       comma-separated footprint sides to sweep, e.g. 256,512,1024,1700.
///   --tile       single square tile side (default 256); ignored if --tiles set.
///   --tiles      comma-separated tile sides to sweep, e.g. 64,128,256,512,1024.
///   --reps       timed repetitions per phase (default 50).
///   --warmup     untimed warmup repetitions per phase (default 5).
///   --csv        write one row per (footprint, tile) config to this path.
///   --validate   plant known peaks and cross-check both results (off by default;
///                skip it for clean profiling runs).
///
/// Nsight Compute (single launch of each phase, one tile size):
///   ncu --set full ./bench_tiled_argmax --tile=256 --reps=1 --warmup=0

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

namespace core = fast_deconv::core;
namespace matrix = fast_deconv::matrix;

#define BENCH_CHECK_CUDA(call)                                                                            \
  do {                                                                                                    \
    cudaError_t _e = (call);                                                                              \
    if (_e != cudaSuccess) {                                                                              \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(_e)); \
      std::exit(1);                                                                                       \
    }                                                                                                     \
  } while (0)

// ------------------------------------------------------------------------- //
//  Device data fill (avoids a multi-GB host buffer for large images)
// ------------------------------------------------------------------------- //

// SplitMix64-style hash mapped to [0, 1): deterministic per index, so the image
// content is reproducible across runs without uploading anything from the host.
__global__ void fill_pattern(float* data, uint64_t n, uint64_t seed)
{
  uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    uint64_t x = (i + seed) * 0x9E3779B97F4A7C15ull;
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 32;
    data[i] = static_cast<float>(x >> 40) * (1.0f / 16777216.0f);  // 24 bits -> [0,1)
  }
}

static void fill_image(float* d_data, uint64_t npix, uint64_t seed, const core::exec_ctx& sr)
{
  fill_pattern<<<1024, 256, 0, sr.cuda_stream>>>(d_data, npix, seed);
  BENCH_CHECK_CUDA(cudaGetLastError());
  sr.sync();
}

// ------------------------------------------------------------------------- //
//  CLI
// ------------------------------------------------------------------------- //

struct options {
  int width = 20000;
  int height = 20000;
  int psf = 1700;
  int tile = 256;
  std::vector<int> tiles;  // empty -> single {tile}
  std::vector<int> psfs;   // empty -> single {psf}
  int reps = 50;
  int warmup = 5;
  int device = 0;
  uint64_t seed = 1234;
  std::string csv_path;
  bool validate = false;
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

// ------------------------------------------------------------------------- //
//  Timing
// ------------------------------------------------------------------------- //

struct timing {
  double mean_ms = 0, min_ms = 0, max_ms = 0;
};

// Time @p reps blocking calls of @p fn (warmup excluded). The argmax entry points
// synchronize internally, so wall-clock around each call measures the full GPU
// pass plus the result read-back -- exactly the per-iteration cost the clean loop
// pays.
template <typename Fn>
static timing time_phase(int warmup, int reps, Fn&& fn)
{
  for (int i = 0; i < warmup; ++i) fn();

  timing t;
  t.min_ms = 1e300;
  double acc = 0.0;
  for (int i = 0; i < reps; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    acc += ms;
    t.min_ms = std::min(t.min_ms, ms);
    t.max_ms = std::max(t.max_ms, ms);
  }
  t.mean_ms = reps > 0 ? acc / reps : 0.0;
  return t;
}

// ------------------------------------------------------------------------- //
//  Per-tile benchmark
// ------------------------------------------------------------------------- //

struct tile_result {
  int tile = 0;
  int psf = 0;
  int n_tiles_x = 0, n_tiles_y = 0;
  long long n_tiles = 0;
  long long dirty_tiles = 0;  // tiles the incremental pass recomputes
  timing full, incr;
};

// Run the full + incremental phases for one (tile, psf) pair over the shared image.
static tile_result run_tile(const core::exec_ctx& sr, float* d_data, const options& opt, int tile, int psf)
{
  tile_result tr;
  tr.tile = tile;
  tr.psf = psf;
  tr.n_tiles_x = (opt.width + tile - 1) / tile;
  tr.n_tiles_y = (opt.height + tile - 1) / tile;
  tr.n_tiles = static_cast<long long>(tr.n_tiles_x) * tr.n_tiles_y;

  matrix::tiled_argmax_ctx ws{sr, core::dims<2>(opt.height, opt.width), tile};
  const core::span2d<float> view(d_data, opt.height, opt.width);

  // The incremental pass dirties a footprint centered here; fixed across reps.
  const int peak_row = opt.height / 2;
  const int peak_col = opt.width / 2;

  // Real dirty tile rectangle, mirroring argmax_incremental's mapping, so the CSV
  // records the actual count of recomputed tiles (not the worst-case bound).
  const int px0 = std::max(peak_col - psf / 2, 0);
  const int py0 = std::max(peak_row - psf / 2, 0);
  const int px1 = std::min(peak_col - psf / 2 + psf, opt.width);
  const int py1 = std::min(peak_row - psf / 2 + psf, opt.height);
  const int tx0 = px0 / tile, ty0 = py0 / tile;
  const int tx1 = (px1 - 1) / tile, ty1 = (py1 - 1) / tile;
  tr.dirty_tiles = static_cast<long long>(tx1 - tx0 + 1) * (ty1 - ty0 + 1);

  // Phase 1 (full) also seeds every tile, which argmax_incremental requires.
  tr.full = time_phase(opt.warmup, opt.reps, [&] { ws.run(view); });
  tr.incr = time_phase(opt.warmup, opt.reps, [&] { ws.run_incremental(view, peak_row, peak_col, psf, psf); });

  sr.sync();  // ws releases its device buffers in its destructor
  return tr;
}

// ------------------------------------------------------------------------- //
//  CSV
// ------------------------------------------------------------------------- //

// speedup_incr / speedup_full are measured against the native cub::DeviceReduce
// ::ArgMax baseline (the full-image argmax the clean loop runs today), NOT against
// the same-row tiled full pass -- the latter craters for tiny tiles and would
// report meaningless "speedups". native_* is constant across the sweep.
// Smallest reseed interval N at which the amortized tiled cost (full/N + incr)
// drops to the native baseline, i.e. native = full/N + incr -> N = full/(native -
// incr). Returns -1 when incr >= native (the tiled approach never beats native, no
// matter how many incremental calls amortize the seeding full pass).
static double breakeven_n(const timing& native, const tile_result& r)
{
  const double denom = native.mean_ms - r.incr.mean_ms;
  return denom > 0.0 ? r.full.mean_ms / denom : -1.0;
}

static const char* CSV_HEADER =
    "width,height,psf,tile,n_tiles_x,n_tiles_y,n_tiles,dirty_tiles,reps,warmup,"
    "native_mean_ms,native_min_ms,native_max_ms,"
    "full_mean_ms,full_min_ms,full_max_ms,"
    "incr_mean_ms,incr_min_ms,incr_max_ms,speedup_incr,speedup_full,breakeven_n\n";

static void write_csv_row(std::ostream& os, const options& opt, const timing& native, const tile_result& r)
{
  const double su_incr = r.incr.mean_ms > 0 ? native.mean_ms / r.incr.mean_ms : 0.0;
  const double su_full = r.full.mean_ms > 0 ? native.mean_ms / r.full.mean_ms : 0.0;
  os << opt.width << ',' << opt.height << ',' << r.psf << ',' << r.tile << ',' << r.n_tiles_x << ',' << r.n_tiles_y
     << ',' << r.n_tiles << ',' << r.dirty_tiles << ',' << opt.reps << ',' << opt.warmup << ',' << native.mean_ms << ','
     << native.min_ms << ',' << native.max_ms << ',' << r.full.mean_ms << ',' << r.full.min_ms << ',' << r.full.max_ms
     << ',' << r.incr.mean_ms << ',' << r.incr.min_ms << ',' << r.incr.max_ms << ',' << su_incr << ',' << su_full << ','
     << breakeven_n(native, r) << '\n';
}

// ------------------------------------------------------------------------- //
//  Optional correctness cross-check
// ------------------------------------------------------------------------- //

static void validate(const core::exec_ctx& sr, float* d_data, const options& opt, int tile, int psf)
{
  const int peak_row = opt.height / 2, peak_col = opt.width / 2;

  matrix::tiled_argmax_ctx ws{sr, core::dims<2>(opt.height, opt.width), tile};
  const core::span2d<float> view(d_data, opt.height, opt.width);

  // Plant a unique global peak (data is in [0,1)); the full pass must find it.
  const int gidx = peak_row * opt.width + peak_col;
  const float big = 5.0f;
  BENCH_CHECK_CUDA(cudaMemcpyAsync(d_data + gidx, &big, sizeof(float), cudaMemcpyHostToDevice, sr.cuda_stream));
  auto [fv, fi] = ws.run(view);
  printf("  [validate] FULL %s: peak %.3f at %lld (want %.3f at %d)\n", (fv == big && fi == gidx) ? "ok" : "MISMATCH",
         fv, static_cast<long long>(fi), big, gidx);

  // Dirty the footprint: plant an even bigger peak inside it, refresh incrementally.
  const int bidx = (peak_row + 3) * opt.width + (peak_col + 5);
  const float bigger = 9.0f;
  BENCH_CHECK_CUDA(cudaMemcpyAsync(d_data + bidx, &bigger, sizeof(float), cudaMemcpyHostToDevice, sr.cuda_stream));
  auto [iv, ii] = ws.run_incremental(view, peak_row, peak_col, psf, psf);
  printf("  [validate] INCREMENTAL %s: peak %.3f at %lld (want %.3f at %d)\n",
         (iv == bigger && ii == bidx) ? "ok" : "MISMATCH", iv, static_cast<long long>(ii), bigger, bidx);

  sr.sync();  // ws releases its device buffers in its destructor
  fill_image(d_data, static_cast<uint64_t>(opt.width) * opt.height, opt.seed, sr);  // restore
}

// ------------------------------------------------------------------------- //
//  Main
// ------------------------------------------------------------------------- //

int main(int argc, char** argv)
{
  options opt;
  auto usage = [&]() {
    fprintf(stderr,
            "Usage: %s [--size=N | --width=N --height=N] [--psf=N] [--tile=N | --tiles=L]\n"
            "          [--reps=N] [--warmup=N] [--device=N] [--seed=N] [--csv=PATH] [--validate]\n",
            argv[0]);
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto val = [&](const char* pfx) { return a.substr(std::strlen(pfx)); };
    if (a.rfind("--size=", 0) == 0) {
      opt.width = opt.height = std::atoi(val("--size=").c_str());
    } else if (a.rfind("--width=", 0) == 0)
      opt.width = std::atoi(val("--width=").c_str());
    else if (a.rfind("--height=", 0) == 0)
      opt.height = std::atoi(val("--height=").c_str());
    else if (a.rfind("--psfs=", 0) == 0)
      opt.psfs = parse_int_list(val("--psfs="));
    else if (a.rfind("--psf=", 0) == 0)
      opt.psf = std::atoi(val("--psf=").c_str());
    else if (a.rfind("--tiles=", 0) == 0)
      opt.tiles = parse_int_list(val("--tiles="));
    else if (a.rfind("--tile=", 0) == 0)
      opt.tile = std::atoi(val("--tile=").c_str());
    else if (a.rfind("--reps=", 0) == 0)
      opt.reps = std::atoi(val("--reps=").c_str());
    else if (a.rfind("--warmup=", 0) == 0)
      opt.warmup = std::atoi(val("--warmup=").c_str());
    else if (a.rfind("--device=", 0) == 0)
      opt.device = std::atoi(val("--device=").c_str());
    else if (a.rfind("--seed=", 0) == 0)
      opt.seed = std::strtoull(val("--seed=").c_str(), nullptr, 10);
    else if (a.rfind("--csv=", 0) == 0)
      opt.csv_path = val("--csv=");
    else if (a == "--validate")
      opt.validate = true;
    else {
      fprintf(stderr, "Unrecognized argument: %s\n", a.c_str());
      usage();
      return 1;
    }
  }
  if (opt.width <= 0 || opt.height <= 0) {
    fprintf(stderr, "width/height must be positive\n");
    return 1;
  }
  if (opt.reps < 1) opt.reps = 1;
  if (opt.warmup < 0) opt.warmup = 0;

  std::vector<int> tiles = opt.tiles.empty() ? std::vector<int>{opt.tile} : opt.tiles;
  std::vector<int> psfs = opt.psfs.empty() ? std::vector<int>{opt.psf} : opt.psfs;
  for (int t : tiles)
    if (t <= 0) {
      fprintf(stderr, "tile sizes must be positive\n");
      return 1;
    }
  for (int p : psfs)
    if (p <= 0 || p > std::min(opt.width, opt.height)) {
      fprintf(stderr, "psf sizes must be in (0, min(width,height)]\n");
      return 1;
    }
  std::sort(tiles.begin(), tiles.end());
  std::sort(psfs.begin(), psfs.end());

  BENCH_CHECK_CUDA(cudaSetDevice(opt.device));

  const uint64_t npix = static_cast<uint64_t>(opt.width) * opt.height;
  const double img_mb = npix * sizeof(float) / (1024.0 * 1024.0);

  auto print_list = [](const char* label, const std::vector<int>& v) {
    printf("  %-10s : ", label);
    for (std::size_t i = 0; i < v.size(); ++i) printf("%d%s", v[i], i + 1 < v.size() ? "," : "");
    printf("\n");
  };

  printf("==================== tiled argmax bench ====================\n");
  printf("  image      : %d x %d  (%.0f Mpix, %.2f MiB)\n", opt.width, opt.height, npix / 1e6, img_mb);
  print_list("footprints", psfs);
  print_list("tiles", tiles);
  printf("  reps       : %d timed, %d warmup  (%zu x %zu = %zu configs)\n", opt.reps, opt.warmup, psfs.size(),
         tiles.size(), psfs.size() * tiles.size());
  printf("============================================================\n\n");

  // OOM pre-check with an actionable message: the image is by far the biggest
  // allocation (tile buffers are O(n_tiles) and tiny next to it), so a
  // small slack factor over the image bytes is enough.
  {
    size_t free_b = 0, total_b = 0;
    BENCH_CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
    const double need = static_cast<double>(npix) * sizeof(float) * 1.1;
    if (need > static_cast<double>(free_b)) {
      fprintf(stderr,
              "Image %dx%d needs ~%.0f MiB but only %.0f MiB of device memory is free; "
              "re-run with a smaller --size.\n",
              opt.width, opt.height, need / (1024.0 * 1024.0), free_b / (1024.0 * 1024.0));
      return 1;
    }
  }

  core::exec_resources res(opt.device);
  const core::exec_ctx sr = res.make_stream();

  float* d_data = sr.alloc_async<float>(npix);
  fill_image(d_data, npix, opt.seed, sr);

  if (opt.validate) {
    validate(sr, d_data, opt, tiles.front(), psfs.front());
    printf("\n");
  }

  // Native full-image argmax baseline: cub::DeviceReduce::ArgMax over every pixel
  // -- exactly what the clean loop runs today, so it is THE reference the
  // incremental approach must beat. Independent of tile/psf, so measured once.
  timing native;
  {
    matrix::argmax_ctx native_ws{sr, npix};
    native =
        time_phase(opt.warmup, opt.reps, [&] { native_ws.run(core::span2d<float>(d_data, opt.height, opt.width)); });
  }
  printf("native argmax (cub ArgMax over %.0f Mpix): mean %.4f ms  min %.4f  (%.1f GiB/s)\n\n", npix / 1e6,
         native.mean_ms, native.min_ms, native.mean_ms > 0 ? img_mb / 1024.0 / (native.mean_ms / 1000.0) : 0.0);

  std::ofstream csv;
  if (!opt.csv_path.empty()) {
    csv.open(opt.csv_path);
    if (!csv) {
      fprintf(stderr, "Failed to open %s for writing\n", opt.csv_path.c_str());
      return 1;
    }
    csv << CSV_HEADER;
  }

  // incr_vs_native = native/incr (the N->inf speedup); breakeven_N = reseed
  // interval at which the amortized tiled cost first beats native ("never" if it
  // can't, regardless of N).
  printf("%-7s %-7s %12s %12s | %10s %10s %10s | %14s %10s\n", "psf", "tile", "n_tiles", "dirty_tiles", "native_ms",
         "full_ms", "incr_ms", "incr_vs_native", "breakeven_N");
  for (int p : psfs) {
    for (int t : tiles) {
      const tile_result r = run_tile(sr, d_data, opt, t, p);
      const double su_incr = r.incr.mean_ms > 0 ? native.mean_ms / r.incr.mean_ms : 0.0;
      const double ben = breakeven_n(native, r);
      printf("%-7d %-7d %12lld %12lld | %10.4f %10.4f %10.4f | %13.1fx ", r.psf, r.tile, r.n_tiles, r.dirty_tiles,
             native.mean_ms, r.full.mean_ms, r.incr.mean_ms, su_incr);
      if (ben < 0.0)
        printf("%10s\n", "never");
      else
        printf("%10.1f\n", ben);
      fflush(stdout);
      if (csv.is_open()) {
        write_csv_row(csv, opt, native, r);
        csv.flush();
      }
    }
  }

  if (csv.is_open()) printf("\nWrote %s\n", opt.csv_path.c_str());

  sr.free_async(d_data);
  sr.sync();
  return 0;
}
