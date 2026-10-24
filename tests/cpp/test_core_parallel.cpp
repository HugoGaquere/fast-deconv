// Compiled by nvcc under the CUDA backend (see tests/cpp/CMakeLists.txt): it instantiates the primitives' kernels.
#include <gtest/gtest.h>

#include <cstdint>
#include <fast_deconv/core/macro.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/matrix/peak.hpp>
#include <fd_backend/core/for_each.hpp>
#include <fd_backend/core/reduce.hpp>
#include <vector>

#include "helpers/backend_test.hpp"

namespace core = fast_deconv::core;
namespace matrix = fast_deconv::matrix;
namespace fdtest = fast_deconv::test;

namespace {

// Functors, not lambdas: nvcc rejects __host__ __device__ lambdas inside gtest's private TestBody().

struct index_of {
  FD_HOST_DEVICE std::int64_t operator()(std::int64_t i) const { return i; }
};

struct plus {
  FD_HOST_DEVICE std::int64_t operator()(std::int64_t a, std::int64_t b) const { return a + b; }
};

struct pixel_peak {
  const float* data;
  FD_HOST_DEVICE matrix::peak operator()(std::int64_t i) const { return {.index = i, .value = data[i]}; }
};

// += rather than =: an index visited twice reads back doubled.
struct mark_1d {
  std::int64_t* out;
  FD_HOST_DEVICE void operator()(std::int64_t i) const { out[i] += i + 1; }
};

struct mark_2d {
  std::int64_t* out;
  std::int64_t ncol;
  FD_HOST_DEVICE void operator()(std::int64_t r, std::int64_t c) const { out[r * ncol + c] += r * ncol + c + 1; }
};

struct mark_3d {
  std::int64_t* out;
  std::int64_t nrow, ncol;
  FD_HOST_DEVICE void operator()(std::int64_t d, std::int64_t r, std::int64_t c) const
  {
    const std::int64_t i = (d * nrow + r) * ncol + c;
    out[i] += i + 1;
  }
};

// Runs @p visit on a zeroed buffer of @p n slots and checks slot i holds i + 1: every index once, none twice.
template <typename Visit>
void expect_each_index_once(const core::exec_ctx& ctx, std::int64_t n, Visit visit)
{
  auto d_out = ctx.alloc_mdcontainer<std::int64_t>(n);
  ctx.zero(d_out);
  visit(d_out.data_handle());
  std::vector<std::int64_t> out(static_cast<std::size_t>(n));
  ctx.copy(out.data(), d_out.data_handle(), n);
  ctx.wait();
  for (std::int64_t i = 0; i < n; i++) ASSERT_EQ(out.at(static_cast<std::size_t>(i)), i + 1) << "index=" << i;
}

}  // namespace

class CoreReduce : public fdtest::BackendTest {};

// sum of i over [0, n) is exact in int64, on both sides of the host's serial threshold. n = 0 is the identity.
TEST_F(CoreReduce, SumsEveryIndexOnce)
{
  const auto sr = res().make_ctx();
  for (const std::int64_t n : {std::int64_t{0}, std::int64_t{100}, std::int64_t{8193}, std::int64_t{1} << 20})
    EXPECT_EQ(core::reduce(sr, n, index_of{}, plus{}, std::int64_t{0}), n * (n - 1) / 2) << "n=" << n;
}

// Equal maxima far apart land in different threads' chunks and blocks; the combine still keeps the smallest index.
TEST_F(CoreReduce, TiedPeaksResolveToTheSmallestIndexAcrossChunks)
{
  constexpr std::int64_t kN = std::int64_t{1} << 17;
  std::vector<float> data(kN, 0.0f);
  for (const std::int64_t i : {std::int64_t{100000}, std::int64_t{5000}, kN - 1}) data.at(i) = 1.0f;

  const auto sr = res().make_ctx();
  const auto d_data = sr.stage(core::host_span1d<const float>(data.data(), kN));
  const auto p = core::reduce(sr, kN, pixel_peak{d_data.data_handle()}, matrix::peak_max{}, matrix::kPeakIdentity);
  EXPECT_EQ(p.index, 5000);
  EXPECT_FLOAT_EQ(p.value, 1.0f);
}

class CoreForEachIndex : public fdtest::BackendTest {};

// Sizes on both sides of the host's 64k fork threshold.
TEST_F(CoreForEachIndex, VisitsEveryIndexOnce1d)
{
  const auto sr = res().make_ctx();
  for (const std::int64_t n : {1000, 70000})
    expect_each_index_once(sr, n, [&](std::int64_t* out) { core::for_each_index(sr, n, mark_1d{out}); });
}

// 33 x 47 divides neither the 32 x 8 CUDA block nor anything else; 300 x 300 forks on host.
TEST_F(CoreForEachIndex, VisitsEveryIndexOnce2d)
{
  const auto sr = res().make_ctx();
  struct shape {
    std::int64_t nrow, ncol;
  };
  for (const shape s : {shape{33, 47}, shape{300, 300}})
    expect_each_index_once(sr, s.nrow * s.ncol,
                           [&](std::int64_t* out) { core::for_each_index(sr, s.nrow, s.ncol, mark_2d{out, s.ncol}); });
}

TEST_F(CoreForEachIndex, VisitsEveryIndexOnce3d)
{
  const auto sr = res().make_ctx();
  struct shape {
    std::int64_t depth, nrow, ncol;
  };
  for (const shape s : {shape{3, 33, 47}, shape{4, 200, 100}})
    expect_each_index_once(sr, s.depth * s.nrow * s.ncol, [&](std::int64_t* out) {
      core::for_each_index(sr, s.depth, s.nrow, s.ncol, mark_3d{out, s.nrow, s.ncol});
    });
}

// An empty range calls nothing: the null buffer would fault if it did.
TEST_F(CoreForEachIndex, EmptyRangesAreNoOps)
{
  const auto sr = res().make_ctx();
  core::for_each_index(sr, 0, mark_1d{nullptr});
  core::for_each_index(sr, 0, 5, mark_2d{nullptr, 5});
  core::for_each_index(sr, 5, 0, mark_2d{nullptr, 0});
  core::for_each_index(sr, 0, 4, 4, mark_3d{nullptr, 4, 4});
  core::for_each_index(sr, 2, 0, 4, mark_3d{nullptr, 0, 4});
  sr.wait();
}

#ifdef __CUDACC__
#include <fd_backend/core/launch.cuh>
#include <stdexcept>

namespace {
__global__ void noop_kernel() {}
}  // namespace

class CudaErrors : public fdtest::BackendTest {};

// A failed launch or runtime call raises instead of exiting the process, so a Python caller gets a RuntimeError.
// 2048 threads per block is over every device's limit; the error is not sticky, so the context still works after.
TEST_F(CudaErrors, BadLaunchAndRuntimeErrorsThrowAndLeaveTheContextUsable)
{
  const auto sr = res().make_ctx();
  EXPECT_THROW(core::launch(sr, core::launch_cfg::custom(1, 2048), noop_kernel), std::runtime_error);
  EXPECT_THROW(CHECK_CUDA(cudaErrorInvalidValue), std::runtime_error);

  expect_each_index_once(sr, 1000, [&](std::int64_t* out) { core::for_each_index(sr, 1000, mark_1d{out}); });
}
#endif
