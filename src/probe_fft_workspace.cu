#include <array>
#include <cmath>
#include <cstdio>
#include <cufft.h>

#define CUFFT_CHECK(x)                                                    \
  do {                                                                    \
    cufftResult r = (x);                                                  \
    if (r != CUFFT_SUCCESS) {                                             \
      std::fprintf(stderr, "cuFFT error %d at line %d\n", r, __LINE__);   \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

static double gib(size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

static size_t plan_work(cufftType type, int n0, int n1, int batch)
{
  cufftHandle plan;
  CUFFT_CHECK(cufftCreate(&plan));
  CUFFT_CHECK(cufftSetAutoAllocation(plan, 0));
  std::array<int, 2> fft_size{n0, n1};
  size_t work = 0;
  CUFFT_CHECK(cufftMakePlanMany(plan, 2, fft_size.data(), nullptr, 1, 0, nullptr, 1, 0, type, batch, &work));
  CUFFT_CHECK(cufftDestroy(plan));
  return work;
}

static int next_fast_size(int n)
{
  static constexpr int radices[] = {2, 3, 5, 7};
  while (true) {
    int m = n;
    for (int r : radices) while (m % r == 0) m /= r;
    if (m == 1) return n;
    ++n;
  }
}

int main(int argc, char** argv)
{
  int input_n = 19845;
  float padding = 1.7f;
  if (argc >= 2) input_n = std::atoi(argv[1]);
  if (argc >= 3) padding = static_cast<float>(std::atof(argv[2]));

  const int npad_raw = static_cast<int>(std::ceil((padding - 1.0f) * input_n / 2.0f));
  const int padded_raw = input_n + 2 * npad_raw;
  const int padded_smooth = next_fast_size(padded_raw);

  auto report = [&](const char* label, int padded) {
    const int freq_n = padded / 2 + 1;
    const size_t spatial_bytes = static_cast<size_t>(padded) * padded * sizeof(float);
    const size_t freq_bytes = static_cast<size_t>(padded) * freq_n * sizeof(float) * 2;

    std::printf("====================================================================\n");
    std::printf("%s  padded %d  freq %dx%d\n", label, padded, padded, freq_n);
    std::printf("per-slice spatial buffer  : %.3f GiB\n", gib(spatial_bytes));
    std::printf("per-slice freq buffer     : %.3f GiB\n\n", gib(freq_bytes));

    for (int batch : {1, 2, 3, 4, 5}) {
      size_t fwd = plan_work(CUFFT_R2C, padded, padded, batch);
      size_t bwd = plan_work(CUFFT_C2R, padded, padded, batch);
      std::printf("batch=%d  R2C work %.4f GiB  C2R work %.4f GiB\n", batch, gib(fwd), gib(bwd));
    }

    std::printf("\n-- Mean-dirty convolve totals (forward batch=1, backward batch=B) --\n");
    size_t fwd1 = plan_work(CUFFT_R2C, padded, padded, 1);
    for (int B : {1, 2, 3, 4, 5}) {
      size_t bwdB = plan_work(CUFFT_C2R, padded, padded, B);
      size_t shared_work = std::max(fwd1, bwdB);
      size_t buffers = spatial_bytes + freq_bytes + static_cast<size_t>(B) * freq_bytes
                       + static_cast<size_t>(B) * spatial_bytes;
      std::printf("B=%d  buffers %.2f GiB  shared cuFFT work %.2f GiB  TOTAL %.2f GiB\n", B, gib(buffers),
                  gib(shared_work), gib(buffers + shared_work));
    }
    std::printf("\n");
  };

  std::printf("input %d  padding %.4f  -> padded_raw %d  padded_smooth %d  (gap %.3f%%)\n\n", input_n, padding,
              padded_raw, padded_smooth,
              100.0 * (padded_smooth - padded_raw) / static_cast<double>(padded_raw));

  report("RAW   ", padded_raw);
  report("SMOOTH", padded_smooth);
  return 0;
}
