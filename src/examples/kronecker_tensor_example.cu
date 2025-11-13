#include "kronecker_tensor_example.cuh"

#include <cuda_runtime.h>

#include <cutensor.h>
#include <fast_deconv/core/mdarray.hpp>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#define HANDLE_ERROR(x)                                   \
  {                                                       \
    const auto err = x;                                   \
    if (err != CUTENSOR_STATUS_SUCCESS) {                 \
      printf("Error: %s\n", cutensorGetErrorString(err)); \
      exit(-1);                                           \
    }                                                     \
  };

void run_kronecker_tensor_example()
{
  std::printf("Running Tensor Kronecker example\n");

  // Host element type definition
  using floatTypeA       = float;
  using floatTypeB       = float;
  using floatTypeC       = float;
  using floatTypeCompute = float;

  // CUDA types
  cutensorDataType_t typeA                = CUTENSOR_R_32F;
  cutensorDataType_t typeB                = CUTENSOR_R_32F;
  cutensorDataType_t typeC                = CUTENSOR_R_32F;
  cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

  // Create vector of modes
  std::vector<int> modeC{'m', 'u', 'n', 'v'};
  std::vector<int> modeA{'m', 'h', 'k', 'n'};
  std::vector<int> modeB{'u', 'k', 'v', 'h'};
  int nmodeA = modeA.size();
  int nmodeB = modeB.size();
  int nmodeC = modeC.size();

  // Extents
  std::unordered_map<int, int64_t> extent;
  extent['m'] = 96;
  extent['n'] = 96;
  extent['u'] = 96;
  extent['v'] = 64;
  extent['h'] = 64;
  extent['k'] = 64;

  // Create a vector of extents for each tensor
  std::vector<int64_t> extentC;
  for (auto mode : modeC)
    extentC.push_back(extent[mode]);
  std::vector<int64_t> extentA;
  for (auto mode : modeA)
    extentA.push_back(extent[mode]);
  std::vector<int64_t> extentB;
  for (auto mode : modeB)
    extentB.push_back(extent[mode]);

  // Number of elements of each tensor
  size_t elementsA = 1;
  for (auto mode : modeA)
    elementsA *= extent[mode];
  size_t elementsB = 1;
  for (auto mode : modeB)
    elementsB *= extent[mode];
  size_t elementsC = 1;
  for (auto mode : modeC)
    elementsC *= extent[mode];

  // Allocate data
  auto A = fast_deconv::core::make_managed_mdarray<floatTypeA>(elementsA);
  auto B = fast_deconv::core::make_managed_mdarray<floatTypeB>(elementsB);
  auto C = fast_deconv::core::make_managed_mdarray<floatTypeC>(elementsC);

  // Initialize data
  fill_with_random(A.view());
  fill_with_random(B.view());
  fill_with_random(C.view());

  const uint32_t kAlignment = 128;  // Alignment of the global-memory device pointers (bytes)
  assert(uintptr_t(A_d) % kAlignment == 0);
  assert(uintptr_t(B_d) % kAlignment == 0);
  assert(uintptr_t(C_d) % kAlignment == 0);

  /*************************
   * cuTENSOR
   *************************/

  cutensorHandle_t handle;
  cutensorCreate(&handle);
  // HANDLE_ERROR();
  printf("Tet");

  // ====================================================================================
  // fast_deconv::core::stream_resources resources;
  // uint m     = 10;
  // uint n     = 10;
  // uint k     = 601;
  // uint p     = 601;
  // uint total = m * n * p * k;
  //
  // auto A     = fast_deconv::core::make_managed_mdarray<float>(m, n);
  // auto B     = fast_deconv::core::make_managed_mdarray<float>(k, p);
  // auto C_gpu = fast_deconv::core::make_managed_mdarray<float>(m * k, n * p);
  // auto C_cpu = fast_deconv::core::make_managed_mdarray<float>(m * k, n * p);
  //
  // fill_with_random(A.view());
  // fill_with_random(B.view());
  //
  // fast_deconv::util::run_benchmark([&]() -> void {
  //   fast_deconv::linalg::kronecker_async(resources,
  //                                        A.view().data_handle(),
  //                                        B.view().data_handle(),
  //                                        C_gpu.view().data_handle(),
  //                                        m,
  //                                        n,
  //                                        p,
  //                                        k);
  // });
  //
  // kron_cpu(A.view().data_handle(), B.view().data_handle(), C_cpu.view().data_handle(), m, n, k,
  // p);
  //
  // printf("Equals: %d\n", are_equals(C_gpu.view().data_handle(), C_cpu.view().data_handle(),
  // total));
}
