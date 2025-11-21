// kron_cutensor.cu
#include "kronecker_tensor_example.cuh"

#include <cuda_runtime.h>

#include <cutensor.h>
#include <fast_deconv/util/kernel_bench.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define HANDLE_CUDA(err)                                                                          \
  do {                                                                                            \
    cudaError_t err__ = (err);                                                                    \
    if (err__ != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                                    \
    }                                                                                             \
  } while (0)

#define HANDLE_CUTENSOR(err)                  \
  do {                                        \
    cutensorStatus_t err__ = (err);           \
    if (err__ != CUTENSOR_STATUS_SUCCESS) {   \
      fprintf(stderr,                         \
              "cuTENSOR error %s at %s:%d\n", \
              cutensorGetErrorString(err__),  \
              __FILE__,                       \
              __LINE__);                      \
      std::exit(EXIT_FAILURE);                \
    }                                         \
  } while (0)

template <typename T>
bool are_equals(const T* A, const T* B, size_t size)
{
  for (int i = 0; i < size; i++) {
    if (A[i] != B[i]) return false;
  }
  return true;
}

void run_kronecker_tensor()
{
  using floatType = float;

  // ---- Dimensions ----
  const int64_t m = 100;  // rows of A
  const int64_t n = 100;  // cols of A
  const int64_t k = 601;  // rows of B
  const int64_t p = 601;  // cols of B

  const int64_t numA = m * n;
  const int64_t numB = k * p;
  const int64_t numD = m * k * n * p;  // 4D tensor D(i,r,j,s)
  const int64_t numC = numD;           // 2D tensor C((i*k+r),(n*j+s))

  // ---- Host buffers ----
  std::vector<floatType> hA(numA);
  std::vector<floatType> hB(numB);
  std::vector<floatType> hC(numC);  // will hold result

  // Simple initialization
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j)
      hA[i * n + j] = static_cast<floatType>(1 + i * n + j);

  for (int64_t r = 0; r < k; ++r)
    for (int64_t s = 0; s < p; ++s)
      hB[r * p + s] = static_cast<floatType>(1 + r * p + s);

  // ---- Device buffers ----
  floatType* dA = nullptr;
  floatType* dB = nullptr;
  floatType* dD = nullptr;  // 4D tensor D, same storage as 2D C

  HANDLE_CUDA(cudaMalloc(&dA, numA * sizeof(floatType)));
  HANDLE_CUDA(cudaMalloc(&dB, numB * sizeof(floatType)));
  HANDLE_CUDA(cudaMalloc(&dD, numD * sizeof(floatType)));

  HANDLE_CUDA(cudaMemcpy(dA, hA.data(), numA * sizeof(floatType), cudaMemcpyHostToDevice));
  HANDLE_CUDA(cudaMemcpy(dB, hB.data(), numB * sizeof(floatType), cudaMemcpyHostToDevice));

  // ---- cuTENSOR setup ----
  cutensorHandle_t handle;
  HANDLE_CUTENSOR(cutensorCreate(&handle));

  const cutensorDataType_t type                 = CUTENSOR_R_32F;
  const cutensorComputeDescriptor_t computeDesc = CUTENSOR_COMPUTE_DESC_32F;
  const uint32_t alignment                      = 256;

  // --- Tensor A: A[i,j], row-major ---
  int32_t modeA[2]   = {'i', 'j'};
  int64_t extentA[2] = {m, n};
  int64_t strideA[2] = {n, 1};  // row-major: (i,j) -> i*n + j

  cutensorTensorDescriptor_t descA;
  HANDLE_CUTENSOR(
    cutensorCreateTensorDescriptor(handle, &descA, 2, extentA, strideA, type, alignment));

  // --- Tensor B: B[r,s], row-major ---
  int32_t modeB[2]   = {'r', 's'};
  int64_t extentB[2] = {k, p};
  int64_t strideB[2] = {p, 1};  // (r,s) -> r*p + s

  cutensorTensorDescriptor_t descB;
  HANDLE_CUTENSOR(
    cutensorCreateTensorDescriptor(handle, &descB, 2, extentB, strideB, type, alignment));

  // --- Tensor D: D[i,r,j,s], row-major, but chosen so that
  //     linear index(i,r,j,s) == index in C[(i*k+r),(n*j+s)] ---
  //
  // extentsD = {m, k, n, p}
  // stridesD = {
  //   k*n*p,   // stride for i
  //   n*p,     // stride for r
  //   p,       // stride for j
  //   1        // stride for s
  // }
  //
  // Then:
  //   idxD = i*k*n*p + r*n*p + j*p + s
  //   idxC = (i*k + r)*(n*p) + j*p + s
  //        = i*k*n*p + r*n*p + j*p + s  (same!)
  //
  int32_t modeD[4]   = {'i', 'r', 'j', 's'};
  int64_t extentD[4] = {m, k, n, p};
  int64_t strideD[4] = {k * n * p, n * p, p, 1};

  cutensorTensorDescriptor_t descD;
  HANDLE_CUTENSOR(
    cutensorCreateTensorDescriptor(handle, &descD, 4, extentD, strideD, type, alignment));

  // We'll also use D as C in the alpha * A*B + beta*C formula
  // since we set beta = 0.
  cutensorTensorDescriptor_t descC = descD;
  int32_t* modeC                   = modeD;

  // ---- Operation descriptor: contraction with no contracted indices
  // D[i,r,j,s] = A[i,j] * B[r,s]
  // (C descriptor is only there for the "+ beta*C" part; we reuse D)
  cutensorOperationDescriptor_t opDesc;
  HANDLE_CUTENSOR(cutensorCreateContraction(handle,
                                            &opDesc,
                                            descA,
                                            modeA,
                                            CUTENSOR_OP_IDENTITY,
                                            descB,
                                            modeB,
                                            CUTENSOR_OP_IDENTITY,
                                            descC,
                                            modeC,
                                            CUTENSOR_OP_IDENTITY,
                                            descD,
                                            modeD,
                                            computeDesc));

  // Optional: check scalar type
  cutensorDataType_t scalarType;
  HANDLE_CUTENSOR(cutensorOperationDescriptorGetAttribute(
    handle, opDesc, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, &scalarType, sizeof(scalarType)));
  if (scalarType != CUTENSOR_R_32F) {
    fprintf(stderr, "Unexpected scalar type\n");
    std::exit(EXIT_FAILURE);
  }

  floatType alpha = 1.0f;
  floatType beta  = 0.0f;

  // ---- Plan & workspace ----
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
  cutensorPlanPreference_t planPref;
  HANDLE_CUTENSOR(cutensorCreatePlanPreference(handle, &planPref, algo, CUTENSOR_JIT_MODE_NONE));

  uint64_t workspaceSizeEstimate = 0;
  HANDLE_CUTENSOR(cutensorEstimateWorkspaceSize(
    handle, opDesc, planPref, CUTENSOR_WORKSPACE_DEFAULT, &workspaceSizeEstimate));

  cutensorPlan_t plan;
  HANDLE_CUTENSOR(cutensorCreatePlan(handle, &plan, opDesc, planPref, workspaceSizeEstimate));

  uint64_t actualWorkspaceSize = 0;
  HANDLE_CUTENSOR(cutensorPlanGetAttribute(handle,
                                           plan,
                                           CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                                           &actualWorkspaceSize,
                                           sizeof(actualWorkspaceSize)));

  void* dWorkspace = nullptr;
  if (actualWorkspaceSize > 0) { HANDLE_CUDA(cudaMalloc(&dWorkspace, actualWorkspaceSize)); }

  // ---- Execute contraction: D = alpha*A*B + beta*D ----
  cudaStream_t stream = nullptr;
  fast_deconv::util::run_benchmark(
    [&]() -> void {
      HANDLE_CUTENSOR(cutensorContract(handle,
                                       plan,
                                       &alpha,
                                       dA,
                                       dB,
                                       &beta,
                                       dD,  // C (input)  = D, but beta=0 so it's ignored
                                       dD,  // D (output) = D
                                       dWorkspace,
                                       actualWorkspaceSize,
                                       stream));
    },
    stream);

  HANDLE_CUDA(cudaStreamSynchronize(stream));

  // ---- Copy result back, view as C[(m*k) x (n*p)] ----
  HANDLE_CUDA(cudaMemcpy(hC.data(), dD, numC * sizeof(floatType), cudaMemcpyDeviceToHost));

  // Print input A
  // printf("A (%ld x %ld):\n", m, n);
  // for (int64_t i = 0; i < m; ++i) {
  //   for (int64_t j = 0; j < n; ++j) {
  //     printf("%6.1f ", hA[i * n + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // Print input B
  // printf("B (%ld x %ld):\n", k, p);
  // for (int64_t r = 0; r < k; ++r) {
  //   for (int64_t s = 0; s < p; ++s) {
  //     printf("%6.1f ", hB[r * p + s]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // Print Kronecker product C = A ⊗ B
  const int64_t rowsC = m * k;
  const int64_t colsC = n * p;

  // printf("Tensor: C = A ⊗ B  (%ld x %ld):\n", rowsC, colsC);
  // for (int64_t row = 0; row < rowsC; ++row) {
  //   for (int64_t col = 0; col < colsC; ++col) {
  //     printf("%6.1f ", hC[row * colsC + col]);
  //   }
  //   printf("\n");
  // }
  //
  // kron_cpu(hA.data(), hB.data(), hC.data(), m, n, k, p);
  // printf("CPU: C = A ⊗ B  (%ld x %ld):\n", rowsC, colsC);
  // for (int64_t row = 0; row < rowsC; ++row) {
  //   for (int64_t col = 0; col < colsC; ++col) {
  //     printf("%6.1f ", hC[row * colsC + col]);
  //   }
  //   printf("\n");
  // }

  // ---- Cleanup ----
  if (dWorkspace) cudaFree(dWorkspace);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dD);

  cutensorDestroyPlan(plan);
  cutensorDestroyPlanPreference(planPref);
  cutensorDestroyTensorDescriptor(descA);
  cutensorDestroyTensorDescriptor(descB);
  cutensorDestroyTensorDescriptor(descD);
  cutensorDestroyOperationDescriptor(opDesc);
  cutensorDestroy(handle);
}
