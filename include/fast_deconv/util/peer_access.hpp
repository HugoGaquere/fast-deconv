#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::util {

/// Hardware capability check: can device @p src reach device @p dst via peer access
/// (NVLink, PCIe peer, etc.)? Does NOT enable access; just queries the link.
inline bool can_peer_access(int src_device, int dst_device)
{
  if (src_device == dst_device) return true;
  int can = 0;
  CHECK_CUDA(cudaDeviceCanAccessPeer(&can, src_device, dst_device));
  return can != 0;
}

/// Enable peer access from @p src_device to @p dst_device (one-way).
/// Returns true if access is now possible (newly enabled or already enabled).
/// Idempotent: safe to call multiple times.
inline bool enable_peer_access(int src_device, int dst_device)
{
  if (src_device == dst_device) return true;
  if (!can_peer_access(src_device, dst_device)) return false;

  int prev = -1;
  cudaGetDevice(&prev);
  cudaSetDevice(src_device);

  cudaError_t err = cudaDeviceEnablePeerAccess(dst_device, 0);
  bool ok = (err == cudaSuccess) || (err == cudaErrorPeerAccessAlreadyEnabled);
  if (!ok) {
    fprintf(stderr, "cudaDeviceEnablePeerAccess(%d -> %d) failed: %s\n", src_device, dst_device,
            cudaGetErrorString(err));
  } else if (err == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();  // clear sticky error so later checks don't trip
  }

  if (prev >= 0) cudaSetDevice(prev);
  return ok;
}

struct peer_bandwidth_result {
  size_t bytes_per_copy = 0;
  int n_iterations = 0;
  float total_ms = 0.0f;       // wall time across all timed copies
  float avg_ms = 0.0f;         // per-copy time
  double bandwidth_gbps = 0.0; // bytes_per_copy / avg_ms
  bool ok = false;             // false if peer access wasn't available
};

/// Benchmark cudaMemcpyPeerAsync bandwidth from @p src_device to @p dst_device.
/// Allocates @p bytes on each device, does a warm-up copy, then times @p n_iter copies.
/// Caller must have already enabled peer access (or use probe_peer_link below).
inline peer_bandwidth_result benchmark_peer_copy(int src_device, int dst_device, size_t bytes, int n_iter = 10)
{
  peer_bandwidth_result r{};
  r.bytes_per_copy = bytes;
  r.n_iterations = n_iter;

  int prev = -1;
  cudaGetDevice(&prev);

  // src buffer on src_device
  CHECK_CUDA(cudaSetDevice(src_device));
  void* src_ptr = nullptr;
  CHECK_CUDA(cudaMalloc(&src_ptr, bytes));

  // dst buffer on dst_device
  CHECK_CUDA(cudaSetDevice(dst_device));
  void* dst_ptr = nullptr;
  CHECK_CUDA(cudaMalloc(&dst_ptr, bytes));

  // Stream + events on src_device (the stream's device must match the calling device for the copy)
  CHECK_CUDA(cudaSetDevice(src_device));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // Warm up — first peer copy can be slower (lazy mapping).
  CHECK_CUDA(cudaMemcpyPeerAsync(dst_ptr, dst_device, src_ptr, src_device, bytes, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Timed copies
  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < n_iter; i++) {
    CHECK_CUDA(cudaMemcpyPeerAsync(dst_ptr, dst_device, src_ptr, src_device, bytes, stream));
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  CHECK_CUDA(cudaEventElapsedTime(&r.total_ms, start, stop));
  r.avg_ms = r.total_ms / n_iter;
  r.bandwidth_gbps = (static_cast<double>(bytes) / 1e9) / (r.avg_ms / 1000.0);
  r.ok = true;

  // Cleanup
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaSetDevice(src_device));
  CHECK_CUDA(cudaFree(src_ptr));
  CHECK_CUDA(cudaSetDevice(dst_device));
  CHECK_CUDA(cudaFree(dst_ptr));
  if (prev >= 0) cudaSetDevice(prev);

  return r;
}

/// Pretty-print a benchmark result.
inline void print_peer_bandwidth(int src_device, int dst_device, const peer_bandwidth_result& r)
{
  if (!r.ok) {
    fprintf(stderr, "[peer] device %d -> %d: link unavailable\n", src_device, dst_device);
    return;
  }
  fprintf(stderr, "[peer] device %d -> %d: %.2f GB/s  (%.3f ms/copy, %.1f MB x %d iters)\n", src_device, dst_device,
          r.bandwidth_gbps, r.avg_ms, r.bytes_per_copy / (1024.0 * 1024.0), r.n_iterations);
}

/// One-shot: capability check + enable + benchmark + print. Returns the result so
/// callers can also branch on bandwidth.
inline peer_bandwidth_result probe_peer_link(int src_device, int dst_device, size_t bytes, int n_iter = 10)
{
  peer_bandwidth_result r{};
  if (!can_peer_access(src_device, dst_device)) {
    fprintf(stderr, "[peer] device %d cannot access device %d (no peer link)\n", src_device, dst_device);
    return r;
  }
  if (!enable_peer_access(src_device, dst_device)) {
    fprintf(stderr, "[peer] failed to enable %d -> %d\n", src_device, dst_device);
    return r;
  }
  r = benchmark_peer_copy(src_device, dst_device, bytes, n_iter);
  print_peer_bandwidth(src_device, dst_device, r);
  return r;
}

}  // namespace fast_deconv::util
