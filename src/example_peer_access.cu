/// Example: probe peer-access capability and bandwidth between every visible
/// CUDA device. Useful before deciding whether splitting work across GPUs is
/// worth it (NVLink fast, PCIe slow).
///
/// Usage:  example_peer_access [bytes_per_copy] [n_iter]
///
/// Defaults to 1 GiB per copy, 20 iterations. The benchmark allocates `bytes`
/// on each device of the pair, copies n_iter times, and prints GB/s.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fast_deconv/util/peer_access.hpp>

namespace util = fast_deconv::util;

int main(int argc, char** argv)
{
  size_t bytes = 1ULL << 30;  // 1 GiB
  int n_iter = 20;

  if (argc >= 2) bytes = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
  if (argc >= 3) n_iter = std::atoi(argv[2]);

  int n_devices = 0;
  cudaError_t err = cudaGetDeviceCount(&n_devices);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  if (n_devices < 2) {
    fprintf(stderr, "Only %d CUDA device(s) visible — peer access requires at least 2.\n", n_devices);
    return 1;
  }

  printf("Probing peer access across %d device(s) — %.1f MB per copy x %d iters\n", n_devices,
         bytes / (1024.0 * 1024.0), n_iter);
  for (int d = 0; d < n_devices; d++) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, d);
    printf("  device %d: %s (%d.%d, %.1f GB)\n", d, prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  }

  printf("\n--- Capability matrix (cudaDeviceCanAccessPeer) ---\n");
  printf("       ");
  for (int j = 0; j < n_devices; j++) printf("  d%-2d", j);
  printf("\n");
  for (int i = 0; i < n_devices; i++) {
    printf("  d%-2d  ", i);
    for (int j = 0; j < n_devices; j++) {
      if (i == j) printf("   - ");
      else printf("   %c ", util::can_peer_access(i, j) ? 'Y' : '.');
    }
    printf("\n");
  }

  printf("\n--- Bandwidth (one direction at a time) ---\n");
  for (int i = 0; i < n_devices; i++) {
    for (int j = 0; j < n_devices; j++) {
      if (i == j) continue;
      util::probe_peer_link(i, j, bytes, n_iter);
    }
  }

  return 0;
}
