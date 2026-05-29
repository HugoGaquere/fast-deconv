#pragma once

#include <cuda_runtime.h>
#include <fmt/format.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

// Debug helper: write any mdspan (host or device) to a NumPy .npy file.
//
// Usage:
//   fast_deconv::util::dump_npy("residual.npy", mean_residual);
//
// In Python:
//   import numpy as np
//   import matplotlib.pyplot as plt
//   a = np.load("residual.npy")
//   plt.imshow(a); plt.colorbar(); plt.show()
//
// Supports layout_right (C-order) and layout_left (Fortran-order) mdspans.
// Strided / non-contiguous layouts are rejected at compile time.

namespace fast_deconv::util {

namespace detail {

template <typename T>
constexpr const char* npy_descr() noexcept
{
  if constexpr (std::is_same_v<T, float>) return "<f4";
  else if constexpr (std::is_same_v<T, double>) return "<f8";
  else if constexpr (std::is_same_v<T, std::int8_t>) return "|i1";
  else if constexpr (std::is_same_v<T, std::uint8_t>) return "|u1";
  else if constexpr (std::is_same_v<T, std::int16_t>) return "<i2";
  else if constexpr (std::is_same_v<T, std::uint16_t>) return "<u2";
  else if constexpr (std::is_same_v<T, std::int32_t>) return "<i4";
  else if constexpr (std::is_same_v<T, std::uint32_t>) return "<u4";
  else if constexpr (std::is_same_v<T, std::int64_t>) return "<i8";
  else if constexpr (std::is_same_v<T, std::uint64_t>) return "<u8";
  else if constexpr (std::is_same_v<T, bool>) return "|b1";
  else {
    static_assert(!sizeof(T*), "dump_npy: unsupported element type");
    return "";
  }
}

inline bool is_device_pointer(const void* ptr)
{
  cudaPointerAttributes attr{};
  const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
    cudaGetLastError();  // swallow sticky error from non-CUDA pointer
    return false;
  }
  return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
}

}  // namespace detail

template <core::cpts::mdspan M>
void dump_npy(std::string_view path, const M& span)
{
  static_assert(core::cpts::is_layout_right<M> || core::cpts::is_layout_left<M>,
                "dump_npy only supports layout_right or layout_left mdspans");

  using T = std::remove_const_t<typename M::element_type>;
  constexpr int R = M::rank();
  constexpr bool fortran_order = core::cpts::is_layout_left<M>;

  // Build the shape tuple "(d0, d1, ..., )". Trailing comma is valid Python.
  std::string shape_str = "(";
  std::size_t total = 1;
  for (int d = 0; d < R; ++d) {
    shape_str += fmt::format("{}, ", span.extent(d));
    total *= static_cast<std::size_t>(span.extent(d));
  }
  shape_str += ")";

  std::string header = fmt::format("{{'descr': '{}', 'fortran_order': {}, 'shape': {}, }}",
                                   detail::npy_descr<T>(), fortran_order ? "True" : "False",
                                   shape_str);

  // .npy v1.0: total of (magic[6] + version[2] + len[2] + header) must be a
  // multiple of 64. Header is space-padded and must end with '\n'.
  const std::size_t prefix = 10;
  const std::size_t unpadded = prefix + header.size() + 1;
  const std::size_t pad = (64 - (unpadded % 64)) % 64;
  header.append(pad, ' ');
  header.push_back('\n');
  const std::uint16_t header_len = static_cast<std::uint16_t>(header.size());

  std::ofstream out(std::string{path}, std::ios::binary);
  if (!out) throw std::runtime_error(fmt::format("dump_npy: cannot open '{}'", path));

  out.write("\x93NUMPY", 6);
  const char ver[2] = {1, 0};
  out.write(ver, 2);
  out.write(reinterpret_cast<const char*>(&header_len), 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));

  const std::size_t bytes = total * sizeof(T);
  if (bytes == 0) return;

  if (detail::is_device_pointer(span.data_handle())) {
    auto host = std::make_unique<T[]>(total);  // raw buffer — std::vector<bool> is bit-packed
    CHECK_CUDA(cudaMemcpy(host.get(), span.data_handle(), bytes, cudaMemcpyDeviceToHost));
    out.write(reinterpret_cast<const char*>(host.get()), static_cast<std::streamsize>(bytes));
  } else {
    out.write(reinterpret_cast<const char*>(span.data_handle()),
              static_cast<std::streamsize>(bytes));
  }
}

}  // namespace fast_deconv::util
