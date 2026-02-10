#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Element formatting (host side) - "Python-ish" style
// -----------------------------------------------------------------------------
template <typename T>
inline void format_element(std::ostream& os, const T& value)
{
  os << value;
}

// cuFloatComplex -> "(a+bj)" like Python complex
template <>
inline void format_element<cuFloatComplex>(std::ostream& os, const cuFloatComplex& z)
{
  float re = cuCrealf(z);
  float im = cuCimagf(z);
  os << "(" << re;
  if (im >= 0.f) os << "+";
  os << im << "j)";
}

// bool -> True / False, like Python
template <>
inline void format_element<bool>(std::ostream& os, const bool& b)
{
  os << (b ? "True" : "False");
}

// -----------------------------------------------------------------------------
// Host-side pretty printers (for already-host data)
// -----------------------------------------------------------------------------
template <typename T>
inline void print_host_vector(const T* data, std::size_t n, int indent = 0)
{
  std::string pad(indent, ' ');
  std::cout << pad << "[";

  for (std::size_t i = 0; i < n; ++i) {
    format_element(std::cout, data[i]);
    if (i + 1 < n) std::cout << ", ";
  }

  std::cout << "]";
}

template <typename T>
inline void print_host_matrix(const T* data, std::size_t rows, std::size_t cols, int indent = 0)
{
  std::string pad(indent, ' ');
  std::cout << pad << "[" << std::endl;

  for (std::size_t r = 0; r < rows; ++r) {
    std::cout << pad << "  [";
    for (std::size_t c = 0; c < cols; ++c) {
      const T& val = data[r * cols + c];  // row-major
      format_element(std::cout, val);
      if (c + 1 < cols) std::cout << ", ";
    }
    std::cout << "]";
    if (r + 1 < rows) std::cout << ",";
    std::cout << std::endl;
  }

  std::cout << pad << "]" << std::endl;
}

// -----------------------------------------------------------------------------
// Device-side wrappers: copy device -> host, then use host printers
// -----------------------------------------------------------------------------
template <typename T>
inline void print_device_vector(const T* d_ptr,
                                std::size_t n,
                                cudaStream_t stream = 0,
                                int indent          = 0)
{
  if (!d_ptr) { throw std::invalid_argument("print_device_vector: d_ptr is null"); }
  if (n == 0) {
    std::string pad(indent, ' ');
    std::cout << pad << "[]" << std::endl;
    return;
  }

  std::vector<T> host(n);

  cudaError_t err =
    cudaMemcpyAsync(host.data(), d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("print_device_vector: cudaMemcpyAsync failed: ") +
                             cudaGetErrorString(err));
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("print_device_vector: cudaStreamSynchronize failed: ") +
                             cudaGetErrorString(err));
  }

  print_host_vector(host.data(), n, indent);
  std::cout << std::endl;
}

template <typename T>
inline void print_device_matrix(
  const T* d_ptr, std::size_t rows, std::size_t cols, cudaStream_t stream = 0, int indent = 0)
{
  if (!d_ptr) { throw std::invalid_argument("print_device_matrix: d_ptr is null"); }
  if (rows == 0 || cols == 0) {
    std::string pad(indent, ' ');
    std::cout << pad << "[]" << std::endl;
    return;
  }

  std::size_t n = rows * cols;
  std::vector<T> host(n);

  cudaError_t err =
    cudaMemcpyAsync(host.data(), d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("print_device_matrix: cudaMemcpyAsync failed: ") +
                             cudaGetErrorString(err));
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("print_device_matrix: cudaStreamSynchronize failed: ") +
                             cudaGetErrorString(err));
  }

  print_host_matrix(host.data(), rows, cols, indent);
}
