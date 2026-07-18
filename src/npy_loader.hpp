#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

/**
 * Simple .npy file loader for test data.
 * Supports: float32, int32, bool, complex64
 */
struct NpyArray {
  std::vector<size_t> shape;
  std::string dtype;
  std::vector<std::byte> data;
  bool fortran_order = false;

  size_t size() const
  {
    size_t s = 1;
    for (auto d : shape) s *= d;
    return s;
  }

  size_t ndim() const { return shape.size(); }

  bool is_float32() const { return dtype.find("f4") != std::string::npos; }
  bool is_float64() const { return dtype.find("f8") != std::string::npos; }
  bool is_int32() const { return dtype.find("i4") != std::string::npos; }
  bool is_int64() const { return dtype.find("i8") != std::string::npos; }
  bool is_bool() const { return dtype.find("b1") != std::string::npos; }
  bool is_complex64() const { return dtype.find("c8") != std::string::npos; }

  template <typename T>
  T* as()
  {
    return reinterpret_cast<T*>(data.data());
  }

  template <typename T>
  const T* as() const
  {
    return reinterpret_cast<const T*>(data.data());
  }

  float* as_float32() { return as<float>(); }
  double* as_float64() { return as<double>(); }
  int32_t* as_int32() { return as<int32_t>(); }
  int64_t* as_int64() { return as<int64_t>(); }
  bool* as_bool() { return as<bool>(); }
  std::complex<float>* as_complex64() { return as<std::complex<float>>(); }

  const float* as_float32() const { return as<float>(); }
  const double* as_float64() const { return as<double>(); }
  const int32_t* as_int32() const { return as<int32_t>(); }
  const int64_t* as_int64() const { return as<int64_t>(); }
  const bool* as_bool() const { return as<bool>(); }
  const std::complex<float>* as_complex64() const { return as<std::complex<float>>(); }

  /// Read a scalar value, casting from the stored dtype to T.
  template <typename T>
  T scalar() const
  {
    if (is_float64()) return static_cast<T>(*as<double>());
    if (is_float32()) return static_cast<T>(*as<float>());
    if (is_int64()) return static_cast<T>(*as<int64_t>());
    if (is_int32()) return static_cast<T>(*as<int32_t>());
    if (is_bool()) return static_cast<T>(*as<bool>());
    throw std::runtime_error("scalar(): unsupported dtype " + dtype);
  }
};

inline NpyArray load_npy(const std::string& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + path);
  }

  // Read magic number
  char magic[6];
  file.read(magic, 6);
  if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
    throw std::runtime_error("Invalid .npy magic number");
  }

  // Read version
  uint8_t major, minor;
  file.read(reinterpret_cast<char*>(&major), 1);
  file.read(reinterpret_cast<char*>(&minor), 1);

  // Read header length
  uint32_t header_len;
  if (major == 1) {
    uint16_t len16;
    file.read(reinterpret_cast<char*>(&len16), 2);
    header_len = len16;
  } else {
    file.read(reinterpret_cast<char*>(&header_len), 4);
  }

  // Read header
  std::string header(header_len, '\0');
  file.read(&header[0], header_len);

  NpyArray arr;

  // Parse dtype
  auto descr_pos = header.find("'descr':");
  if (descr_pos != std::string::npos) {
    auto quote1 = header.find('\'', descr_pos + 8);
    auto quote2 = header.find('\'', quote1 + 1);
    arr.dtype = header.substr(quote1 + 1, quote2 - quote1 - 1);
  }

  // Parse fortran_order
  arr.fortran_order = header.find("'fortran_order': True") != std::string::npos;

  // Parse shape
  auto shape_pos = header.find("'shape':");
  if (shape_pos != std::string::npos) {
    auto paren1 = header.find('(', shape_pos);
    auto paren2 = header.find(')', paren1);
    std::string shape_str = header.substr(paren1 + 1, paren2 - paren1 - 1);

    size_t pos = 0;
    while (pos < shape_str.size()) {
      while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) pos++;
      if (pos >= shape_str.size()) break;
      size_t end = pos;
      while (end < shape_str.size() && shape_str[end] >= '0' && shape_str[end] <= '9') end++;
      if (end > pos) {
        arr.shape.push_back(std::stoull(shape_str.substr(pos, end - pos)));
      }
      pos = end;
    }
  }

  // Handle scalar (empty shape)
  if (arr.shape.empty()) {
    arr.shape.push_back(1);
  }

  // Determine element size
  size_t elem_size = 0;
  if (arr.dtype.find("f8") != std::string::npos) {
    elem_size = sizeof(double);
  } else if (arr.dtype.find("f4") != std::string::npos) {
    elem_size = sizeof(float);
  } else if (arr.dtype.find("i8") != std::string::npos) {
    elem_size = sizeof(int64_t);
  } else if (arr.dtype.find("i4") != std::string::npos) {
    elem_size = sizeof(int32_t);
  } else if (arr.dtype.find("b1") != std::string::npos) {
    elem_size = sizeof(bool);
  } else if (arr.dtype.find("c8") != std::string::npos) {
    elem_size = sizeof(std::complex<float>);
  } else {
    throw std::runtime_error("Unsupported dtype: " + arr.dtype);
  }

  // Read data
  size_t total_bytes = arr.size() * elem_size;
  arr.data.resize(total_bytes);
  file.read(reinterpret_cast<char*>(arr.data.data()), total_bytes);

  return arr;
}

}  // namespace npy
