// #include "examples/argmax_example.hpp"
// #include "examples/subtract_example.hpp"
// #include "examples/subtract_psf_from_dirty_example.hpp"
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/kernel_traits.hpp>
#include <fast_deconv/core/dispatcher.hpp>
#include <fast_deconv/core/access_policy.hpp>

#include <fmt/base.h>

int main(int argc, char** argv) {
  fmt::println("Hello gpu world");

  // fast_deconv::example::run_argmax();
  // fast_deconv::example::run_subtract();
  // fast_deconv::example::run_subtract_psf_from_dirty();

  return 0;
}
