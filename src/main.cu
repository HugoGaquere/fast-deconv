#include "examples/argmax_example.hpp"
#include "examples/subtract_example.hpp"
#include "examples/subtract_psf_from_dirty_example.hpp"

#include <fmt/base.h>

int main(int argc, char** argv) {
  fmt::println("Hello gpu world");

  // fast_deconv::example::run_argmax();
  // fast_deconv::example::run_subtract();
  fast_deconv::example::run_subtract_psf_from_dirty();

  return 0;
}
