#include "examples/argmax_example.hpp"

#include <fmt/base.h>

int main(int argc, char** argv)
{
  fmt::println("Hello gpu world");

  fast_deconv::example::run_argmax();

  return 0;
}
