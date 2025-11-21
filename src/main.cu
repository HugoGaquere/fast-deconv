#include <cstdio>

#include "examples/kronecker_example.cuh"
#include "examples/kronecker_mixed_product_example.cuh"
// #include "examples/kronecker_tensor_example.cuh"
#include "examples/wscms_example.cuh"


int main(int argc, char** argv)
{
  std::printf("Hello gpu world\n");

  run_kronecker_example();
  // run_kronecker_tensor();
  run_kronecker_mixed_product_example();


  return 0;
}
